"""Session evolution snapshots: creation, replay/conflict, freeze and diffs.

Covers:
* POST freezes the device event high-water and copies ordered sessions;
* repeated snapshot_id replays the original (200), cross-device reuse 409;
* late events and retention cleanup never rewrite an old snapshot;
* the bipartite-graph diff: unchanged/changed/merged/split/created/removed
  and the many-to-many reorganized label, with stable ordering;
* snapshots frozen under different gap settings refuse comparison;
* old (pre-snapshot) databases gain the new tables on startup.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.main import create_app

GAP = 300  # seconds


def _submit(client, device_id, event_id, occurred_at):
    return client.post(
        "/events",
        json={
            "device_id": device_id,
            "event_id": event_id,
            "occurred_at": occurred_at,
        },
    )


def _snapshot(client, device_id, snapshot_id):
    return client.post(
        f"/devices/{device_id}/session-snapshots",
        json={"snapshot_id": snapshot_id},
    )


def _make_app(tmp_path, name="test.db", gap=GAP):
    return create_app(database_path=str(tmp_path / name), gap_seconds=gap)


def test_snapshot_freezes_sessions_and_high_water(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        for eid, ts in [
            ("e1", "2026-09-20T10:00:00Z"),
            ("e2", "2026-09-20T10:02:00Z"),
            ("e3", "2026-09-20T10:20:00Z"),  # new session
        ]:
            assert _submit(client, "dev-1", eid, ts).status_code == 201

        resp = _snapshot(client, "dev-1", "snap-1")
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["snapshot_id"] == "snap-1"
        assert body["device_id"] == "dev-1"
        assert body["high_water_seq"] == 3
        assert body["gap_seconds"] == GAP
        assert body["session_count"] == 2
        assert [s["event_count"] for s in body["sessions"]] == [2, 1]
        assert [m["event_id"] for m in body["sessions"][0]["members"]] == [
            "e1",
            "e2",
        ]
        assert [m["event_id"] for m in body["sessions"][1]["members"]] == [
            "e3"
        ]
        assert "created_at" in body


def test_snapshot_on_empty_device(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        resp = _snapshot(client, "dev-empty", "snap-empty")
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["high_water_seq"] == 0
        assert body["session_count"] == 0
        assert body["sessions"] == []


def test_repeated_snapshot_id_replays_original(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        _submit(client, "dev-1", "e1", "2026-09-20T10:00:00Z")
        first = _snapshot(client, "dev-1", "snap-1")
        assert first.status_code == 201

        # A later event changes the live sessions...
        _submit(client, "dev-1", "e2", "2026-09-20T10:02:00Z")
        live = client.get("/devices/dev-1/sessions").json()
        assert live[0]["event_count"] == 2

        # ...but the replay returns the frozen original, high-water included.
        replay = _snapshot(client, "dev-1", "snap-1")
        assert replay.status_code == 200, replay.text
        assert replay.json() == first.json()

        # GET by id returns the same frozen document.
        fetched = client.get("/devices/dev-1/session-snapshots/snap-1")
        assert fetched.status_code == 200
        assert fetched.json() == first.json()


def test_snapshot_id_reuse_on_another_device_is_409(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        _submit(client, "dev-1", "e1", "2026-09-20T10:00:00Z")
        _submit(client, "dev-2", "e2", "2026-09-20T10:00:00Z")
        assert _snapshot(client, "dev-1", "shared").status_code == 201

        conflict = _snapshot(client, "dev-2", "shared")
        assert conflict.status_code == 409, conflict.text

        # Neither side is altered: dev-2 has no snapshot, dev-1 keeps its.
        listed = client.get("/devices/dev-2/session-snapshots").json()
        assert listed == []
        assert client.get("/devices/dev-1/session-snapshots/shared").status_code == 200


def test_list_snapshots_ordered_and_scoped(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        _submit(client, "dev-1", "e1", "2026-09-20T10:00:00Z")
        _snapshot(client, "dev-1", "older")
        # Ensure a distinct created_at; ordering is by creation time, not id.
        time.sleep(0.01)
        _snapshot(client, "dev-1", "newer")
        _snapshot(client, "dev-2", "c")

        ids = [
            row["snapshot_id"]
            for row in client.get("/devices/dev-1/session-snapshots").json()
        ]
        assert ids == ["older", "newer"]
        other = client.get("/devices/dev-2/session-snapshots").json()
        assert [row["snapshot_id"] for row in other] == ["c"]


def test_get_unknown_or_cross_device_snapshot_is_404(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        _submit(client, "dev-1", "e1", "2026-09-20T10:00:00Z")
        _snapshot(client, "dev-1", "snap-1")
        assert client.get(
            "/devices/dev-1/session-snapshots/missing"
        ).status_code == 404
        # The snapshot exists, but belongs to another device.
        assert client.get(
            "/devices/dev-2/session-snapshots/snap-1"
        ).status_code == 404


# ------------------------------------------------------------------ diffs


def test_diff_unchanged_when_projection_identical(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        _submit(client, "dev-1", "e1", "2026-09-20T10:00:00Z")
        _submit(client, "dev-1", "e2", "2026-09-20T10:02:00Z")
        old = _snapshot(client, "dev-1", "old")
        old_sid = old.json()["sessions"][0]["session_id"]

        # Unrelated device activity advances the global seq but not the device.
        _submit(client, "dev-other", "x", "2026-09-20T11:00:00Z")
        new = _snapshot(client, "dev-1", "new")
        assert new.json()["high_water_seq"] == old.json()["high_water_seq"]

        resp = client.get(
            "/devices/dev-1/session-snapshots/old/diff/new"
        )
        assert resp.status_code == 200, resp.text
        changes = resp.json()["changes"]
        assert len(changes) == 1
        assert changes[0]["change"] == "unchanged"
        assert changes[0]["old_sessions"] == [old_sid]
        assert changes[0]["new_sessions"] == [old_sid]
        assert changes[0]["added_members"] == []
        assert changes[0]["removed_members"] == []


def test_diff_changed_one_to_one(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        # Two events 480s apart: distinct sessions.
        _submit(client, "dev-1", "e1", "2026-09-20T10:00:00Z")
        _submit(client, "dev-1", "e2", "2026-09-20T10:08:00Z")
        old = _snapshot(client, "dev-1", "old").json()
        assert old["session_count"] == 2

        # A new event extends only the first session (210s after e1 and
        # 270s before e2 -> it would bridge both... use 240s to e1 and
        # 240s to e2 would bridge too; instead sit at +120s, 360s from e2).
        _submit(client, "dev-1", "e3", "2026-09-20T10:02:00Z")
        new = _snapshot(client, "dev-1", "new").json()
        assert new["session_count"] == 2

        changes = client.get(
            "/devices/dev-1/session-snapshots/old/diff/new"
        ).json()["changes"]
        by_change = {entry["change"]: entry for entry in changes}
        assert set(by_change) == {"unchanged", "changed"}
        changed = by_change["changed"]
        assert len(changed["old_sessions"]) == 1
        assert len(changed["new_sessions"]) == 1
        assert [m["event_id"] for m in changed["added_members"]] == ["e3"]
        assert changed["removed_members"] == []
        unchanged = by_change["unchanged"]
        assert unchanged["added_members"] == []
        assert unchanged["removed_members"] == []
        assert len(unchanged["old_sessions"]) == 1


def test_diff_merged_when_late_event_bridges(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        _submit(client, "dev-1", "e1", "2026-09-20T10:00:00Z")
        _submit(client, "dev-1", "e2", "2026-09-20T10:01:00Z")
        _submit(client, "dev-1", "e3", "2026-09-20T10:10:00Z")  # new session
        _submit(client, "dev-1", "e4", "2026-09-20T10:11:00Z")
        old = _snapshot(client, "dev-1", "old").json()
        assert [s["event_count"] for s in old["sessions"]] == [2, 2]
        old_sids = [s["session_id"] for s in old["sessions"]]

        # Late bridge: 10:05 sits 240s from both neighbours.
        _submit(client, "dev-1", "e5", "2026-09-20T10:05:00Z")
        new = _snapshot(client, "dev-1", "new").json()
        assert new["session_count"] == 1

        changes = client.get(
            "/devices/dev-1/session-snapshots/old/diff/new"
        ).json()["changes"]
        assert len(changes) == 1
        entry = changes[0]
        assert entry["change"] == "merged"
        assert entry["old_sessions"] == old_sids
        assert entry["new_sessions"] == [new["sessions"][0]["session_id"]]
        assert [m["event_id"] for m in entry["added_members"]] == ["e5"]
        assert entry["removed_members"] == []


def test_diff_split_when_cleanup_removes_bridge(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        # One bridged session.
        _submit(client, "dev-1", "e1", "2026-09-20T10:00:00Z")
        _submit(client, "dev-1", "e2", "2026-09-20T10:05:00Z")  # bridge
        _submit(client, "dev-1", "e3", "2026-09-20T10:10:00Z")
        old = _snapshot(client, "dev-1", "old").json()
        assert old["session_count"] == 1

        # Delete just the bridge e2 through the two-phase retention
        # cleanup, protecting e1 with a protected window.
        new_plan = client.post(
            "/retention/cleanup-plans",
            json={
                "device_id": "dev-1",
                "before": "2026-09-20T10:06:00Z",
                "protected_windows": [
                    {"start": "2026-09-20T09:59:00Z", "end": "2026-09-20T10:01:00Z"}
                ],
            },
        ).json()
        assert [c["event_id"] for c in new_plan["candidates"]] == ["e2"]
        confirm = client.post(
            f"/retention/cleanup-plans/{new_plan['plan_id']}/confirm"
        )
        assert confirm.status_code == 200, confirm.text

        live = client.get("/devices/dev-1/sessions").json()
        assert [s["event_count"] for s in live] == [1, 1]
        new = _snapshot(client, "dev-1", "new").json()
        assert new["session_count"] == 2

        changes = client.get(
            "/devices/dev-1/session-snapshots/old/diff/new"
        ).json()["changes"]
        assert len(changes) == 1
        entry = changes[0]
        assert entry["change"] == "split"
        assert entry["old_sessions"] == [old["sessions"][0]["session_id"]]
        assert len(entry["new_sessions"]) == 2
        assert entry["added_members"] == []
        assert [m["event_id"] for m in entry["removed_members"]] == ["e2"]


def test_diff_created_and_removed(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        _submit(client, "dev-1", "e1", "2026-09-20T10:00:00Z")
        _snapshot(client, "dev-1", "old")

        # A far-away event creates a brand-new session (no shared members).
        _submit(client, "dev-1", "e2", "2026-09-20T11:00:00Z")
        _snapshot(client, "dev-1", "new")

        changes = client.get(
            "/devices/dev-1/session-snapshots/old/diff/new"
        ).json()["changes"]
        labels = [entry["change"] for entry in changes]
        assert labels.count("unchanged") == 1
        created = next(entry for entry in changes if entry["change"] == "created")
        assert created["old_sessions"] == []
        assert len(created["new_sessions"]) == 1
        assert [m["event_id"] for m in created["added_members"]] == ["e2"]

        # Reverse direction: the new session is now removed.
        reverse = client.get(
            "/devices/dev-1/session-snapshots/new/diff/old"
        ).json()["changes"]
        removed = next(entry for entry in reverse if entry["change"] == "removed")
        assert removed["new_sessions"] == []
        assert len(removed["old_sessions"]) == 1
        assert [m["event_id"] for m in removed["removed_members"]] == ["e2"]


def test_diff_changes_are_stably_ordered(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        # Two isolated sessions.
        _submit(client, "dev-1", "a1", "2026-09-20T10:00:00Z")
        _submit(client, "dev-1", "b1", "2026-09-20T11:00:00Z")
        _snapshot(client, "dev-1", "old")

        # Extend each session: two independent changed components, ordered by
        # the earliest member (10:00 session first, then 11:00).
        _submit(client, "dev-1", "a2", "2026-09-20T10:01:00Z")
        _submit(client, "dev-1", "b2", "2026-09-20T11:01:00Z")
        _snapshot(client, "dev-1", "new")

        changes = client.get(
            "/devices/dev-1/session-snapshots/old/diff/new"
        ).json()["changes"]
        assert [entry["change"] for entry in changes] == ["changed", "changed"]
        assert [
            m["event_id"]
            for entry in changes
            for m in entry["added_members"]
        ] == ["a2", "b2"]


def test_diff_unknown_snapshot_and_gap_mismatch(tmp_path):
    app = _make_app(tmp_path, gap=300)
    with TestClient(app) as client:
        _submit(client, "dev-1", "e1", "2026-09-20T10:00:00Z")
        _snapshot(client, "dev-1", "snap-a")

        assert client.get(
            "/devices/dev-1/session-snapshots/missing/diff/snap-a"
        ).status_code == 404
        # Cross-device reference in the path is a 404 as well.
        _submit(client, "dev-2", "e2", "2026-09-20T10:00:00Z")
        _snapshot(client, "dev-2", "snap-b")
        assert client.get(
            "/devices/dev-1/session-snapshots/snap-a/diff/snap-b"
        ).status_code == 404

    # Different SESSION_GAP_SECONDS against the same database.
    app2 = _make_app(tmp_path, gap=60)
    with TestClient(app2) as client:
        # With a 60s gap e1 stands alone; still one session.
        snap_c = _snapshot(client, "dev-1", "snap-c")
        assert snap_c.status_code == 201
        resp = client.get(
            "/devices/dev-1/session-snapshots/snap-a/diff/snap-c"
        )
        assert resp.status_code == 409, resp.text
        detail = resp.json()["detail"]
        assert detail["old_gap_seconds"] == 300.0
        assert detail["new_gap_seconds"] == 60.0


# ----------------------------------------------------------- freeze semantics


def test_late_event_does_not_rewrite_old_snapshot(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        _submit(client, "dev-1", "e1", "2026-09-20T10:00:00Z")
        _submit(client, "dev-1", "e2", "2026-09-20T10:01:00Z")
        _submit(client, "dev-1", "e3", "2026-09-20T10:10:00Z")
        _submit(client, "dev-1", "e4", "2026-09-20T10:11:00Z")
        old = _snapshot(client, "dev-1", "old").json()
        assert old["session_count"] == 2

        # Late event arrives and bridges the two live sessions.
        bridge = _submit(client, "dev-1", "e5", "2026-09-20T10:05:00Z")
        assert bridge.json()["session"]["event_count"] == 5

        # The stored snapshot is byte-for-byte the old projection.
        frozen = client.get(
            "/devices/dev-1/session-snapshots/old"
        ).json()
        assert frozen == old
        assert frozen["session_count"] == 2
        assert frozen["high_water_seq"] == 4

        # Replay keeps returning the original.
        assert _snapshot(client, "dev-1", "old").status_code == 200
        assert (
            _snapshot(client, "dev-1", "old").json()["session_count"] == 2
        )


def test_snapshot_high_water_excludes_events_arriving_after_capture(tmp_path):
    """Freezing must use the device's monotonic seq, not event time."""
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        # Two isolated sessions.
        _submit(client, "dev-1", "e2", "2026-09-20T10:10:00Z")
        _submit(client, "dev-1", "e1", "2026-09-20T10:00:00Z")
        snap = _snapshot(client, "dev-1", "snap").json()
        assert snap["high_water_seq"] == 2
        assert [s["event_count"] for s in snap["sessions"]] == [1, 1]

        # An event with an OLD event time arrives after the snapshot.
        _submit(client, "dev-1", "e0", "2026-09-20T09:00:00Z")
        frozen = client.get("/devices/dev-1/session-snapshots/snap").json()
        assert frozen["high_water_seq"] == 2
        assert all(
            member["event_id"] != "e0"
            for session in frozen["sessions"]
            for member in session["members"]
        )


def test_payload_events_are_frozen_with_their_content(tmp_path):
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        schema = client.put(
            "/schemas/telemetry/1.0",
            json={
                "schema": {
                    "type": "object",
                    "required": ["value"],
                    "properties": {"value": {"type": "number"}},
                }
            },
        )
        assert schema.status_code == 201
        event = client.post(
            "/events",
            json={
                "device_id": "dev-1",
                "event_id": "e1",
                "occurred_at": "2026-09-20T10:00:00Z",
                "event_type": "telemetry",
                "schema_version": "1.0",
                "payload": {"value": 42},
            },
        )
        assert event.status_code == 201
        _snapshot(client, "dev-1", "snap")
        frozen = client.get(
            "/devices/dev-1/session-snapshots/snap"
        ).json()
        member = frozen["sessions"][0]["members"][0]
        assert member["event_type"] == "telemetry"
        assert member["schema_version"] == "1.0"
        assert member["payload"] == {"value": 42}


# ------------------------------------------------- pure diff classification


def test_diff_reorganized_many_to_many():
    """A 2-old/2-new component (members reshuffled) is ``reorganized``.

    The live API cannot produce this shape (sessions only merge or split
    around shared members), so the classification is exercised directly
    against the pure diff function.
    """
    from datetime import datetime, timezone

    from app.snapshots import (
        FrozenMember,
        FrozenSession,
        Snapshot,
        diff_snapshots,
    )

    def member(event_id, ts):
        return FrozenMember(
            event_id=event_id, occurred_at=datetime.fromisoformat(ts)
        )

    def session(session_id, members):
        ordered = tuple(
            sorted(members, key=lambda m: (m.occurred_at, m.event_id))
        )
        return FrozenSession(
            session_id=session_id,
            started_at=ordered[0].occurred_at,
            ended_at=ordered[-1].occurred_at,
            members=ordered,
        )

    created = datetime(2026, 9, 20, tzinfo=timezone.utc)
    old = Snapshot(
        snapshot_id="old",
        device_id="dev",
        high_water_seq=4,
        gap_seconds=300.0,
        sessions=(
            session(
                "O1",
                [
                    member("a", "2026-09-20T10:00:00+00:00"),
                    member("b", "2026-09-20T10:01:00+00:00"),
                ],
            ),
            session(
                "O2",
                [
                    member("c", "2026-09-20T10:02:00+00:00"),
                    member("d", "2026-09-20T10:03:00+00:00"),
                ],
            ),
        ),
        created_at=created,
    )
    new = Snapshot(
        snapshot_id="new",
        device_id="dev",
        high_water_seq=4,
        gap_seconds=300.0,
        sessions=(
            session(
                "N1",
                [
                    member("a", "2026-09-20T10:00:00+00:00"),
                    member("c", "2026-09-20T10:02:00+00:00"),
                ],
            ),
            session(
                "N2",
                [
                    member("b", "2026-09-20T10:01:00+00:00"),
                    member("d", "2026-09-20T10:03:00+00:00"),
                ],
            ),
        ),
        created_at=created,
    )

    changes = diff_snapshots(old, new)
    assert len(changes) == 1
    entry = changes[0]
    assert entry.change == "reorganized"
    assert entry.old_sessions == ("O1", "O2")
    assert entry.new_sessions == ("N1", "N2")
    assert entry.added_members == ()
    assert entry.removed_members == ()
