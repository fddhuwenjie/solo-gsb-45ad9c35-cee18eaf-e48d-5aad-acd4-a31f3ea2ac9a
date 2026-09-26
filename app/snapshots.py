"""Session evolution snapshots and old/new snapshot diffs.

A snapshot freezes one device's session projection at a point in time:

* ``high_water_seq`` is the device's current event monotonic high-water
  (``events.seq``), captured in the caller's write transaction;
* the frozen members are **copied** into dedicated snapshot tables rather
  than referenced from ``events``, so later arrivals (including late
  events that bridge sessions) and retention cleanup deletes can never
  rewrite an old snapshot;
* sessions are grouped from the frozen members with the *current* gap
  configuration; ``gap_seconds`` is persisted with the snapshot, because
  two snapshots grouped under different gaps are not comparable.

The diff between an old and a new snapshot is computed purely from the
frozen copies: shared ``event_id`` values connect the old and new
sessions into a bipartite graph, and the connected components classify
each evolution as ``unchanged``, ``changed``, ``merged``, ``split``,
``created``, ``removed`` (many-to-many components are ``reorganized``).

Apart from the small SQL helpers taking an open connection, the logic is
pure: callers (the FastAPI endpoints) own the transactions.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .ingestion import canonical_json
from .sessions import Event, from_storage, group_sessions, session_id, to_storage

CHANGE_UNCHANGED = "unchanged"
CHANGE_CHANGED = "changed"
CHANGE_MERGED = "merged"
CHANGE_SPLIT = "split"
CHANGE_CREATED = "created"
CHANGE_REMOVED = "removed"
CHANGE_REORGANIZED = "reorganized"

SNAPSHOT_COLUMNS = (
    "snapshot_id, device_id, high_water_seq, gap_seconds, "
    "session_count, created_at"
)


# ----------------------------------------------------------------- models


@dataclass(frozen=True)
class FrozenMember:
    """One event copied into a snapshot."""

    event_id: str
    occurred_at: datetime  # always timezone-aware, normalized to UTC
    event_type: str | None = None
    schema_version: str | None = None
    payload: Any = None


@dataclass(frozen=True)
class FrozenSession:
    """One session of a snapshot; ``members`` are in (occurred_at, event_id) order."""

    session_id: str
    started_at: datetime
    ended_at: datetime
    members: tuple[FrozenMember, ...]

    @property
    def event_count(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    device_id: str
    high_water_seq: int
    gap_seconds: float
    sessions: tuple[FrozenSession, ...]
    created_at: datetime

    @property
    def session_count(self) -> int:
        return len(self.sessions)


@dataclass(frozen=True)
class DiffChange:
    """One connected component of the old/new session bipartite graph."""

    change: str
    old_sessions: tuple[str, ...]
    new_sessions: tuple[str, ...]
    added_members: tuple[FrozenMember, ...]
    removed_members: tuple[FrozenMember, ...]


# ------------------------------------------------------------- building


def device_event_high_water(conn: sqlite3.Connection, device_id: str) -> int:
    """Maximum arrival seq of the device's events; 0 when there are none."""
    row = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) AS m FROM events WHERE device_id = ?",
        (device_id,),
    ).fetchone()
    return int(row["m"])


def _row_to_member(row: sqlite3.Row) -> FrozenMember:
    return FrozenMember(
        event_id=row["event_id"],
        occurred_at=from_storage(row["occurred_at"]),
        event_type=row["event_type"],
        schema_version=row["schema_version"],
        payload=json.loads(row["payload"]) if row["payload"] is not None else None,
    )


def frozen_device_events(
    conn: sqlite3.Connection, device_id: str, high_water_seq: int
) -> list[Event]:
    """The device's events with ``seq <= high_water_seq``, session-sorted.

    Reading through ``seq`` freezes the row set: events inserted by a
    concurrent writer after the high-water was captured (or a late event
    arriving afterwards) are excluded even though it shares the same open
    transaction snapshot.
    """
    rows = conn.execute(
        "SELECT event_id, occurred_at, event_type, schema_version, payload "
        "FROM events WHERE device_id = ? AND seq <= ? "
        "ORDER BY occurred_at, event_id",
        (device_id, high_water_seq),
    ).fetchall()
    return [
        Event(
            device_id=device_id,
            event_id=row["event_id"],
            occurred_at=from_storage(row["occurred_at"]),
            event_type=row["event_type"],
            schema_version=row["schema_version"],
            payload=json.loads(row["payload"]) if row["payload"] is not None else None,
        )
        for row in rows
    ]


def build_snapshot(
    conn: sqlite3.Connection,
    *,
    snapshot_id: str,
    device_id: str,
    gap_seconds: float,
    created_at: datetime,
) -> Snapshot:
    """Freeze the device's current session projection.

    Must run inside the caller's open write transaction: the high-water
    read and the frozen member scan share that transaction snapshot, so a
    concurrent writer cannot insert an event between capturing the
    high-water and copying the members.
    """
    high_water_seq = device_event_high_water(conn, device_id)
    events = frozen_device_events(conn, device_id, high_water_seq)
    groups = group_sessions(events, gap_seconds)
    sessions = tuple(
        FrozenSession(
            session_id=session_id(device_id, members),
            started_at=members[0].occurred_at,
            ended_at=members[-1].occurred_at,
            members=tuple(
                FrozenMember(
                    event_id=member.event_id,
                    occurred_at=member.occurred_at,
                    event_type=member.event_type,
                    schema_version=member.schema_version,
                    payload=member.payload,
                )
                for member in members
            ),
        )
        for members in groups
    )
    return Snapshot(
        snapshot_id=snapshot_id,
        device_id=device_id,
        high_water_seq=high_water_seq,
        gap_seconds=float(gap_seconds),
        sessions=sessions,
        created_at=created_at,
    )


def save_snapshot(conn: sqlite3.Connection, snapshot: Snapshot) -> None:
    """Persist a snapshot header, its ordered sessions and member copies.

    All three inserts run inside the caller's transaction; the PRIMARY
    KEY on ``session_snapshots`` makes a concurrent reuse of the same
    ``snapshot_id`` raise ``sqlite3.IntegrityError``.
    """
    conn.execute(
        "INSERT INTO session_snapshots (snapshot_id, device_id, "
        "high_water_seq, gap_seconds, session_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            snapshot.snapshot_id,
            snapshot.device_id,
            snapshot.high_water_seq,
            snapshot.gap_seconds,
            snapshot.session_count,
            to_storage(snapshot.created_at),
        ),
    )
    for index, sess in enumerate(snapshot.sessions):
        conn.execute(
            "INSERT INTO session_snapshot_sessions (snapshot_id, session_index, "
            "session_id, started_at, ended_at, event_count, member_ids) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot.snapshot_id,
                index,
                sess.session_id,
                to_storage(sess.started_at),
                to_storage(sess.ended_at),
                sess.event_count,
                json.dumps([member.event_id for member in sess.members]),
            ),
        )
    conn.executemany(
        "INSERT INTO session_snapshot_members (snapshot_id, event_id, "
        "occurred_at, event_type, schema_version, payload) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                snapshot.snapshot_id,
                member.event_id,
                to_storage(member.occurred_at),
                member.event_type,
                member.schema_version,
                (
                    canonical_json(member.payload)
                    if member.payload is not None
                    else None
                ),
            )
            for sess in snapshot.sessions
            for member in sess.members
        ],
    )


# ------------------------------------------------------------- loading


def get_snapshot_row(
    conn: sqlite3.Connection, snapshot_id: str
) -> sqlite3.Row | None:
    return conn.execute(
        f"SELECT {SNAPSHOT_COLUMNS} FROM session_snapshots WHERE snapshot_id = ?",
        (snapshot_id,),
    ).fetchone()


def load_snapshot(conn: sqlite3.Connection, snapshot_id: str) -> Snapshot | None:
    """Load a snapshot header together with its ordered frozen sessions."""
    row = get_snapshot_row(conn, snapshot_id)
    if row is None:
        return None
    members = {
        member_row["event_id"]: _row_to_member(member_row)
        for member_row in conn.execute(
            "SELECT event_id, occurred_at, event_type, schema_version, payload "
            "FROM session_snapshot_members WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchall()
    }
    session_rows = conn.execute(
        "SELECT session_id, started_at, ended_at, member_ids "
        "FROM session_snapshot_sessions WHERE snapshot_id = ? "
        "ORDER BY session_index",
        (snapshot_id,),
    ).fetchall()
    sessions = []
    for session_row in session_rows:
        ordered_members = tuple(
            members[event_id] for event_id in json.loads(session_row["member_ids"])
        )
        sessions.append(
            FrozenSession(
                session_id=session_row["session_id"],
                started_at=from_storage(session_row["started_at"]),
                ended_at=from_storage(session_row["ended_at"]),
                members=ordered_members,
            )
        )
    return Snapshot(
        snapshot_id=row["snapshot_id"],
        device_id=row["device_id"],
        high_water_seq=int(row["high_water_seq"]),
        gap_seconds=float(row["gap_seconds"]),
        sessions=tuple(sessions),
        created_at=from_storage(row["created_at"]),
    )


def list_device_snapshot_rows(
    conn: sqlite3.Connection, device_id: str
) -> list[sqlite3.Row]:
    return conn.execute(
        f"SELECT {SNAPSHOT_COLUMNS} FROM session_snapshots "
        "WHERE device_id = ? ORDER BY created_at, snapshot_id",
        (device_id,),
    ).fetchall()


# ----------------------------------------------------------------- diff


def _member_key(member: FrozenMember) -> tuple[datetime, str]:
    return (member.occurred_at, member.event_id)


def diff_snapshots(old: Snapshot, new: Snapshot) -> list[DiffChange]:
    """Classify the evolution from ``old`` to ``new`` via shared event ids.

    A bipartite graph links an old session to a new session whenever they
    share at least one ``event_id``; each connected component becomes one
    :class:`DiffChange`:

    * one old + one new session with identical members -> ``unchanged``;
    * one old + one new session with differing members -> ``changed``;
    * several old + one new -> ``merged``;
    * one old + several new -> ``split``;
    * several old + several new -> ``reorganized``;
    * only a new session -> ``created``; only an old one -> ``removed``.

    ``added_members`` are members of the new component sessions absent
    from the old component sessions (the reverse for
    ``removed_members``). The result is stably ordered by the earliest
    involved member's ``(occurred_at, event_id)``; session id lists and
    member lists are sorted the same way.
    """
    # Disjoint-set union over ("o", old_index) / ("n", new_index) nodes.
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node: tuple[str, int]) -> tuple[str, int]:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            parent[node], node = root, parent[node]
        return root

    def union(left: tuple[str, int], right: tuple[str, int]) -> None:
        parent[find(left)] = find(right)

    old_nodes = [("o", index) for index in range(len(old.sessions))]
    new_nodes = [("n", index) for index in range(len(new.sessions))]
    for node in old_nodes + new_nodes:
        parent[node] = node

    # event_id -> indices of the old/new sessions containing it; an event
    # belongs to at most one session per snapshot (sessions partition the
    # frozen member set), but stay defensive and accept several links.
    old_owner: dict[str, int] = {}
    for index, sess in enumerate(old.sessions):
        for member in sess.members:
            old_owner[member.event_id] = index
    for new_index, sess in enumerate(new.sessions):
        for member in sess.members:
            old_index = old_owner.get(member.event_id)
            if old_index is not None:
                union(("o", old_index), ("n", new_index))

    components: dict[tuple[str, int], dict[str, list[int]]] = {}
    for side, index in old_nodes:
        components.setdefault(find((side, index)), {"o": [], "n": []})["o"].append(index)
    for side, index in new_nodes:
        components.setdefault(find((side, index)), {"o": [], "n": []})["n"].append(index)

    changes: list[tuple[tuple[Any, ...], DiffChange]] = []
    for component in components.values():
        old_indexes = component["o"]
        new_indexes = component["n"]
        old_sessions = [old.sessions[i] for i in old_indexes]
        new_sessions = [new.sessions[i] for i in new_indexes]
        old_members = {
            member.event_id: member
            for sess in old_sessions
            for member in sess.members
        }
        new_members = {
            member.event_id: member
            for sess in new_sessions
            for member in sess.members
        }
        added = sorted(
            (member for eid, member in new_members.items() if eid not in old_members),
            key=_member_key,
        )
        removed = sorted(
            (member for eid, member in old_members.items() if eid not in new_members),
            key=_member_key,
        )

        old_count, new_count = len(old_sessions), len(new_sessions)
        if old_count == 1 and new_count == 1:
            change = (
                CHANGE_UNCHANGED if not added and not removed else CHANGE_CHANGED
            )
        elif old_count == 1 and new_count >= 2:
            change = CHANGE_SPLIT
        elif old_count >= 2 and new_count == 1:
            change = CHANGE_MERGED
        elif old_count >= 2 and new_count >= 2:
            change = CHANGE_REORGANIZED
        elif new_count == 1:
            change = CHANGE_CREATED
        else:
            change = CHANGE_REMOVED

        # Sessions and members share the (started/occurred_at, id) order.
        old_sessions.sort(key=lambda sess: (sess.started_at, sess.session_id))
        new_sessions.sort(key=lambda sess: (sess.started_at, sess.session_id))
        first_member = min(
            (sess.members[0] for sess in old_sessions + new_sessions),
            key=_member_key,
        )
        # Components are ordered by the earliest involved member's event
        # time (then event_id), independent of the change classification.
        sort_key = (*_member_key(first_member), change)
        item = DiffChange(
            change=change,
            old_sessions=tuple(sess.session_id for sess in old_sessions),
            new_sessions=tuple(sess.session_id for sess in new_sessions),
            added_members=tuple(added),
            removed_members=tuple(removed),
        )
        changes.append((sort_key, item))

    changes.sort(key=lambda entry: entry[0])
    return [item for _, item in changes]
