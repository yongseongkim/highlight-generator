"""Sport events define candidate plays; pose/action scores rank those plays.

A pose pause or a lost shuttle is not a ground-contact observation. Boundary
verification is preserved independently from the interest score of a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class SportEvent:
    time_s: float
    kind: str
    court_id: str = "main"
    source: str = "external"
    confidence: float | None = None
    verified: bool = False
    event_id: str | None = None
    data: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.verified, bool):
            raise ValueError("Event verified must be a boolean")
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("Event timestamps must be finite and nonnegative")
        if self.confidence is not None and (not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1):
            raise ValueError("Event confidence must be in [0, 1] or unknown")


@dataclass
class GameUnit:
    sport: str
    unit_type: str
    play_start_s: float
    play_end_s: float
    start_s: float
    end_s: float
    start_reason: str
    end_reason: str
    start_verified: bool
    end_verified: bool
    court_id: str
    events: list[SportEvent]
    ranking_score: float = 0.0
    features: dict = field(default_factory=dict)

    @property
    def boundary_status(self) -> str:
        if self.start_verified and self.end_verified:
            return "verified"
        if self.start_verified or self.end_verified:
            return "partially_observed"
        return "estimated"


RALLY_ENDS = {"shuttle_landed", "shuttle_landed_out", "net_fault", "fault", "let", "point_end"}
RALLY_ACTIVITY = {"hit", "shuttle_in_flight", "in_play", "pose_activity"}


def badminton_units(events: list[SportEvent], duration_s: float, court_id: str = "main",
                    pre_s: float = 0.75, post_s: float = 1.0,
                    allow_activity_estimates: bool = False) -> list[GameUnit]:
    """Serve -> in play -> explicit terminal event, without losing brief occlusions.

    Pose-only start/pause events are accepted only in the opt-in estimate path.
    A new serve without an observed ending closes an incomplete preceding unit;
    it never fabricates a landing. Padding never merges neighboring rallies.
    """
    if not math.isfinite(duration_s) or duration_s <= 0 or not all(math.isfinite(v) and v >= 0 for v in (pre_s, post_s)):
        raise ValueError("Duration must be positive and padding finite/nonnegative")
    observations = sorted((event for event in events if event.court_id == court_id and event.time_s <= duration_s), key=lambda e: e.time_s)
    units = []
    active: list[SportEvent] = []

    def finish(end: SportEvent, reason: str | None = None):
        nonlocal active
        if not active:
            return
        start = active[0]
        if end.time_s <= start.time_s:
            active = []
            return
        start_known = start.kind == "serve" and start.verified
        end_known = end.kind in RALLY_ENDS and end.verified
        units.append(GameUnit("badminton", "rally" if start_known and end_known else "rally_candidate",
                              start.time_s, end.time_s, max(0, start.time_s - pre_s), min(duration_s, end.time_s + post_s),
                              start.kind, reason or end.kind, start_known, end_known, court_id,
                              [*active, end]))
        active = []

    for event in observations:
        if event.kind == "serve":
            if active:
                previous_count = len(units)
                finish(event, "next_serve_without_terminal")
                # The second serve bounds the clip, but does not verify its end.
                if len(units) > previous_count:
                    units[-1].end_verified = False
                    units[-1].unit_type = "rally_candidate"
                    units[-1].end_s = min(units[-1].end_s, event.time_s)
            active = [event]
        elif event.kind == "activity_start" and allow_activity_estimates:
            if not active:
                active = [event]
        elif event.kind in RALLY_ACTIVITY:
            if not active and event.kind != "pose_activity":
                active = [event]
            elif active:
                active.append(event)
        elif event.kind in RALLY_ENDS:
            if not active:
                if units:
                    continue  # duplicate terminal observations do not start a new rally
                # A clip that begins mid-rally has an unknown opening boundary.
                active = [SportEvent(0.0, "video_start_unknown", court_id, source="video_boundary")]
            finish(event)
        elif event.kind == "activity_pause" and allow_activity_estimates:
            # A visible pause may end an estimate, never a serve-observed rally.
            if active and active[0].kind == "activity_start":
                finish(event, "pose_pause_estimate")
        elif active and event.kind in {"shuttle_missing", "occluded", "tracking_resumed"}:
            active.append(event)
    if active:
        finish(SportEvent(duration_s, "video_end_unknown", court_id, source="video_boundary"))
    return units


def scoring_units(events: list[SportEvent], sport: str, duration_s: float, court_id: str = "main",
                  lead_s: float | None = None, post_s: float = 3.0) -> list[GameUnit]:
    """Anchor basketball/football candidates on outcome events, with lead-up context.

    Goal detection is supplied by an upstream detector/annotation; pose movement
    cannot manufacture a goal. Explicit attack starts take precedence over padding.
    """
    policies = {"basketball": ({"basket_made"}, 8.0), "football": ({"goal"}, 15.0)}
    if sport not in policies:
        raise ValueError("Scoring units support basketball or football")
    accepted, default_lead = policies[sport]
    lead_s = default_lead if lead_s is None else lead_s
    if not math.isfinite(duration_s) or duration_s <= 0 or not all(math.isfinite(v) and v >= 0 for v in (lead_s, post_s)):
        raise ValueError("Duration must be positive and padding finite/nonnegative")
    observations = sorted((e for e in events if e.court_id == court_id and e.time_s <= duration_s), key=lambda e: e.time_s)
    units, seen = [], set()
    attack = None
    previous_goal = -math.inf
    for event in observations:
        if event.kind in {"attack_start", "possession_start"}:
            attack = event
        elif event.kind in accepted:
            # Explicit IDs distinguish two real scores close together. Repeated
            # frames of an un-IDed detection are collapsed within one second.
            key = event.event_id
            if (key is not None and key in seen) or (key is None and event.time_s - previous_goal < 1):
                continue
            if key is not None:
                seen.add(key)
            start = attack.time_s if attack is not None else max(0, event.time_s - lead_s)
            start_known = attack is not None and attack.verified
            unit_events = [attack, event] if attack is not None else [event]
            units.append(GameUnit(sport, "scoring_play", start, event.time_s, start, min(duration_s, event.time_s + post_s),
                                  attack.kind if attack else "lead_up_context", event.kind,
                                  start_known, event.verified, court_id, unit_events,
                                  ranking_score=1.0, features={"outcome_event": event.kind}))
            previous_goal = event.time_s
            attack = None
    return units


def rank_units(units: list[GameUnit], actions: list, max_highlights: int = 5, budget_s: float = 120,
               verified_only: bool = False) -> list[GameUnit]:
    """Rank entire units; a duration budget never cuts a rally in half."""
    if max_highlights < 0 or not math.isfinite(budget_s) or budget_s < 0:
        raise ValueError("Selection budgets must be finite and nonnegative")
    for unit in units:
        inside = [a for a in actions if unit.play_start_s <= a.time_s <= unit.play_end_s]
        duration = unit.play_end_s - unit.play_start_s
        unit.features.update(action_match_count=len(inside), action_types=sorted({a.pattern for a in inside}),
                             sam_supported_action_count=sum(bool(a.sam.get("ranking_bonus", 0)) for a in inside),
                             play_duration_s=duration,
                             peak_action_score=max((a.ranking_score for a in inside), default=0))
        # Duration/activity are interest signals, not boundary or point confidence.
        unit.ranking_score = (0.35 * -math.expm1(-duration / 20) + 0.4 * -math.expm1(-len(inside) / 8)
                              + 0.25 * min(unit.features["peak_action_score"], 1))
        if unit.unit_type == "scoring_play":
            unit.ranking_score += 1
    selected, used = [], 0.0
    for unit in sorted(units, key=lambda u: (-u.ranking_score, u.play_start_s)):
        if len(selected) >= max_highlights:
            break
        if verified_only and unit.boundary_status != "verified":
            continue
        length = unit.end_s - unit.start_s
        if length <= 0 or used + length > budget_s + 1e-8:
            continue
        selected.append(unit)
        used += length
    return sorted(selected, key=lambda u: u.play_start_s)
