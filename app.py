#!/usr/bin/env python3

import hashlib
import itertools
import json
import random
from pathlib import Path
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import requests
from flask import Flask, jsonify, request, render_template

# ─── CONFIG ──────────────────────────────────────────────────────────────────

API_BASE = "https://worldcup26.ir"
API_HEADERS = {"Authorization": "Bearer demo"}
REFRESH_SECS_LIVE = 30        # poll interval when any match is live
REFRESH_SECS_IDLE = 3600      # poll interval when no matches are live
MONTE_CARLO_ITERATIONS = 50_000
ADVANCE_SLOTS = 8
PORT = 8080
SIM_MODE = "elo"  # "elo" or "equal"

_MC_CACHE: dict[str, dict] = {}
_MC_CACHE_HASH: str = ""

_ELO_CACHE: dict[str, float] = {"default": 1500}

def _state_hash(standings: dict) -> str:
    raw = json.dumps(standings, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode()).hexdigest()

def _rebuild_elo(standings: dict[str, list[dict]]) -> None:
    elo: dict[str, float] = {}
    for table in standings.values():
        for row in table:
            pts = row["pts"]
            p = max(row["p"], 1)
            ppm = pts / p
            elo[row["team"]] = 1500.0 + (ppm - 1.0) * 100.0
    if not elo:
        elo["default"] = 1500
    _ELO_CACHE.clear()
    _ELO_CACHE.update(elo)

# ─── DATA FETCHING ──────────────────────────────────────────────────────────

_team_cache: dict[str, str] = {}
_last_fetch: dict[str, Any] = {}
_last_fetch_time = 0.0
_last_api_updated: str = ""  # raw updatedAt string from last successful fetch


def _load_teams() -> dict[str, str]:
    if _team_cache:
        return _team_cache
    try:
        r = requests.get(f"{API_BASE}/get/teams", headers=API_HEADERS, timeout=3)
        r.raise_for_status()
        for t in r.json().get("teams", []):
            _team_cache[t["id"]] = t["name_en"]
        return _team_cache
    except Exception:
        return _team_cache


def _get(path: str) -> dict:
    try:
        r = requests.get(f"{API_BASE}/{path}", headers=API_HEADERS, timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        print("FETCH_ALL_DATA FAILED")
        traceback.print_exc()
        raise

def fetch_all_data(force: bool = False) -> dict[str, Any]:
    global _last_fetch_time, _last_api_updated
    try:
        teams = _load_teams()
        raw_groups = _get("get/groups")
        raw_games = _get("get/games")

        ts = raw_groups.get("updatedAt") or raw_groups.get("_updated")

        # Skip full reparse if server data hasn't changed
        if not force and ts and ts == _last_api_updated and _last_fetch:
            _last_fetch_time = time.time()
            return _last_fetch

        standings: dict[str, list[dict]] = {}
        all_teams_set: set[str] = set()
        for g in raw_groups.get("groups", []):
            gname = f"Group {g['name']}"
            table = []
            for t in g.get("teams", []):
                name = teams.get(t["team_id"], f"Team#{t['team_id']}")
                all_teams_set.add(name)
                table.append({
                    "team": name,
                    "p": int(t.get("mp", 0)),
                    "w": int(t.get("w", 0)),
                    "d": int(t.get("d", 0)),
                    "l": int(t.get("l", 0)),
                    "pts": int(t.get("pts", 0)),
                    "gf": int(t.get("gf", 0)),
                    "ga": int(t.get("ga", 0)),
                    "gd": int(t.get("gd", 0)),
                    "group": gname,
                })
            standings[gname] = table

        ts = raw_groups.get("updatedAt") or raw_groups.get("_updated")
        updated_ts = None
        if ts:
            try:
                updated_ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except Exception:
                pass

        all_matches = []
        group_matches = []
        for g in raw_games.get("games", []):
            group_letter = g.get("group", "")
            is_group = g.get("type") == "group" and group_letter
            finished = g.get("finished") in ("TRUE", True, "true")
            elapsed = g.get("time_elapsed", "")
            status = "finished" if finished else ("upcoming" if elapsed in ("notstarted", None, "") else "live")

            minute = None
            if not finished and elapsed not in ("notstarted", None, ""):
                try:
                    minute = int(elapsed.rstrip("'"))
                except (ValueError, AttributeError):
                    pass

            def _safe_int(val):
                if val is None or val == "null" or val == "":
                    return 0
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return 0

            match = {
                "id": g.get("id"),
                "team1": teams.get(g.get("home_team_id", ""), g.get("home_team_name_en", "TBD")),
                "team2": teams.get(g.get("away_team_id", ""), g.get("away_team_name_en", "TBD")),
                "score": [_safe_int(g.get("home_score")), _safe_int(g.get("away_score"))],
                "status": status,
                "live_minute": minute,
                "group": f"Group {group_letter}" if is_group else "",
                "date": g.get("local_date", "").split(" ")[0] if g.get("local_date") else "",
                "matchday": g.get("matchday"),
            }
            all_matches.append(match)
            if is_group:
                group_matches.append(match)

        _rebuild_elo(standings)
        _last_fetch_time = time.time()
        if ts:
            _last_api_updated = ts
        result = {
            "standings": standings,
            "all_teams": sorted(all_teams_set),
            "matches": group_matches,
            "all_matches": all_matches,
            "updated": updated_ts or time.time(),
        }
        _last_fetch.clear()
        _last_fetch.update(result)
        return result
    except Exception as e:
        print(f"[fetch_all_data] Failed to fetch/parse API data: {e}")
        traceback.print_exc()
        return _last_fetch


# ─── HELPERS ────────────────────────────────────────────────────────────────


def rating(team: str) -> float:
    return _ELO_CACHE.get(team, _ELO_CACHE.get("default", 1500))


def elo_prob(r_a: float, r_b: float) -> tuple[float, float, float]:
    if SIM_MODE == "equal":
        return (0.33, 0.33, 0.34)
    d = r_a - r_b
    p_awin = 1.0 / (10.0 ** (-d / 400.0) + 1.0)
    p_bwin = 1.0 / (10.0 ** (d / 400.0) + 1.0)
    draw_prob = max(0.08, min(0.35, (1.0 - abs(p_awin - p_bwin)) * 0.4))
    total = p_awin + p_bwin
    p_awin = (p_awin / total) * (1.0 - draw_prob)
    p_bwin = (p_bwin / total) * (1.0 - draw_prob)
    return (p_awin, draw_prob, p_bwin)


def sample_result(r_a: float, r_b: float) -> str:
    pa, pd, pb = elo_prob(r_a, r_b)
    rnd = random.random()
    if rnd < pa:
        return "1"
    elif rnd < pa + pd:
        return "X"
    else:
        return "2"


def typical_score(result: str, r_a: float, r_b: float) -> tuple[int, int]:
    diff = abs(r_a - r_b)
    if result == "1":
        if diff > 100:
            return (2, 0)
        elif diff > 50:
            return (2, 1)
        else:
            return (1, 0)
    elif result == "X":
        return (1, 1) if r_a > r_b + 50 else (0, 0)
    else:
        if diff > 100:
            return (0, 2)
        elif diff > 50:
            return (1, 2)
        else:
            return (0, 1)


def rank_key(row: dict) -> tuple:
    # FIFA tiebreakers: Pts → GD → GF → fewest GA
    return (-row["pts"], -row["gd"], -row["gf"], row["ga"])


def apply_match_outcome(table: list[dict], match: dict, outcome: str, score1: int | None = None, score2: int | None = None) -> list[dict]:
    tbl = {r["team"]: dict(r) for r in table}
    t1, t2 = match["team1"], match["team2"]
    if score1 is not None and score2 is not None:
        g1, g2 = score1, score2
    else:
        r1, r2 = rating(t1), rating(t2)
        g1, g2 = typical_score(outcome, r1, r2)

    for name, goals_for, goals_against in [(t1, g1, g2), (t2, g2, g1)]:
        if name in tbl:
            row = tbl[name]
            row["p"] += 1
            row["gf"] += goals_for
            row["ga"] += goals_against
            row["gd"] = row["gf"] - row["ga"]
            if outcome == "1" and name == t1:
                row["w"] += 1; row["pts"] += 3
            elif outcome == "2" and name == t2:
                row["w"] += 1; row["pts"] += 3
            elif outcome == "X":
                row["d"] += 1; row["pts"] += 1
            else:
                row["l"] += 1

    return sorted(tbl.values(), key=rank_key)


def find_remaining(group: str, matches: list[dict]) -> list[dict]:
    return [m for m in matches if m.get("group") == group and m.get("status") != "finished"]


# ─── ANALYSIS ───────────────────────────────────────────────────────────────


def get_third_placed(standings: dict) -> list[dict]:
    thirds = []
    for gname, table in standings.items():
        if len(table) >= 3:
            row = dict(table[2])
            row["group"] = gname
            thirds.append(row)
    thirds.sort(key=rank_key)
    return thirds


def team_status(team: str, standings: dict) -> dict:
    groups = []
    for gname, table in standings.items():
        for i, row in enumerate(table):
            if row["team"] == team:
                groups.append({"group": gname, "position": i + 1, "data": dict(row)})
    if not groups:
        return {"found": False, "position": None, "status": "unknown", "data": None}

    g = groups[0]
    thirds = get_third_placed(standings)
    third_rank = None
    for i, t in enumerate(thirds):
        if t["team"] == team:
            third_rank = i + 1
            break

    status = "unknown"
    if g["position"] <= 2:
        status = "advancing"
    elif g["position"] == 3:
        if third_rank and third_rank <= ADVANCE_SLOTS:
            status = "advancing"
        else:
            status = "bubble"
    else:
        status = "eliminated"

    return {
        "found": True,
        "group": g["group"],
        "position": g["position"],
        "third_place_rank": third_rank,
        "status": status,
        "data": g["data"],
    }


def run_monte_carlo(standings: dict, matches: list[dict], n: int = MONTE_CARLO_ITERATIONS, track_matches: bool = False) -> dict:
    """Run the Monte Carlo simulation.

    When track_matches=True, also records, for every still-unplayed match,
    which simulations sampled which outcome (1/X/2) for that match, plus
    whether the selected... no, ANY team's advancing status held in each
    of those simulations. This lets conditional advance probabilities
    ("given this match goes this way, how often does team T advance?") be
    derived from the *same* simulation run, without resampling anything.
    """
    remaining_by_group = defaultdict(list)
    for m in matches:
        if m.get("status") != "finished":
            g = m.get("group", "")
            remaining_by_group[g].append(m)

    locked = {g for g, tbl in standings.items() if all(r["p"] >= 3 for r in tbl)}

    all_rank_dists: dict[str, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))
    all_found: set[str] = set()

    # match_id -> outcome ("1"/"X"/"2") -> team -> count of sims (in that
    # outcome bucket) where team finished in an advancing position.
    match_cond_advances: dict[str, dict[str, defaultdict[str, int]]] = (
        defaultdict(lambda: {"1": defaultdict(int), "X": defaultdict(int), "2": defaultdict(int)})
        if track_matches else {}
    )
    # match_id -> outcome -> count of sims that sampled that outcome (denominator)
    match_outcome_totals: dict[str, defaultdict[str, int]] = (
        defaultdict(lambda: defaultdict(int)) if track_matches else {}
    )

    for _ in range(n):
        all_thirds = []
        sim_sampled: dict[str, str] = {}
        for gname, table in standings.items():
            if gname in locked:
                if len(table) >= 3:
                    all_thirds.append(table[2])
                continue
            remaining = remaining_by_group.get(gname, [])
            sampled: dict[str, str] = {}
            for m in remaining:
                sampled[str(m.get("id", ""))] = sample_result(rating(m["team1"]), rating(m["team2"]))
            if track_matches:
                sim_sampled.update(sampled)
            cur_tbl = {r["team"]: dict(r) for r in table}
            for m in remaining:
                r1, r2 = rating(m["team1"]), rating(m["team2"])
                g1, g2 = typical_score(sampled.get(str(m["id"]), "X"), r1, r2)
                for name, gf, ga in [(m["team1"], g1, g2), (m["team2"], g2, g1)]:
                    if name in cur_tbl:
                        row = cur_tbl[name]
                        row["p"] += 1
                        row["gf"] += gf
                        row["ga"] += ga
                        row["gd"] = row["gf"] - row["ga"]
                        res = sampled.get(str(m["id"]), "X")
                        if res == "1" and name == m["team1"]:
                            row["w"] += 1; row["pts"] += 3
                        elif res == "2" and name == m["team2"]:
                            row["w"] += 1; row["pts"] += 3
                        elif res == "X":
                            row["d"] += 1; row["pts"] += 1
                        else:
                            row["l"] += 1
            final = sorted(cur_tbl.values(), key=rank_key)
            if len(final) >= 3:
                all_thirds.append(final[2])

        all_thirds.sort(key=rank_key)
        advancing_teams: set[str] = set()
        for gname, table in standings.items():
            for idx, row in enumerate(table[:2]):
                advancing_teams.add(row["team"])
        for i, t in enumerate(all_thirds):
            all_rank_dists[t["team"]][i + 1] += 1
            all_found.add(t["team"])
            if i + 1 <= ADVANCE_SLOTS:
                advancing_teams.add(t["team"])

        if track_matches:
            for mid, res in sim_sampled.items():
                match_outcome_totals[mid][res] += 1
                for team in advancing_teams:
                    match_cond_advances[mid][res][team] += 1

    result: dict[str, dict] = {}
    for team in all_found:
        dist = dict(all_rank_dists[team])
        total = sum(dist.values())
        median_r = 1
        cum = 0
        for r in range(1, 13):
            cum += dist.get(r, 0)
            if cum >= total * 0.5:
                median_r = r
                break
        advances = sum(v for r, v in dist.items() if r <= ADVANCE_SLOTS)
        result[team] = {
            "prob": round(advances / total * 100, 1),
            "avg_rank": round(sum(r * c for r, c in dist.items()) / total, 2),
            "median_rank": median_r,
            "rank_dist": dict(sorted(dist.items())),
            "total_sims": n,
        }

    for gname, table in standings.items():
        for idx, row in enumerate(table):
            t = row["team"]
            if t in result:
                continue
            if idx <= 1:
                result[t] = {"prob": 100.0, "avg_rank": 1.0, "median_rank": 1, "rank_dist": {1: n}, "total_sims": n, "note": "Already advancing (top 2 in group)"}
            else:
                result[t] = {"prob": 0.0, "avg_rank": 12, "median_rank": 12, "rank_dist": {}, "total_sims": n, "note": "Eliminated"}

    if track_matches:
        cond: dict[str, dict] = {}
        for mid, outcome_totals in match_outcome_totals.items():
            cond[mid] = {}
            for outcome in ("1", "X", "2"):
                denom = outcome_totals.get(outcome, 0)
                cond[mid][outcome] = {
                    "n_sims": denom,
                    "advance_pct": {
                        team: round(cnt / denom * 100, 1)
                        for team, cnt in match_cond_advances[mid][outcome].items()
                    } if denom else {},
                }
        result["_conditional"] = cond

    return result


def compute_scenarios(standings: dict, matches: list[dict], country: str = "South Korea") -> dict:
    remaining_by_group = defaultdict(list)
    for m in matches:
        if m.get("status") != "finished":
            g = m.get("group", "")
            remaining_by_group[g].append(m)

    locked = {g for g, tbl in standings.items() if all(r["p"] >= 3 for r in tbl)}

    country_group = None
    for gname, table in standings.items():
        for row in table:
            if row["team"] == country:
                country_group = gname
                break
        if country_group:
            break

    def simulate_group(table, remaining, outcomes_dict) -> tuple:
        cur = {r["team"]: dict(r) for r in table}
        for m in remaining:
            res = outcomes_dict.get(str(m["id"]), "X")
            r1, r2 = rating(m["team1"]), rating(m["team2"])
            g1, g2 = typical_score(res, r1, r2)
            for name, gf, ga in [(m["team1"], g1, g2), (m["team2"], g2, g1)]:
                if name in cur:
                    row = cur[name]
                    row["p"] += 1
                    row["gf"] += gf
                    row["ga"] += ga
                    row["gd"] = row["gf"] - row["ga"]
                    if res == "1" and name == m["team1"]:
                        row["w"] += 1; row["pts"] += 3
                    elif res == "2" and name == m["team2"]:
                        row["w"] += 1; row["pts"] += 3
                    elif res == "X":
                        row["d"] += 1; row["pts"] += 1
                    else:
                        row["l"] += 1
        final = sorted(cur.values(), key=rank_key)
        return (final[2] if len(final) >= 3 else None, final)

    def extreme_third(table, remaining, maximize):
        if not remaining:
            return simulate_group(table, [], {})
        outcomes = ["1", "X", "2"]
        best, best_k = None, None
        for combo in itertools.product(outcomes, repeat=len(remaining)):
            od = {str(m["id"]): res for m, res in zip(remaining, combo)}
            third, _ = simulate_group(table, remaining, od)
            if not third:
                continue
            k = (third["pts"], third["gd"], third["gf"])
            if best is None or (maximize and k > best_k) or (not maximize and k < best_k):
                best, best_k = (third, od), k
        return best

    def likely_group(table, remaining):
        if not remaining:
            return simulate_group(table, [], {})
        od = {}
        for m in remaining:
            pa, pd, pb = elo_prob(rating(m["team1"]), rating(m["team2"]))
            if pa >= pd and pa >= pb:
                od[str(m["id"])] = "1"
            elif pb >= pa and pb >= pd:
                od[str(m["id"])] = "2"
            else:
                od[str(m["id"])] = "X"
        third, _ = simulate_group(table, remaining, od)
        return (third, od)

    def extreme_country(table, remaining, maximize):
        outcomes = ["1", "X", "2"]
        best_od, best_pos, best_third = None, None, None
        for combo in itertools.product(outcomes, repeat=len(remaining)):
            od = {str(m["id"]): res for m, res in zip(remaining, combo)}
            third, final = simulate_group(table, remaining, od)
            pos = next((i + 1 for i, r in enumerate(final) if r["team"] == country), None)
            if pos is None:
                continue
            if best_pos is None or (maximize and pos > best_pos) or (not maximize and pos < best_pos):
                best_od, best_pos, best_third = od, pos, third
        return best_od, best_third, best_pos

    def simulate_country_group(strat, table, rem):
        if strat == "best":
            od, third, pos = extreme_country(table, rem, False)
        elif strat == "worst":
            od, third, pos = extreme_country(table, rem, True)
        else:
            third, od = likely_group(table, rem)
            _, final = simulate_group(table, rem, od)
            pos = next((i + 1 for i, r in enumerate(final) if r["team"] == country), None)
        return od, third, pos

    def build(strat):
        thirds = []
        country_pos = None
        for gname, table in standings.items():
            if gname in locked:
                if len(table) >= 3:
                    thirds.append(table[2])
                if gname == country_group:
                    for i, r in enumerate(table):
                        if r["team"] == country:
                            country_pos = i + 1
                            break
                continue
            rem = remaining_by_group.get(gname, [])
            if gname == country_group:
                od, third, country_pos = simulate_country_group(strat, table, rem)
                if third:
                    third["group"] = gname
                    thirds.append(third)
            else:
                if strat == "best":
                    res = extreme_third(table, rem, False)
                elif strat == "worst":
                    res = extreme_third(table, rem, True)
                else:
                    res = likely_group(table, rem)
                if res:
                    third, _ = res
                    if third:
                        third["group"] = gname
                        thirds.append(third)
        thirds.sort(key=rank_key)
        return thirds, country_pos

    def country_rank(thirds):
        for i, t in enumerate(thirds):
            if t["team"] == country:
                return i + 1
        return None

    def scenario_data(rank, group_pos):
        if group_pos is not None and group_pos <= 2:
            return {"rank": None, "adv": True, "position": group_pos, "status": "top2"}
        elif rank is not None:
            return {"rank": rank, "adv": rank <= ADVANCE_SLOTS, "position": group_pos, "status": "third"}
        else:
            return {"rank": None, "adv": False, "position": group_pos, "status": "eliminated"}

    b, bpos = build("best")
    w, wpos = build("worst")
    l, lpos = build("likely")
    br, wr, lr = country_rank(b), country_rank(w), country_rank(l)
    return {
        "best": scenario_data(br, bpos),
        "worst": scenario_data(wr, wpos),
        "likely": scenario_data(lr, lpos),
    }


def compute_criticality(match: dict, standings: dict, country: str = "South Korea", advance_prob: float | None = None) -> str:
    if advance_prob is not None and (advance_prob >= 100 or advance_prob <= 0):
        return "grey"

    country_group = None
    country_pos = None
    for gname, tbl in standings.items():
        for pos, r in enumerate(tbl):
            if r["team"] == country:
                country_group = gname
                country_pos = pos + 1
                break
        if country_group:
            break

    if country_group is None:
        return "grey"

    match_group = match.get("group", "")

    if match_group == country_group:
        if country in (match["team1"], match["team2"]):
            return "red"
        if country_pos == 3:
            table = standings.get(match_group, [])
            country_row = next((r for r in table if r["team"] == country), None)
            if country_row:
                for row in table:
                    if row["team"] in (match["team1"], match["team2"]):
                        if row["pts"] + 3 > country_row["pts"] or (
                            row["pts"] + 3 == country_row["pts"] and row["gd"] + 2 >= country_row["gd"]
                        ):
                            return "red"
            return "orange"
        return "yellow"

    if country_pos != 3:
        return "grey"

    thirds = get_third_placed(standings)
    target_row = next((t for t in thirds if t["team"] == country), None)
    if not target_row:
        return "grey"

    target_pts = target_row["pts"]
    target_gd = target_row["gd"]

    match_table = standings.get(match_group, [])
    if len(match_table) < 3:
        return "grey"

    current_third = match_table[2]
    t1_row = next((r for r in match_table if r["team"] == match["team1"]), None)
    t2_row = next((r for r in match_table if r["team"] == match["team2"]), None)

    if current_third["team"] in (match["team1"], match["team2"]):
        winner_pts = current_third["pts"] + 3
        if winner_pts > target_pts or (winner_pts == target_pts and current_third["gd"] + 2 >= target_gd):
            return "red"
        return "orange"

    if t1_row and t2_row:
        for winner, loser in [(t1_row, t2_row), (t2_row, t1_row)]:
            others = [r for r in match_table if r["team"] not in (match["team1"], match["team2"])]
            projected = sorted(
                others + [
                    {**winner, "pts": winner["pts"] + 3},
                    {**loser},
                ],
                key=rank_key
            )
            if len(projected) >= 3:
                new_third = projected[2]
                if new_third["pts"] > target_pts or (
                    new_third["pts"] == target_pts and new_third["gd"] >= target_gd
                ):
                    return "orange"

    return "yellow"


# ─── WHAT-IF ANALYSIS ──────────────────────────────────────────────────────


def compute_outcome(standings: dict, match: dict, country: str, outcome: str, score1: int | None = None, score2: int | None = None) -> dict:
    group = match.get("group", "")
    table = standings.get(group, [])
    new_table = apply_match_outcome(table, match, outcome, score1, score2)
    new_standings = {**standings, group: new_table}
    new_thirds = get_third_placed(new_standings)
    tr = next((i + 1 for i, t in enumerate(new_thirds) if t["team"] == country), None)
    pos = next((i + 1 for i, r in enumerate(new_table) if r["team"] == country), None)
    advancing = (pos is not None and pos <= 2) or (tr is not None and tr <= ADVANCE_SLOTS)
    g1, g2 = (score1, score2) if score1 is not None else typical_score(outcome, rating(match["team1"]), rating(match["team2"]))
    return {"country_position": pos, "third_place_rank": tr, "advancing": advancing, "score": [g1, g2]}


def what_if(standings: dict, matches: list[dict], country: str, match_id: Any, outcome: str, score1: int | None = None, score2: int | None = None) -> dict:
    match = next((m for m in matches if str(m.get("id")) == str(match_id)), None)
    if not match:
        return {"error": "Match not found"}

    group = match.get("group", "")
    current_thirds = get_third_placed(standings)

    current_third_rank = None
    for i, t in enumerate(current_thirds):
        if t["team"] == country:
            current_third_rank = i + 1
            break

    current_status = "unknown"
    for gname, table in standings.items():
        for idx, row in enumerate(table):
            if row["team"] == country:
                pos = idx + 1
                if pos <= 2:
                    current_status = "advancing"
                elif pos == 3:
                    current_status = f"3rd (rank {current_third_rank}/12)" if current_third_rank else "3rd"
                else:
                    current_status = "eliminated"
                break

    if group in standings:
        new_table = apply_match_outcome(standings[group], match, outcome, score1, score2)
        new_standings = {**standings, group: new_table}
    else:
        new_standings = standings

    new_thirds = get_third_placed(new_standings)
    new_third_rank = None
    for i, t in enumerate(new_thirds):
        if t["team"] == country:
            new_third_rank = i + 1
            break

    new_status = "unknown"
    for gname, table in new_standings.items():
        for idx, row in enumerate(table):
            if row["team"] == country:
                pos = idx + 1
                if pos <= 2:
                    new_status = "advancing"
                elif pos == 3:
                    new_status = f"3rd (rank {new_third_rank}/12)" if new_third_rank else "3rd"
                else:
                    new_status = "eliminated"
                break

    g1, g2 = (score1, score2) if score1 is not None else typical_score(outcome, rating(match["team1"]), rating(match["team2"]))

    return {
        "current": {
            "third_place_rank": current_third_rank,
            "is_advancing": current_third_rank is not None and current_third_rank <= ADVANCE_SLOTS,
            "status_display": current_status,
        },
        "new": {
            "third_place_rank": new_third_rank,
            "is_advancing": new_third_rank is not None and new_third_rank <= ADVANCE_SLOTS,
            "status_display": new_status,
        },
        "match": {
            "team1": match["team1"],
            "team2": match["team2"],
            "score": match["score"],
            "group": group,
        },
        "outcome_applied": outcome,
        "applied_score": [g1, g2],
        "current_standings": current_thirds,
        "new_standings": new_thirds,
    }


def _margin_to_score(margin: int) -> tuple[int, int, str]:
    """Convert a home-team goal margin into a (score1, score2, outcome) triple."""
    if margin > 0:
        return margin, 0, "1"
    elif margin == 0:
        return 0, 0, "X"
    else:
        return 0, abs(margin), "2"


def _gd_sweep(standings: dict, match: dict, country: str, rival_team: str | None = None) -> list[dict] | None:
    """
    Sweep the home-team goal margin for `match` (-10..+10, where positive
    means team1/home wins by that margin) and return cells showing how the
    selected `country`'s rank and advancing status respond at each scoreline.
    """
    by_margin: dict[int, dict] = {}
    for margin in range(-10, 11):
        s1, s2, out = _margin_to_score(margin)
        res = compute_outcome(standings, match, country, out, s1, s2)
        by_margin[margin] = {
            "margin": margin,
            "score": [s1, s2],
            "rank": res["third_place_rank"],
            "adv": res["advancing"],
        }

    margins = sorted(by_margin.keys())
    ranks = [by_margin[m]["rank"] for m in margins]
    outcomes = [by_margin[m]["adv"] for m in margins]

    rank_varies = len(set(ranks)) > 1
    adv_varies = len(set(outcomes)) > 1

    if not rank_varies and not adv_varies:
        return None

    changed_idxs = [
        i for i in range(1, len(margins))
        if ranks[i] != ranks[i - 1] or outcomes[i] != outcomes[i - 1]
    ]

    if changed_idxs:
        first_change = margins[changed_idxs[0] - 1]
        last_change = margins[changed_idxs[-1]]
        start = max(min(margins), first_change - 1)
        end = min(max(margins), last_change + 1)
    else:
        start, end = -3, 3

    return [by_margin[m] for m in margins if start <= m <= end]


def conditional_outcome_probs(mc_conditional: dict, match_id: Any, country: str) -> dict | None:
    """For a given match and country, return how often `country` advances
    in each of the simulation buckets where that match went 1/X/2 — i.e.
    the conditional Monte Carlo probabilities, derived from the cached
    full-tournament simulation rather than a frozen single-match splice.

    Returns None if this match wasn't part of the tracked simulation
    (e.g. it's already finished, or matches between cache refreshes).
    """
    cond = mc_conditional.get(str(match_id))
    if not cond:
        return None
    out = {}
    for outcome in ("1", "X", "2"):
        bucket = cond.get(outcome, {})
        n_sims = bucket.get("n_sims", 0)
        # advance_pct only contains entries for teams that advanced in at
        # least one tracked simulation; absence means 0%, not "no data" —
        # only an empty/zero n_sims bucket means we truly have nothing.
        pct = bucket.get("advance_pct", {}).get(country, 0.0 if n_sims else None)
        out[outcome] = {
            "advance_pct": pct,
            "n_sims": n_sims,
        }
    return out


def compute_outcomes_for_match(standings: dict, match: dict, country: str, mc_conditional: dict | None = None) -> dict:
    group = match.get("group", "")
    involves_country = country in (match["team1"], match["team2"])
    result: dict[str, Any] = {
        "id": match["id"],
        "team1": match["team1"],
        "team2": match["team2"],
        "involves_country": involves_country,
        "status": match.get("status", "upcoming"),
        "date": match.get("date", ""),
        "group": group,
        "score": match.get("score"),
    }

    for out in ["1", "X", "2"]:
        result[out] = compute_outcome(standings, match, country, out)

    if mc_conditional:
        cond = conditional_outcome_probs(mc_conditional, match["id"], country)
        if cond:
            for out in ["1", "X", "2"]:
                result[out]["sim_advance_pct"] = cond[out]["advance_pct"]
                result[out]["sim_n"] = cond[out]["n_sims"]

    if involves_country:
        is_t1 = match["team1"] == country
        win_out = "1" if is_t1 else "2"
        loss_out = "2" if is_t1 else "1"
        by_margin = {}
        for margin in range(-10, 11):
            if margin > 0:
                s1, s2 = (margin, 0) if is_t1 else (0, margin)
                out = win_out
            elif margin == 0:
                s1, s2 = 0, 0
                out = "X"
            else:
                abs_m = abs(margin)
                s1, s2 = (0, abs_m) if is_t1 else (abs_m, 0)
                out = loss_out
            res = compute_outcome(standings, match, country, out, s1, s2)
            by_margin[margin] = {"margin": margin, "score": [s1, s2], "rank": res["third_place_rank"], "adv": res["advancing"]}

        margins = sorted(by_margin.keys())
        ranks = [by_margin[m]["rank"] for m in margins]
        outcomes = [by_margin[m]["adv"] for m in margins]
        rank_varies = len(set(ranks)) > 1
        adv_varies = len(set(outcomes)) > 1

        if rank_varies or adv_varies:
            changed_idxs = [
                i for i in range(1, len(margins))
                if ranks[i] != ranks[i - 1] or outcomes[i] != outcomes[i - 1]
            ]
            if changed_idxs:
                first_change = margins[changed_idxs[0] - 1]
                last_change = margins[changed_idxs[-1]]
                start = max(min(margins), first_change - 1)
                end = min(max(margins), last_change + 1)
            else:
                start, end = -3, 3
            result["gd"] = [by_margin[m] for m in margins if start <= m <= end]
            result["gd_perspective"] = "country"
    else:
        cells = _gd_sweep(standings, match, country)
        if cells:
            result["gd"] = cells
            result["gd_perspective"] = "match"

    return result


def compute_group_outcomes(standings: dict, matches: list[dict], country: str, mc_conditional: dict | None = None) -> dict:
    country_group = None
    for gname, table in standings.items():
        for row in table:
            if row["team"] == country:
                country_group = gname
                break
        if country_group:
            break

    if not country_group:
        return {"error": "Country not found in any group"}

    remaining = [m for m in matches if m.get("group") == country_group and m.get("status") != "finished"]

    return {
        "country": country,
        "group": country_group,
        "matches": [compute_outcomes_for_match(standings, m, country, mc_conditional) for m in remaining],
    }


def precompute_key_match_outcomes(standings: dict, matches: list[dict], country: str, mc_conditional: dict | None = None) -> list[dict]:
    """Pre-compute all three outcomes for every upcoming key match."""
    result = []
    for m in matches:
        if m.get("status") == "finished":
            continue
        outcome_data = compute_outcomes_for_match(standings, m, country, mc_conditional)
        result.append(outcome_data)
    return result


# ─── SIMULATE ALL ────────────────────────────────────────────────────────────


def simulate_all(standings: dict, matches: list[dict], country: str) -> dict:
    remaining_by_group = defaultdict(list)
    for m in matches:
        if m.get("status") != "finished":
            g = m.get("group", "")
            remaining_by_group[g].append(m)

    locked = {g for g, tbl in standings.items() if all(r["p"] >= 3 for r in tbl)}
    all_sampled: dict[str, str] = {}
    all_thirds = []

    for gname, table in standings.items():
        if gname in locked:
            if len(table) >= 3:
                third = dict(table[2])
                third["group"] = gname
                all_thirds.append(third)
            continue

        remaining = remaining_by_group.get(gname, [])
        sampled = {}
        for m in remaining:
            res = sample_result(rating(m["team1"]), rating(m["team2"]))
            mid = str(m.get("id", ""))
            sampled[mid] = res
            all_sampled[mid] = res

        cur_tbl = {r["team"]: dict(r) for r in table}
        for m in remaining:
            r1, r2 = rating(m["team1"]), rating(m["team2"])
            g1, g2 = typical_score(sampled.get(str(m["id"]), "X"), r1, r2)
            for name, gf, ga in [(m["team1"], g1, g2), (m["team2"], g2, g1)]:
                if name in cur_tbl:
                    row = cur_tbl[name]
                    row["p"] += 1
                    row["gf"] += gf
                    row["ga"] += ga
                    row["gd"] = row["gf"] - row["ga"]
                    res = sampled.get(str(m["id"]), "X")
                    if res == "1" and name == m["team1"]:
                        row["w"] += 1; row["pts"] += 3
                    elif res == "2" and name == m["team2"]:
                        row["w"] += 1; row["pts"] += 3
                    elif res == "X":
                        row["d"] += 1; row["pts"] += 1
                    else:
                        row["l"] += 1
        final = sorted(cur_tbl.values(), key=rank_key)
        if len(final) >= 3:
            third = dict(final[2])
            third["group"] = gname
            all_thirds.append(third)

    all_thirds.sort(key=rank_key)

    country_rank = None
    for i, t in enumerate(all_thirds):
        if t["team"] == country:
            country_rank = i + 1
            break

    match_results = []
    for m in matches:
        if m.get("status") != "finished":
            g = m.get("group", "")
            if not g:
                continue
            mid = str(m.get("id", ""))
            is_live = m.get("status") == "live"
            if is_live:
                res = "X"
                g1, g2 = m.get("score", [0, 0])
            else:
                res = all_sampled.get(mid, "X")
                r1, r2 = rating(m["team1"]), rating(m["team2"])
                g1, g2 = typical_score(res, r1, r2)
            match_results.append({
                "team1": m["team1"],
                "team2": m["team2"],
                "score": [g1, g2],
                "group": g,
                "status": "live" if is_live else "simulated",
                "actual_score": m.get("score"),
                "live_minute": m.get("live_minute") if is_live else None,
            })
    match_results.sort(key=lambda x: x["group"])

    return {
        "standings": all_thirds,
        "country_rank": country_rank,
        "is_advancing": country_rank is not None and country_rank <= ADVANCE_SLOTS,
        "match_results": match_results,
    }


# ─── FLASK APP ──────────────────────────────────────────────────────────────

app = Flask(__name__)

_LATEST_DATA: dict = {}
_ALL_TEAMS: list[str] = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/teams")
def api_teams():
    return jsonify({"teams": _ALL_TEAMS})


@app.route("/api/mode", methods=["GET", "POST"])
def api_mode():
    global SIM_MODE
    if request.method == "POST":
        mode = request.json.get("mode", "elo")
        if mode in ("elo", "equal"):
            SIM_MODE = mode
    return jsonify({"mode": SIM_MODE})


@app.route("/api/status")
def api_status():
    """Lightweight endpoint: returns whether any match is currently live.

    The frontend uses this for adaptive polling — it hits /api/status every
    30 s and only calls the heavier /api/data when a match is actually live
    (or every 5 minutes otherwise).  No Monte Carlo, no scenario computation.
    """
    matches = _LATEST_DATA.get("all_matches", _LATEST_DATA.get("matches", []))
    has_live = any(m.get("status") == "live" for m in matches)
    return jsonify({"live": has_live, "updated": _LATEST_DATA.get("updated", time.time())})


@app.route("/api/data")
def api_data():
    country = request.args.get("country", "South Korea")
    data = _LATEST_DATA
    standings = data.get("standings", {})
    matches = data.get("matches", [])

    thirds = get_third_placed(standings)
    ts = team_status(country, standings)

    global _MC_CACHE, _MC_CACHE_HASH, SIM_MODE
    req_mode = request.args.get("mode", "equal")
    if req_mode not in ("elo", "equal"):
        req_mode = "equal"
    h = _state_hash(standings)
    if h != _MC_CACHE_HASH:
        saved_mode = SIM_MODE
        SIM_MODE = "elo"
        elo_res = run_monte_carlo(standings, matches, track_matches=True)
        SIM_MODE = "equal"
        equal_res = run_monte_carlo(standings, matches, track_matches=True)
        SIM_MODE = saved_mode
        _MC_CACHE = {"elo": elo_res, "equal": equal_res}
        _MC_CACHE_HASH = h
    mc_full = _MC_CACHE.get(req_mode, {})
    mc = mc_full.get(country, {"prob": 0, "avg_rank": 12, "median_rank": 12, "rank_dist": {}, "total_sims": MONTE_CARLO_ITERATIONS, "note": "Not in 3rd-place standings"})
    mc_conditional = mc_full.get("_conditional", {})

    SIM_MODE = req_mode
    scenarios = compute_scenarios(standings, matches, country)

    country_prob = mc.get("prob", 0)
    key_all = [m for m in matches if m.get("status") != "finished"]
    for m in key_all:
        m["severity"] = compute_criticality(m, standings, country, country_prob)

    live_raw = [m for m in key_all if m.get("status") == "live"]
    key_upcoming = [m for m in key_all if m.get("status") != "live"]

    remaining_counts: dict[str, int] = {}
    for m in matches:
        if m.get("status") != "finished":
            remaining_counts[m["team1"]] = remaining_counts.get(m["team1"], 0) + 1
            remaining_counts[m["team2"]] = remaining_counts.get(m["team2"], 0) + 1

    key_outcomes = precompute_key_match_outcomes(standings, key_upcoming, country, mc_conditional)
    live_outcomes = precompute_key_match_outcomes(standings, live_raw, country, mc_conditional)

    severity_map = {str(m["id"]): m["severity"] for m in key_upcoming}
    for ko in key_outcomes:
        ko["severity"] = severity_map.get(str(ko["id"]), "grey")

    live_severity_map = {str(m["id"]): m["severity"] for m in live_raw}
    live_minute_map = {str(m["id"]): m.get("live_minute") for m in live_raw}
    for lo in live_outcomes:
        lo["severity"] = live_severity_map.get(str(lo["id"]), "grey")
        lo["live_minute"] = live_minute_map.get(str(lo["id"]))

    whatif = compute_group_outcomes(standings, matches, country)

    return jsonify({
        "standings": thirds,
        "country_status": ts,
        "monte_carlo": mc,
        "scenarios": scenarios,
        "whatif": whatif,
        "live_matches": live_outcomes,
        "key_matches": key_outcomes,
        "remaining_counts": remaining_counts,
        "updated": data.get("updated", time.time()),
    })


@app.route("/api/whatif")
def api_whatif():
    country = request.args.get("country", "South Korea")
    match_id = request.args.get("match_id")
    outcome = request.args.get("outcome", "X")
    data = _LATEST_DATA
    standings = data.get("standings", {})
    matches = data.get("matches", [])

    if match_id:
        s1 = request.args.get("score1")
        s2 = request.args.get("score2")
        if s1 is not None and s2 is not None:
            try:
                result = what_if(standings, matches, country, match_id, outcome, int(s1), int(s2))
            except ValueError:
                result = what_if(standings, matches, country, match_id, outcome)
        else:
            result = what_if(standings, matches, country, match_id, outcome)
    else:
        result = compute_group_outcomes(standings, matches, country)

    return jsonify(result)


@app.route("/api/simulate")
def api_simulate():
    country = request.args.get("country", "South Korea")
    data = _LATEST_DATA
    result = simulate_all(data.get("standings", {}), data.get("matches", []), country)
    return jsonify(result)


def refresh_data():
    global _LATEST_DATA, _ALL_TEAMS
    data = fetch_all_data()
    if data.get("standings"):
        _LATEST_DATA = data
        _ALL_TEAMS = data.get("all_teams", [])
        try:
            cache_path = Path(__file__).with_name("wc_cache.json")
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

# ─── MAIN ────────────────────────────────────────────────────────────────────

def _bg_refresh():
    consecutive_errors = 0
    while True:
        matches = _LATEST_DATA.get("all_matches", _LATEST_DATA.get("matches", []))
        has_live = any(m.get("status") == "live" for m in matches)
        base_sleep = REFRESH_SECS_LIVE if has_live else REFRESH_SECS_IDLE
        sleep_secs = min(base_sleep * (2 ** consecutive_errors), REFRESH_SECS_IDLE)
        time.sleep(sleep_secs)
        try:
            refresh_data()
            consecutive_errors = 0
        except Exception:
            consecutive_errors += 1
            print(f"[bg_refresh] refresh_data() raised an exception (consecutive errors: {consecutive_errors}):")
            traceback.print_exc()


def _warm_mc_cache():
    data = _LATEST_DATA
    standings = data.get("standings", {})
    matches = data.get("matches", [])
    if not standings or not matches:
        return
    global _MC_CACHE, _MC_CACHE_HASH, SIM_MODE
    h = _state_hash(standings)
    saved = SIM_MODE
    SIM_MODE = "elo"
    elo_res = run_monte_carlo(standings, matches, track_matches=True)
    SIM_MODE = "equal"
    equal_res = run_monte_carlo(standings, matches, track_matches=True)
    SIM_MODE = saved
    _MC_CACHE = {"elo": elo_res, "equal": equal_res}
    _MC_CACHE_HASH = h
    print(f"  Monte Carlo warmed ({MONTE_CARLO_ITERATIONS:,}×2 modes)")


def _startup():
    """Run once at import time (works under both gunicorn and __main__)."""
    global _LATEST_DATA, _ALL_TEAMS
    print("🌐 Fetching initial World Cup data...")
    try:
        refresh_data()
    except Exception:
        print("[startup] refresh_data() failed:")
        traceback.print_exc()

    if not _LATEST_DATA.get("standings"):
        cache_path = Path(__file__).with_name("wc_cache.json")
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                _LATEST_DATA = cached
                _ALL_TEAMS = cached.get("all_teams", [])
                print("  Loaded cached data (API unreachable at startup)")
            except Exception:
                pass

    if not _LATEST_DATA.get("standings"):
        print("⚠️  Warning: No tournament data available. Starting with empty data.")
    else:
        print(f"  {len(_ALL_TEAMS)} teams loaded.")

    t = threading.Thread(target=_bg_refresh, daemon=True)
    t.start()
    print("  Background refresh thread started.")


_startup()


if __name__ == "__main__":
    # _startup() already ran at import time (initial fetch + background thread).
    # Just warm the Monte Carlo cache and launch the dev server.
    if _LATEST_DATA.get("standings"):
        print(f"  Warming Monte Carlo ({MONTE_CARLO_ITERATIONS:,} × 2 modes)...")
        _warm_mc_cache()

    print(f"  Starting server at http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
