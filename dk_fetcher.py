import re
from datetime import datetime
import pandas as pd
import requests


def _parse_dk_date(date_val):
    if not date_val:
        return ""
    try:
        if "/Date(" in str(date_val):
            epoch_ms = int(re.search(r"\d+", str(date_val)).group())
            return datetime.fromtimestamp(epoch_ms / 1000).strftime("%a %m/%d %I:%M %p")
        cleaned_iso = str(date_val).replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned_iso).strftime("%a %m/%d %I:%M %p")
    except Exception:
        return ""


def get_vegas_game_data(team_a: str, team_b: str) -> dict:
    default_data = {
        "spread": 0.0,
        "over_under": 45.0,
        "favorite": team_a,
        "underdog": team_b,
        "fav_implied": 22.5,
        "und_implied": 22.5,
        "recommended_script": "balanced"
    }

    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        if res.status_code != 200:
            return default_data

        events = res.json().get("events", [])
        team_abbr_map = {"LAR": "LA", "WSH": "WAS"}

        t1 = team_abbr_map.get(team_a, team_a).upper()
        t2 = team_abbr_map.get(team_b, team_b).upper()

        for ev in events:
            comps = ev.get("competitions", [{}])[0]
            competitors = comps.get("competitors", [])
            abbrs = [c.get("team", {}).get("abbreviation", "").upper() for c in competitors]

            if any(t in abbrs for t in [t1, team_a]) and any(t in abbrs for t in [t2, team_b]):
                odds_list = comps.get("odds", [])
                if not odds_list:
                    break

                odds = odds_list[0]
                ou = float(odds.get("overUnder") or 45.0)
                spread_val = float(odds.get("spread") or 0.0)

                home_team = next((c["team"]["abbreviation"] for c in competitors if c.get("homeAway") == "home"), team_a)
                away_team = next((c["team"]["abbreviation"] for c in competitors if c.get("homeAway") == "away"), team_b)

                home_dk = team_a if home_team in [t1, team_a] else team_b
                away_dk = team_b if home_dk == team_a else team_a

                if spread_val < 0:
                    favorite = home_dk
                    underdog = away_dk
                    spread_abs = abs(spread_val)
                elif spread_val > 0:
                    favorite = away_dk
                    underdog = home_dk
                    spread_abs = spread_val
                else:
                    favorite = team_a
                    underdog = team_b
                    spread_abs = 0.0

                fav_implied = round((ou + spread_abs) / 2.0, 1)
                und_implied = round((ou - spread_abs) / 2.0, 1)

                if spread_abs >= 6.5 and und_implied <= 18.0:
                    script = "5_1_onslaught"
                elif ou <= 40.0:
                    script = "defensive_slugfest"
                elif ou >= 47.0 and spread_abs <= 4.5:
                    script = "shootout"
                elif ou >= 47.0 and spread_abs > 4.5:
                    script = "underdog_chase"
                else:
                    script = "balanced"

                return {
                    "spread": spread_abs,
                    "over_under": ou,
                    "favorite": favorite,
                    "underdog": underdog,
                    "fav_implied": fav_implied,
                    "und_implied": und_implied,
                    "recommended_script": script
                }

    except Exception as err:
        print(f"[Vegas Fetcher] Fallback used: {err}")

    return default_data


def get_all_upcoming_nfl_slates():
    url = "https://www.draftkings.com/lobby/getcontests?sport=NFL"
    headers = {"User-Agent": "Mozilla/5.0"}

    res = requests.get(url, headers=headers, timeout=10)
    if res.status_code != 200:
        raise ConnectionError(f"DraftKings API returned HTTP {res.status_code}")

    data = res.json()
    contests = data.get("Contests", [])
    draft_groups = {dg.get("DraftGroupId"): dg for dg in data.get("DraftGroups", [])}

    slates = []
    seen_dg_ids = set()

    for c in contests:
        dg_id = c.get("dg")
        if not dg_id or dg_id in seen_dg_ids:
            continue

        c_name = c.get("n", "")
        name_lower = c_name.lower()

        if any(unsupported in name_lower for unsupported in [
            "2nd half", "4th quarter", "2h", "4q", "snake", "tier", "single stat", "flash"
        ]):
            continue

        game_type_id = c.get("gameType")
        dg_info = draft_groups.get(dg_id, {})
        game_count = dg_info.get("GameCount", 0)

        is_showdown = (game_type_id == 96) or ("showdown" in name_lower) or (game_count == 1)
        slate_type = "showdown" if is_showdown else "classic"

        raw_date = c.get("sd") or dg_info.get("StartDateEst") or dg_info.get("StartDate")
        formatted_date = _parse_dk_date(raw_date)
        date_str = f" - {formatted_date}" if formatted_date else ""

        matchup_search = re.search(r"([A-Za-z0-9]{2,3}\s*(?:@|vs\.?)\s*[A-Za-z0-9]{2,3})", c_name, re.IGNORECASE)

        if is_showdown:
            if matchup_search:
                matchup = matchup_search.group(1).upper().replace("VS.", "@").replace("VS", "@")
                label = f"Showdown: {matchup}{date_str}"
            else:
                clean_title = re.sub(r"NFL\s*(Showdown)?\s*(\$[\d,KM]+)?", "", c_name, flags=re.IGNORECASE).strip(" -[]()")
                label = f"Showdown: {clean_title or 'Single Game'}{date_str}"
        else:
            tag = dg_info.get("DraftGroupTag") or "Classic Slate"
            count_str = f" ({game_count} Games)" if game_count else ""
            label = f"{tag}: {formatted_date}{count_str}".strip(" :")

        seen_dg_ids.add(dg_id)
        slates.append({
            "id": int(dg_id),
            "label": label,
            "type": slate_type
        })

    return sorted(slates, key=lambda x: (x["type"] != "showdown", x["label"]))


def _extract_fppg(draftable: dict) -> float:
    stats = draftable.get("draftStatAttributes", [])
    if not stats:
        return 0.0

    for item in stats:
        name = (str(item.get("id", "")) + " " + str(item.get("name", ""))).upper()
        if any(keyword in name for keyword in ["FPPG", "AVG", "PTS"]):
            raw_val = item.get("value")
            if raw_val is not None:
                try:
                    return float(str(raw_val).replace(",", "").strip())
                except (ValueError, TypeError):
                    continue

    for item in stats:
        raw_val = item.get("value")
        if raw_val is not None:
            val_str = str(raw_val).replace(",", "").strip()
            if any(val_str.endswith(sfx) for sfx in ["th", "st", "nd", "rd"]):
                continue
            try:
                return float(val_str)
            except (ValueError, TypeError):
                continue

    return 0.0


def get_slate_players(draft_group_id: int):
    url = f"https://api.draftkings.com/draftgroups/v1/draftgroups/{draft_group_id}/draftables?format=json"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    res = requests.get(url, headers=headers, timeout=10)
    if res.status_code != 200:
        raise ConnectionError(f"DraftKings API returned HTTP {res.status_code} for DraftGroup {draft_group_id}")

    draftables = res.json().get("draftables", [])
    if not draftables:
        return pd.DataFrame()

    is_showdown = any(
        any(attr.get("rosterSlot", {}).get("name") == "CPT"
            for attr in d.get("draftableRosterSlotAttributes", []))
        for d in draftables
    )

    matchup_map = {}
    for d in draftables:
        comp = d.get("competition") or (d.get("competitions")[0] if d.get("competitions") else None)
        if isinstance(comp, dict):
            text = comp.get("name") or comp.get("description") or ""
            if "@" in text or "VS" in text.upper():
                parts = re.split(r"\s+(?:@|vs\.?)\s+", text, flags=re.IGNORECASE)
                if len(parts) == 2:
                    t1, t2 = parts[0].strip().upper(), parts[1].strip().upper()
                    matchup_map[t1] = t2
                    matchup_map[t2] = t1

    players_by_id = {}

    for d in draftables:
        p_id = d.get("playerId") or d.get("draftableId")
        if not p_id:
            continue

        status = str(d.get("status", "")).strip().upper()
        if status in ["O", "IR", "OUT", "PUP", "SUS", "D"]:
            continue

        salary = int(d.get("salary") or 0)
        if salary <= 0:
            for attr in d.get("draftableRosterSlotAttributes", []):
                if attr.get("salary"):
                    salary = int(attr["salary"])
                    break

        if salary < 600:
            continue

        pos = str(d.get("position", "UTIL")).upper().strip()
        if pos in ["D", "DEF", "DST"]:
            pos = "DST"

        if pos in ["UTIL", "FLEX"]:
            for attr in d.get("draftableRosterSlotAttributes", []):
                slot_name = str(attr.get("rosterSlot", {}).get("name", "")).upper()
                if slot_name in ["DST", "DEF", "D"]:
                    pos = "DST"
                    break

        if pos == "QB":
            if is_showdown and salary < 7500:
                continue
            elif not is_showdown and salary < 4800:
                continue

        team_abbr = str(d.get("teamAbbreviation", "UNK")).upper().strip()
        opp_abbr = matchup_map.get(team_abbr, "UNK")

        if p_id in players_by_id:
            if salary < players_by_id[p_id]["salary"]:
                players_by_id[p_id]["salary"] = salary
            continue

        avg_fpts = _extract_fppg(d)
        if avg_fpts <= 0.0:
            avg_fpts = 1.0

        players_by_id[p_id] = {
            "id": p_id,
            "name": d.get("displayName") or f"{d.get('firstName', '')} {d.get('lastName', '')}".strip() or "Unknown",
            "position": pos,
            "team": team_abbr,
            "opponent": opp_abbr,
            "salary": salary,
            "projected_points": avg_fpts
        }

    return pd.DataFrame(list(players_by_id.values()))