import re
from datetime import datetime
import pandas as pd
import requests

DK_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _parse_dk_date(date_val):
  if not date_val:
    return ""
  try:
    if "/Date(" in str(date_val):
      epoch_ms = int(re.search(r"\d+", str(date_val)).group())
      return datetime.fromtimestamp(epoch_ms / 1000).strftime(
          "%a %m/%d %I:%M %p"
      )
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
      "recommended_script": "balanced",
  }

  url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
  try:
    res = requests.get(url, headers=DK_HEADERS, timeout=7)
    if res.status_code != 200:
      return default_data

    events = res.json().get("events", [])
    team_abbr_map = {"LAR": "LA", "WSH": "WAS"}

    t1 = team_abbr_map.get(team_a, team_a).upper()
    t2 = team_abbr_map.get(team_b, team_b).upper()

    for ev in events:
      comps = ev.get("competitions", [{}])[0]
      competitors = comps.get("competitors", [])
      abbrs = [
          c.get("team", {}).get("abbreviation", "").upper() for c in competitors
      ]

      if any(t in abbrs for t in [t1, team_a]) and any(
          t in abbrs for t in [t2, team_b]
      ):
        odds_list = comps.get("odds", [])
        if not odds_list:
          break

        odds = odds_list[0]
        ou = float(odds.get("overUnder") or 45.0)
        spread_val = float(odds.get("spread") or 0.0)

        home_team = next(
            (
                c["team"]["abbreviation"]
                for c in competitors
                if c.get("homeAway") == "home"
            ),
            team_a,
        )
        away_team = next(
            (
                c["team"]["abbreviation"]
                for c in competitors
                if c.get("homeAway") == "away"
            ),
            team_b,
        )

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
            "recommended_script": script,
        }

  except Exception as err:
    print(f"[Vegas Fetcher] Fallback used: {err}")

  return default_data


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
  res = requests.get(url, headers=DK_HEADERS, timeout=12)
  if res.status_code != 200:
    raise ConnectionError(
        f"DraftKings API returned HTTP {res.status_code} for DraftGroup"
        f" {draft_group_id}"
    )

  draftables = res.json().get("draftables", [])
  if not draftables:
    return pd.DataFrame()

  is_showdown = any(
      any(
          attr.get("rosterSlot", {}).get("name") == "CPT"
          for attr in d.get("draftableRosterSlotAttributes", [])
      )
      for d in draftables
  )

  matchup_map = {}
  for d in draftables:
    comp = d.get("competition") or (
        d.get("competitions")[0] if d.get("competitions") else None
    )
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
    if status in ["O", "IR", "OUT", "PUP", "SUS", "D", "INACTIVE", "NA"]:
      continue
    if d.get("isDisabled", False) or d.get("isSuspended", False):
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

    avg_fpts = _extract_fppg(d)
    if avg_fpts <= 0.0:
      continue

    # Filter out inactive depth on Classic slates
    if not is_showdown and pos != "DST" and salary <= 3000 and avg_fpts < 2.0:
      continue

    team_abbr = str(d.get("teamAbbreviation", "UNK")).upper().strip()
    opp_abbr = matchup_map.get(team_abbr, "UNK")

    # Record the true base/FLEX salary (lowest across CPT/FLEX records)
    if p_id in players_by_id:
      if salary < players_by_id[p_id]["salary"]:
        players_by_id[p_id]["salary"] = salary
      continue

    players_by_id[p_id] = {
        "id": p_id,
        "name": (
            d.get("displayName")
            or f"{d.get('firstName', '')} {d.get('lastName', '')}".strip()
            or "Unknown"
        ),
        "position": pos,
        "team": team_abbr,
        "opponent": opp_abbr,
        "salary": salary,
        "projected_points": avg_fpts,
    }

  player_list = list(players_by_id.values())

  # SYSTEMIC STARTING QB RULE:
  # On Showdown slates, retain ONLY the highest-priced QB per team.
  # Backups (e.g., Winston at $6k behind Dart at $9.6k) are eliminated completely.
  if is_showdown:
    qbs_by_team = {}
    for p in player_list:
      if p["position"] == "QB" and p["team"] != "UNK":
        qbs_by_team.setdefault(p["team"], []).append(p)

    backup_qb_ids = set()
    for team, team_qbs in qbs_by_team.items():
      if len(team_qbs) > 1:
        # Sort descending by base salary; keep only the top starter
        sorted_qbs = sorted(team_qbs, key=lambda x: x["salary"], reverse=True)
        for backup in sorted_qbs[1:]:
          backup_qb_ids.add(backup["id"])

    player_list = [p for p in player_list if p["id"] not in backup_qb_ids]

  return pd.DataFrame(player_list)