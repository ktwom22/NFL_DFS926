import random
import re
import pandas as pd
import pulp


def _clean_id(val) -> str:
  """Windows-safe identifier: strips characters illegal in temp filenames."""
  return re.sub(r"[^a-zA-Z0-9_]", "_", str(val))


class DraftKingsOptimizer:

  def __init__(self, player_df: pd.DataFrame, **kwargs):
    self.df = player_df.copy()
    self.df["position"] = (
        self.df["position"].astype(str).str.upper().str.strip()
    )
    self.df["position"] = self.df["position"].replace(
        {"D": "DST", "DEF": "DST"}
    )
    self.df["pulp_id"] = self.df["id"].apply(_clean_id)
    self.df["user_boost"] = 1.0

    # Apply Opportunity Ceiling Curve based on DraftKings Salary
    self.df["opportunity_ceiling"] = self.df.apply(
        self._calculate_opportunity_ceiling, axis=1
    )
    self.df["adjusted_points"] = self.df["opportunity_ceiling"]

    self.locks = []
    self.fades = []

  @staticmethod
  def _calculate_opportunity_ceiling(row) -> float:
    raw_pts = float(row.get("projected_points", 0.0))
    salary = int(row.get("salary", 0))
    pos = str(row.get("position", "")).upper()

    if pos in ["K", "DST"]:
      return raw_pts

    if salary >= 8500:
      opp_mult = 1.14
    elif salary >= 7000:
      opp_mult = 1.08
    elif salary >= 5000:
      opp_mult = 1.00
    elif salary >= 3500:
      opp_mult = 0.92
    else:
      opp_mult = 0.82

    return round(raw_pts * opp_mult, 2)

  def apply_user_guide(
      self, boosts: dict = None, locks: list = None, fades: list = None
  ):
    if boosts:
      for player, mult in boosts.items():
        self.df.loc[self.df["name"] == player, "user_boost"] = mult

    self.df["adjusted_points"] = (
        self.df["opportunity_ceiling"] * self.df["user_boost"]
    )

    if fades:
      self.df = self.df[~self.df["name"].isin(fades)].copy()

    self.locks = locks or []

  def optimize_showdown_multi(
      self,
      n_lineups: int = 10,
      max_overlap: int = 4,
      game_script: str = "balanced",
      contest_type: str = "limited_entry",
      favorite_team: str = None,
      underdog_team: str = None,
      allow_k_dst: bool = True,
      min_proj_floor: float = 4.0,
      **kwargs,
  ):
    clean_df = self.df.drop_duplicates(subset=["pulp_id"]).copy()
    if len(clean_df) < 6:
      raise ValueError("Draft group has fewer than 6 eligible active players.")

    if contest_type == "single_entry":
      max_exposure = 0.80
      randomness = 0.08
      salary_ceiling = 50000
    elif contest_type == "milly_maker":
      max_exposure = 0.45
      randomness = 0.18
      salary_ceiling = 50000
    else:
      max_exposure = 0.60
      randomness = 0.12
      salary_ceiling = 50000

    prob = pulp.LpProblem("DK_NFL_Showdown_Multi", pulp.LpMaximize)
    players = clean_df.to_dict("records")

    cpt = {
        p["pulp_id"]: pulp.LpVariable(f"cpt_{p['pulp_id']}", cat="Binary")
        for p in players
    }
    flex = {
        p["pulp_id"]: pulp.LpVariable(f"flex_{p['pulp_id']}", cat="Binary")
        for p in players
    }

    # 1. Salary Cap
    prob += (
        pulp.lpSum(
            (p["salary"] * 1.5 * cpt[p["pulp_id"]])
            + (p["salary"] * flex[p["pulp_id"]])
            for p in players
        )
        <= salary_ceiling
    )

    # 2. Roster Slot Sizes: 1 CPT, 5 FLEX
    prob += pulp.lpSum(cpt[p["pulp_id"]] for p in players) == 1
    prob += pulp.lpSum(flex[p["pulp_id"]] for p in players) == 5

    # 3. Exclusivity & Non-Factor Floor
    for p in players:
      prob += cpt[p["pulp_id"]] + flex[p["pulp_id"]] <= 1
      if p["adjusted_points"] < min_proj_floor:
        prob += cpt[p["pulp_id"]] + flex[p["pulp_id"]] == 0

    teams = list(
        set(p["team"] for p in players if p.get("team") and p["team"] != "UNK")
    )
    if len(teams) >= 2:
      prob += (
          pulp.lpSum(
              cpt[p["pulp_id"]] + flex[p["pulp_id"]]
              for p in players
              if p["team"] == teams[0]
          )
          >= 1
      )
      prob += (
          pulp.lpSum(
              cpt[p["pulp_id"]] + flex[p["pulp_id"]]
              for p in players
              if p["team"] == teams[1]
          )
          >= 1
      )

    for t in teams:
      team_qbs = [
          p for p in players if p["team"] == t and p["position"] == "QB"
      ]
      if len(team_qbs) > 1:
        prob += (
            pulp.lpSum(
                cpt[p["pulp_id"]] + flex[p["pulp_id"]] for p in team_qbs
            )
            <= 1
        )

      team_tes = [
          p for p in players if p["team"] == t and p["position"] == "TE"
      ]
      if len(team_tes) > 1:
        prob += (
            pulp.lpSum(
                cpt[p["pulp_id"]] + flex[p["pulp_id"]] for p in team_tes
            )
            <= 1
        )

    kickers = [p for p in players if p["position"] == "K"]
    if kickers:
      prob += (
          pulp.lpSum(cpt[p["pulp_id"]] + flex[p["pulp_id"]] for p in kickers)
          <= 1
      )

    for p in players:
      if p["position"] == "DST":
        opp_qbs = [
            q
            for q in players
            if q["position"] == "QB" and q["team"] != p["team"]
        ]
        for qb in opp_qbs:
          prob += (
              cpt[qb["pulp_id"]]
              + flex[qb["pulp_id"]]
              + cpt[p["pulp_id"]]
              + flex[p["pulp_id"]]
              <= 1
          )

        opp_kickers = [
            k for k in players if k["position"] == "K" and k["team"] != p["team"]
        ]
        for k in opp_kickers:
          prob += (
              cpt[k["pulp_id"]]
              + flex[k["pulp_id"]]
              + cpt[p["pulp_id"]]
              + flex[p["pulp_id"]]
              <= 1
          )

    fav = (
        favorite_team
        if favorite_team in teams
        else (teams[0] if teams else None)
    )
    und = (
        underdog_team
        if underdog_team in teams
        else (teams[1] if len(teams) > 1 else None)
    )

    if game_script == "shootout":
      qbs = [p for p in players if p["position"] == "QB"]
      if qbs:
        prob += (
            pulp.lpSum(cpt[p["pulp_id"]] + flex[p["pulp_id"]] for p in qbs)
            >= 1
        )
        prob += (
            pulp.lpSum(cpt[p["pulp_id"]] + flex[p["pulp_id"]] for p in qbs)
            <= 2
        )
      for p in [p for p in players if p["position"] in ["K", "DST"]]:
        prob += cpt[p["pulp_id"]] + flex[p["pulp_id"]] == 0
      for qb in qbs:
        team_wrs = [
            p
            for p in players
            if p["team"] == qb["team"] and p["position"] == "WR"
        ]
        if team_wrs:
          prob += pulp.lpSum(
              cpt[p["pulp_id"]] + flex[p["pulp_id"]] for p in team_wrs
          ) >= (cpt[qb["pulp_id"]] + flex[qb["pulp_id"]])
      for p in players:
        if p["salary"] < 6000:
          prob += cpt[p["pulp_id"]] == 0

    elif game_script == "5_1_onslaught" and fav and und:
      prob += (
          pulp.lpSum(
              cpt[p["pulp_id"]] + flex[p["pulp_id"]]
              for p in players
              if p["team"] == fav
          )
          == 5
      )
      prob += (
          pulp.lpSum(
              cpt[p["pulp_id"]] + flex[p["pulp_id"]]
              for p in players
              if p["team"] == und
          )
          == 1
      )
      prob += (
          pulp.lpSum(cpt[p["pulp_id"]] for p in players if p["team"] == fav)
          == 1
      )
      for p in players:
        if p["salary"] < 4000:
          prob += cpt[p["pulp_id"]] == 0

    elif game_script == "defensive_slugfest":
      qbs = [p for p in players if p["position"] == "QB"]
      k_dst = [p for p in players if p["position"] in ["K", "DST"]]
      prob += (
          pulp.lpSum(cpt[p["pulp_id"]] + flex[p["pulp_id"]] for p in qbs) <= 1
      )
      prob += (
          pulp.lpSum(cpt[p["pulp_id"]] + flex[p["pulp_id"]] for p in k_dst)
          >= 2
      )
      for p in players:
        if p["salary"] < 3000:
          prob += cpt[p["pulp_id"]] == 0

    elif game_script == "underdog_chase" and und:
      und_wrs = [
          p for p in players if p["team"] == und and p["position"] == "WR"
      ]
      und_qbs = [
          p for p in players if p["team"] == und and p["position"] == "QB"
      ]
      if und_wrs:
        prob += pulp.lpSum(cpt[p["pulp_id"]] for p in und_wrs) == 1
      if und_qbs:
        prob += pulp.lpSum(flex[p["pulp_id"]] for p in und_qbs) >= 1

    elif game_script == "naked_receiver":
      for p in players:
        if p["position"] in ["WR", "TE"]:
          team_qb = next(
              (
                  q
                  for q in players
                  if q["position"] == "QB" and q["team"] == p["team"]
              ),
              None,
          )
          if team_qb:
            prob += (
                cpt[team_qb["pulp_id"]] + flex[team_qb["pulp_id"]]
                <= 1 - cpt[p["pulp_id"]]
            )

    else:
      if len(teams) >= 2:
        prob += (
            pulp.lpSum(
                cpt[p["pulp_id"]] + flex[p["pulp_id"]]
                for p in players
                if p["team"] == teams[0]
            )
            >= 2
        )
        prob += (
            pulp.lpSum(
                cpt[p["pulp_id"]] + flex[p["pulp_id"]]
                for p in players
                if p["team"] == teams[1]
            )
            >= 2
        )
      for p in players:
        if p["salary"] < 5000:
          prob += cpt[p["pulp_id"]] == 0

    if not allow_k_dst:
      for p in players:
        if p["position"] in ["K", "DST"]:
          prob += cpt[p["pulp_id"]] + flex[p["pulp_id"]] == 0

    for p in players:
      if p["name"] in self.locks:
        prob += cpt[p["pulp_id"]] + flex[p["pulp_id"]] == 1

    lineups = []
    player_counts = {p["pulp_id"]: 0 for p in players}
    max_allowed_appearances = max(1, int(n_lineups * max_exposure))

    for line_num in range(1, n_lineups + 1):
      temp_constraints = []
      if line_num > 1:
        for p in players:
          if p["name"] not in self.locks and p["position"] != "QB":
            if player_counts[p["pulp_id"]] >= max_allowed_appearances:
              c = cpt[p["pulp_id"]] + flex[p["pulp_id"]] == 0
              prob += c
              temp_constraints.append(c)

      if randomness > 0 and line_num > 1:
        perturbed_pts = {}
        for p in players:
          vol = randomness * (1.25 if p["position"] == "WR" else 0.85)
          perturbed_pts[p["pulp_id"]] = max(
              0.5,
              random.gauss(
                  p["adjusted_points"], p["adjusted_points"] * vol
              ),
          )
      else:
        perturbed_pts = {p["pulp_id"]: p["adjusted_points"] for p in players}

      prob.objective = pulp.lpSum(
          (perturbed_pts[p["pulp_id"]] * 1.5 * cpt[p["pulp_id"]])
          + (perturbed_pts[p["pulp_id"]] * flex[p["pulp_id"]])
          for p in players
      )

      solver_status = prob.solve(
          pulp.PULP_CBC_CMD(msg=False, keepFiles=False)
      )
      if pulp.LpStatus[solver_status] != "Optimal":
        break

      current_cpt_id = None
      current_flex_ids = []
      roster_rows = []

      for p in players:
        if pulp.value(cpt[p["pulp_id"]]) == 1:
          current_cpt_id = p["pulp_id"]
          player_counts[p["pulp_id"]] += 1
          roster_rows.append({
              "Slot": "CPT",
              "Name": p["name"],
              "Position": p["position"],
              "Team": p["team"],
              "Salary": int(p["salary"] * 1.5),
              "Proj_FPTS": round(p["adjusted_points"] * 1.5, 2),
          })
        elif pulp.value(flex[p["pulp_id"]]) == 1:
          current_flex_ids.append(p["pulp_id"])
          player_counts[p["pulp_id"]] += 1
          roster_rows.append({
              "Slot": "FLEX",
              "Name": p["name"],
              "Position": p["position"],
              "Team": p["team"],
              "Salary": int(p["salary"]),
              "Proj_FPTS": round(p["adjusted_points"], 2),
          })

      ldf = pd.DataFrame(roster_rows)
      cpt_r = ldf[ldf["Slot"] == "CPT"]
      flex_r = ldf[ldf["Slot"] == "FLEX"].sort_values(
          by="Salary", ascending=False
      )
      sorted_ldf = pd.concat([cpt_r, flex_r], ignore_index=True)

      lineups.append({
          "lineup_num": line_num,
          "total_salary": int(sorted_ldf["Salary"].sum()),
          "total_proj": round(float(sorted_ldf["Proj_FPTS"].sum()), 2),
          "players": sorted_ldf.to_dict("records"),
      })

      prob += (
          cpt[current_cpt_id]
          + pulp.lpSum(flex[pid] for pid in current_flex_ids)
      ) <= max_overlap

    return lineups

  def optimize_classic_multi(
      self,
      n_lineups: int = 10,
      contest_type: str = "milly_maker",
      stack_style: str = "game_stack",
      primary_stack_team: str = "ANY",
      **kwargs,
  ):
    """DraftKings Classic Slate: 9 Roster Spots ($50,000 Cap)

    - Allows dual-TE dynamically when an elite TE is rostered (hurdle logic)
    - Safe Windows identifiers and solver configurations
    """
    valid_pos = ["QB", "RB", "WR", "TE", "DST"]
    df_classic = self.df[self.df["position"].isin(valid_pos)].copy()
    df_classic = df_classic.drop_duplicates(subset=["pulp_id"]).copy()

    if len(df_classic) < 9:
      raise ValueError(
          f"Player pool has only {len(df_classic)} players (minimum 9"
          " required)."
      )

    if contest_type == "single_entry":
      max_overlap = 8
      max_exposure = 0.85
      max_qb_exposure = 0.60
      randomness = 0.08
      salary_cap = 50000
    elif contest_type == "milly_maker":
      max_overlap = 5
      max_exposure = 0.40
      max_qb_exposure = 0.25
      randomness = 0.18
      salary_cap = 50000
    else:
      max_overlap = 6
      max_exposure = 0.55
      max_qb_exposure = 0.35
      randomness = 0.12
      salary_cap = 50000

    players = df_classic.to_dict("records")

    qbs = [p for p in players if p["position"] == "QB"]
    dsts = [p for p in players if p["position"] == "DST"]
    rbs = [p for p in players if p["position"] == "RB"]
    wrs = [p for p in players if p["position"] == "WR"]
    tes = [p for p in players if p["position"] == "TE"]

    if not qbs:
      raise ValueError("No active Starting Quarterbacks found in this slate.")
    if not dsts:
      raise ValueError("No Team Defenses (DST) found in this slate.")
    if len(rbs) < 2 or len(wrs) < 3 or len(tes) < 1:
      raise ValueError(
          "Insufficient skill players (RB/WR/TE) to fill a legal roster."
      )

    prob = pulp.LpProblem("DK_NFL_Classic_Multi", pulp.LpMaximize)
    x = {
        p["pulp_id"]: pulp.LpVariable(f"p_{p['pulp_id']}", cat="Binary")
        for p in players
    }

    # 1. Salary Cap & Total Roster Count
    prob += (
        pulp.lpSum(p["salary"] * x[p["pulp_id"]] for p in players) <= salary_cap
    )
    prob += pulp.lpSum(x[p["pulp_id"]] for p in players) == 9

    # 2. Positional Constraints
    prob += pulp.lpSum(x[p["pulp_id"]] for p in qbs) == 1
    prob += pulp.lpSum(x[p["pulp_id"]] for p in dsts) == 1

    # Optional Team Stack Lock
    if primary_stack_team and primary_stack_team != "ANY":
      team_qbs = [p["pulp_id"] for p in qbs if p["team"] == primary_stack_team]
      if team_qbs:
        prob += pulp.lpSum(x[qid] for qid in team_qbs) == 1

    prob += pulp.lpSum(x[p["pulp_id"]] for p in rbs) >= 2
    prob += pulp.lpSum(x[p["pulp_id"]] for p in rbs) <= 3
    prob += pulp.lpSum(x[p["pulp_id"]] for p in wrs) >= 3
    prob += pulp.lpSum(x[p["pulp_id"]] for p in wrs) <= 4
    prob += pulp.lpSum(x[p["pulp_id"]] for p in tes) >= 1

    # Dual-TE Hurdle Model: Allow 2nd TE into FLEX only if an elite TE is present
    if contest_type in ["milly_maker", "limited_entry"]:
      elite_tes = [
          p["pulp_id"]
          for p in tes
          if p["adjusted_points"] >= 14.0 or p["salary"] >= 6000
      ]
      if elite_tes:
        prob += pulp.lpSum(x[p["pulp_id"]] for p in tes) <= 1 + pulp.lpSum(
            x[pid] for pid in elite_tes
        )
        prob += pulp.lpSum(x[p["pulp_id"]] for p in tes) <= 2
      else:
        prob += pulp.lpSum(x[p["pulp_id"]] for p in tes) == 1
    else:
      prob += pulp.lpSum(x[p["pulp_id"]] for p in tes) <= 2

    all_skill_ids = [p["pulp_id"] for p in (rbs + wrs + tes)]
    prob += pulp.lpSum(x[pid] for pid in all_skill_ids) == 7

    # 3. DraftKings Team Diversity: Max 8 players from any single team
    teams = set(
        p["team"] for p in players if p.get("team") and p["team"] != "UNK"
    )
    for t in teams:
      team_pids = [p["pulp_id"] for p in players if p["team"] == t]
      if len(team_pids) > 8:
        prob += pulp.lpSum(x[pid] for pid in team_pids) <= 8

    # 4. User-Selected Stacking Architecture
    for qb in qbs:
      qb_team = qb["team"]
      opp_team = qb.get("opponent", "UNK")

      teammates = [
          p["pulp_id"]
          for p in players
          if p["team"] == qb_team and p["position"] in ["WR", "TE"]
      ]
      bring_backs = [
          p["pulp_id"]
          for p in players
          if opp_team != "UNK"
          and p["team"] == opp_team
          and p["position"] in ["WR", "RB", "TE"]
      ]

      if stack_style in ["game_stack", "double_game_stack"]:
        if teammates:
          if stack_style == "double_game_stack" and len(teammates) >= 2:
            prob += (
                pulp.lpSum(x[pid] for pid in teammates)
                >= 2 * x[qb["pulp_id"]]
            )
          else:
            prob += (
                pulp.lpSum(x[pid] for pid in teammates) >= x[qb["pulp_id"]]
            )
        if bring_backs:
          prob += (
              pulp.lpSum(x[pid] for pid in bring_backs) >= x[qb["pulp_id"]]
          )

      elif stack_style in ["double_stack", "single_stack"]:
        if teammates:
          if stack_style == "double_stack" and len(teammates) >= 2:
            prob += (
                pulp.lpSum(x[pid] for pid in teammates)
                >= 2 * x[qb["pulp_id"]]
            )
          else:
            prob += (
                pulp.lpSum(x[pid] for pid in teammates) >= x[qb["pulp_id"]]
            )

      elif stack_style == "naked_qb":
        if teammates:
          for pid in teammates:
            prob += x[pid] + x[qb["pulp_id"]] <= 1

    # 5. User Locks
    for p in players:
      if p["name"] in self.locks:
        prob += x[p["pulp_id"]] == 1

    lineups = []
    player_counts = {p["pulp_id"]: 0 for p in players}
    max_allowed_skill = max(1, int(n_lineups * max_exposure))
    max_allowed_qb = max(1, int(n_lineups * max_qb_exposure))

    for line_num in range(1, n_lineups + 1):
      if line_num > 1:
        for p in players:
          if p["name"] not in self.locks:
            if (
                p["position"] == "QB"
                and player_counts[p["pulp_id"]] >= max_allowed_qb
            ):
              prob += x[p["pulp_id"]] == 0
            elif (
                p["position"] != "QB"
                and player_counts[p["pulp_id"]] >= max_allowed_skill
            ):
              prob += x[p["pulp_id"]] == 0

      if randomness > 0 and line_num > 1:
        perturbed_pts = {}
        for p in players:
          vol = randomness * (1.20 if p["position"] == "WR" else 0.85)
          perturbed_pts[p["pulp_id"]] = max(
              0.5,
              random.gauss(
                  p["adjusted_points"], p["adjusted_points"] * vol
              ),
          )
      else:
        perturbed_pts = {
            p["pulp_id"]: p["adjusted_points"] for p in players
        }

      prob.objective = pulp.lpSum(
          perturbed_pts[p["pulp_id"]] * x[p["pulp_id"]] for p in players
      )

      solver_status = prob.solve(
          pulp.PULP_CBC_CMD(msg=False, keepFiles=False)
      )
      if pulp.LpStatus[solver_status] != "Optimal":
        break

      current_pids = [
          p["pulp_id"] for p in players if pulp.value(x[p["pulp_id"]]) == 1
      ]
      for pid in current_pids:
        player_counts[pid] += 1

      selected = [p for p in players if p["pulp_id"] in current_pids]
      selected = sorted(
          selected,
          key=lambda p: (
              p["position"] != "QB",
              p["position"] != "DST",
              -p["salary"],
          ),
      )

      counts = {"RB": 0, "WR": 0, "TE": 0}
      roster = []
      for p in selected:
        pos = p["position"]
        if pos == "QB":
          slot = "QB"
        elif pos == "DST":
          slot = "DST"
        elif pos == "RB" and counts["RB"] < 2:
          counts["RB"] += 1
          slot = f"RB{counts['RB']}"
        elif pos == "WR" and counts["WR"] < 3:
          counts["WR"] += 1
          slot = f"WR{counts['WR']}"
        elif pos == "TE" and counts["TE"] < 1:
          counts["TE"] += 1
          slot = "TE"
        else:
          slot = "FLEX"

        roster.append({
            "Slot": slot,
            "Name": p["name"],
            "Position": p["position"],
            "Team": p["team"],
            "Salary": int(p["salary"]),
            "Proj_FPTS": round(p["adjusted_points"], 2),
        })

      slot_order = {
          "QB": 0,
          "RB1": 1,
          "RB2": 2,
          "WR1": 3,
          "WR2": 4,
          "WR3": 5,
          "TE": 6,
          "FLEX": 7,
          "DST": 8,
      }
      roster = sorted(roster, key=lambda r: slot_order.get(r["Slot"], 99))

      lineups.append({
          "lineup_num": line_num,
          "total_salary": sum(r["Salary"] for r in roster),
          "total_proj": round(sum(r["Proj_FPTS"] for r in roster), 2),
          "players": roster,
      })

      prob += pulp.lpSum(x[pid] for pid in current_pids) <= max_overlap

    return lineups