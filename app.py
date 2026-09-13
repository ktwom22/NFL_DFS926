import traceback
from flask import Flask, render_template, request, jsonify
from optimizer import DraftKingsOptimizer
from dk_fetcher import get_all_upcoming_nfl_slates, get_slate_players, get_vegas_game_data

app = Flask(__name__)


@app.route("/")
def home():
    try:
        slates = get_all_upcoming_nfl_slates()
    except Exception as e:
        print(f"Error loading slates: {e}")
        slates = []
    return render_template("index.html", slates=slates)


@app.route("/api/vegas-info", methods=["POST"])
def vegas_info():
    data = request.get_json() or {}
    draft_group_id = data.get("draft_group_id")

    player_df = get_slate_players(draft_group_id)
    if player_df.empty:
        return jsonify({"status": "error", "message": "Slate player data unavailable."}), 404

    teams = sorted(list(set(t for t in player_df["team"].dropna() if t != "UNK")))

    if len(teams) >= 2:
        vegas = get_vegas_game_data(teams[0], teams[1])
    else:
        vegas = {
            "spread": 0.0, "over_under": 45.0,
            "favorite": "UNK", "underdog": "UNK",
            "fav_implied": 22.5, "und_implied": 22.5,
            "recommended_script": "balanced"
        }

    return jsonify({"status": "success", "vegas": vegas, "teams": teams})


@app.route("/api/optimize", methods=["POST"])
def optimize():
    data = request.get_json() or {}
    draft_group_id = data.get("draft_group_id")
    slate_type = data.get("slate_type", "showdown")
    game_script = data.get("game_script", "auto")
    stack_style = data.get("stack_style", "game_stack")
    primary_stack_team = data.get("primary_stack_team", "ANY")
    contest_type = data.get("contest_type", "milly_maker")
    use_boom_engine = data.get("use_boom_engine", True)
    favorite_team = data.get("favorite_team")
    underdog_team = data.get("underdog_team")
    n_lineups = int(data.get("n_lineups", 10))

    try:
        player_df = get_slate_players(draft_group_id)
        if player_df.empty:
            return jsonify({"status": "error", "message": "Player pool is empty."}), 404

        teams = list(set(player_df["team"].dropna()))
        teams = [t for t in teams if t != "UNK"]

        if game_script == "auto" and len(teams) >= 2:
            vegas = get_vegas_game_data(teams[0], teams[1])
            game_script = vegas["recommended_script"]
            favorite_team = vegas["favorite"]
            underdog_team = vegas["underdog"]

        opt = DraftKingsOptimizer(player_df, use_boom_engine=use_boom_engine)

        if slate_type == "showdown":
            lineups = opt.optimize_showdown_multi(
                n_lineups=n_lineups,
                max_overlap=4,
                game_script=game_script,
                contest_type=contest_type,
                favorite_team=favorite_team,
                underdog_team=underdog_team,
                allow_k_dst=data.get("allow_k_dst", True)
            )
            executed_script = f"{game_script} ({contest_type.replace('_', ' ')})"
        else:
            lineups = opt.optimize_classic_multi(
                n_lineups=n_lineups,
                contest_type=contest_type,
                stack_style=stack_style,
                primary_stack_team=primary_stack_team
            )
            team_tag = f" | {primary_stack_team}" if primary_stack_team and primary_stack_team != "ANY" else ""
            executed_script = f"{stack_style}{team_tag} ({contest_type.replace('_', ' ')})"

        if not lineups:
            return jsonify({
                "status": "error",
                "message": f"Could not find any viable lineups for '{executed_script}'."
            }), 400

        return jsonify({
            "status": "success",
            "executed_script": executed_script,
            "count": len(lineups),
            "lineups": lineups,
            "lineup": lineups[0]["players"],
            "total_salary": lineups[0]["total_salary"],
            "total_projected_fpts": lineups[0]["total_proj"]
        })

    except Exception as err:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(err)}), 400


import os

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port, debug=False)