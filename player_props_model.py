"""NFL player-props projection models: point projections for individual
player stat lines, one CatBoost regressor per market.

No player-prop sportsbook lines exist in any nflverse/sportsdataverse data
source (see README) - there is nothing to compute a market edge against.
`backtest` decides which markets are reliable enough to trust (does the model
beat just using the player's own trailing average?); `predict` stamps that
verdict onto every row so a market that hasn't cleared the bar is never
mistaken for one that has. Pass --line-file to compare projections against
whatever number you see at your own book - computed locally from the model's
projection and its backtested error spread, no odds API involved.

    uv run python player_props_model.py train --market pass_yards
    uv run python player_props_model.py backtest --market rush_yards --start-season 2018
    uv run python player_props_model.py predict --market receptions --season 2025 --week 15
    uv run python player_props_model.py predict --market rec_yards --line-file lines.csv

Markets: pass_yards, pass_tds, pass_interceptions, rush_yards, rush_tds,
receptions, rec_yards, rec_tds.
"""

from player_props_common import run_cli

if __name__ == "__main__":
    run_cli()
