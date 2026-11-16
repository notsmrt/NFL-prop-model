"""Shared engine for the NFL player-props projection models.

Same walk-forward discipline as the game-level models (`bet_common.py`): train
on seasons < S, predict season S, only then grade it. But there is no market
line to settle against here (see `player_props_features.py`'s docstring), so
these are point projections, not bet recommendations. Reliability per market
is decided empirically: `backtest` compares the model against a naive
baseline - the player's own trailing average, which is already one of the
model's own features - and only markets that clearly beat it are worth
trusting. `predict` stamps whatever the most recent `backtest` found onto its
output so a stale or never-run reliability check is obvious rather than silent.

A market's target is a count/yardage stat, not a win/cover binary, so this
uses `CatBoostRegressor` rather than the classifier + calibration pipeline the
moneyline/spread/total models share. CatBoost trees don't need feature
scaling, so (unlike `bet_common.py`) there's no `StandardScaler` here.

`player_props_model.py` is a thin CLI over this module, with `--market`
selecting one of `MARKETS` below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import norm

import nflreadpy as nfl

import nfl_features as nff
import player_props_features as ppf

CATBOOST_PARAMS = dict(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
)

# 80% interval half-width in std devs, for the low/high range shown in `predict`.
INTERVAL_Z = 1.28

# Empirical reliability gate: the model must beat the naive trailing-average
# baseline's MAE by at least this fraction to be marked RELIABLE.
MIN_IMPROVEMENT = 0.03

# A season needs at least this many prior-season residuals before `expanding_fold_std`
# will estimate its std - the residual spread is never fit on the same data it prices.
MIN_RESIDUAL_SAMPLE = 200

# Legs on players the reliability-flag pipeline already warns about aren't worth
# pricing: too little of a track record, buried on the depth chart, or ruled out.
DROP_FLAGS = ("LOW_SAMPLE", "BACKUP_ROLE", "OUT")

MARKETS = {
    "pass_yards": {
        "label": "passing yards", "positions": ["QB"],
        "target_col": "passing_yards", "participation_col": "attempts",
        "artifact": "artifacts/model_props_pass_yards.joblib",
    },
    "pass_tds": {
        "label": "passing TDs", "positions": ["QB"],
        "target_col": "passing_tds", "participation_col": "attempts",
        "artifact": "artifacts/model_props_pass_tds.joblib",
    },
    "pass_interceptions": {
        "label": "interceptions thrown", "positions": ["QB"],
        "target_col": "passing_interceptions", "participation_col": "attempts",
        "artifact": "artifacts/model_props_pass_int.joblib",
    },
    "rush_yards": {
        "label": "rushing yards", "positions": ["RB"],
        "target_col": "rushing_yards", "participation_col": "carries",
        "artifact": "artifacts/model_props_rush_yards.joblib",
    },
    "rush_tds": {
        "label": "rushing TDs", "positions": ["RB"],
        "target_col": "rushing_tds", "participation_col": "carries",
        "artifact": "artifacts/model_props_rush_tds.joblib",
    },
    "receptions": {
        "label": "receptions", "positions": ["RB", "WR", "TE"],
        "target_col": "receptions", "participation_col": "targets",
        "artifact": "artifacts/model_props_receptions.joblib",
    },
    "rec_yards": {
        "label": "receiving yards", "positions": ["RB", "WR", "TE"],
        "target_col": "receiving_yards", "participation_col": "targets",
        "artifact": "artifacts/model_props_rec_yards.joblib",
    },
    "rec_tds": {
        "label": "receiving TDs", "positions": ["RB", "WR", "TE"],
        "target_col": "receiving_tds", "participation_col": "targets",
        "artifact": "artifacts/model_props_rec_tds.joblib",
    },
}

FEATURE_EXTRA_COLS = ["games_played", "team_implied_total", "opp_implied_total", "is_home"]


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #
def _prepare_player_frame(seasons: list[int], market: str) -> pd.DataFrame:
    """Position-filtered player-week rows for every game the player actually
    played, with snap share and injury status attached. A played game with 0
    targets/attempts stays in as a 0 - it settles as a loss on the over, so
    both the label and the trailing average have to see it (see
    `player_props_features.load_played_frame`)."""
    spec = MARKETS[market]
    return ppf.load_played_frame(seasons, spec["positions"], spec["participation_col"])


def build_player_dataset(seasons: list[int], market: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (meta, features) for one market: one row per player-game that
    actually recorded the qualifying stat, `meta["y"]` the target."""
    spec = MARKETS[market]
    print(f"Loading player-props data for {seasons[0]}-{seasons[-1]} (market: {market}) ...")
    stats = _prepare_player_frame(seasons, market)

    lagged = ppf.lag_player_frame(stats, ppf.STAT_COLUMNS + ["offense_pct"])
    merged = stats.merge(lagged, on=["player_id", "season", "week"], how="left")

    schedules = nff.load_schedules(seasons)
    implied = ppf.build_team_implied_totals(schedules)
    merged = merged.merge(
        implied[["team", "season", "week", "team_implied_total", "opp_implied_total", "is_home"]],
        on=["team", "season", "week"], how="left")

    merged["reliability_flag"] = ppf.add_reliability_flags(merged)

    meta_cols = ["player_id", "player_display_name", "position", "team", "opponent_team",
                "season", "week", "game_id", "reliability_flag", spec["target_col"]]
    meta = merged[meta_cols].rename(columns={spec["target_col"]: "y"}).reset_index(drop=True)

    feature_cols = [c for c in merged.columns if c.startswith("lag_")] + FEATURE_EXTRA_COLS
    features = merged[feature_cols].reset_index(drop=True)
    print(f"  {len(meta)} qualifying player-games, {features.shape[1]} raw features")
    return meta, features


def build_inference_frame(market: str, season: int, week: int,
                          feature_columns) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per active, roster-eligible player at the target market's
    position(s) for every team playing in (season, week), with lag features
    computed from their qualifying games through week-1 (leak-free, same
    'inject a placeholder row so the roll emits a value' trick as
    `predict.py.build_inference_frame`)."""
    spec = MARKETS[market]
    hist_seasons = list(range(ppf.FIRST_SEASON, season + 1))

    stats = _prepare_player_frame(hist_seasons, market)
    before = stats[(stats["season"] < season) | (stats["week"] < week)]

    schedules = nff.load_schedules(hist_seasons)
    games = schedules[(schedules["season"] == season) & (schedules["week"] == week)]
    if games.empty:
        raise SystemExit(f"No games scheduled for {season} week {week}.")
    teams_playing = set(pd.unique(pd.concat([games["home_team"], games["away_team"]])))
    opp_map = dict(zip(games["home_team"], games["away_team"]))
    opp_map.update(dict(zip(games["away_team"], games["home_team"])))

    roster = ppf.load_rosters_weekly_safe(season)
    roster = roster[(roster["week"] == week) & roster["position"].isin(spec["positions"])
                    & (roster["status"] == "ACT") & roster["team"].isin(teams_playing)]
    roster = roster.rename(columns={"gsis_id": "player_id"}).drop_duplicates("player_id")
    if roster.empty:
        raise SystemExit(f"No active {'/'.join(spec['positions'])} on a {season} week {week} roster.")

    placeholder = pd.DataFrame({
        "player_id": roster["player_id"].to_numpy(),
        "player_display_name": roster["full_name"].to_numpy(),
        "position": roster["position"].to_numpy(),
        "team": roster["team"].to_numpy(),
    })
    placeholder["opponent_team"] = placeholder["team"].map(opp_map)
    placeholder["season"] = season
    placeholder["week"] = week

    combined = pd.concat([before, placeholder], ignore_index=True)
    lagged = ppf.lag_player_frame(combined, ppf.STAT_COLUMNS + ["offense_pct"])
    target_lag = lagged[(lagged["season"] == season) & (lagged["week"] == week)]

    out = placeholder.merge(target_lag.drop(columns=["season", "week"]),
                            on="player_id", how="left")

    implied = ppf.build_team_implied_totals(schedules)
    out = out.merge(
        implied[["team", "season", "week", "team_implied_total", "opp_implied_total", "is_home"]],
        on=["team", "season", "week"], how="left")

    injuries_now = ppf.load_injuries([season])
    out = ppf.attach_injury_status(out, injuries_now)
    out["reliability_flag"] = ppf.add_reliability_flags(out)

    features = out.reindex(columns=feature_columns)
    return out, features


# --------------------------------------------------------------------------- #
# Walk-forward backtest (cached - the slow part)
# --------------------------------------------------------------------------- #
def walk_forward(market: str, start_season: int, end_season: int,
                 refresh: bool = False) -> pd.DataFrame:
    spec = MARKETS[market]
    cache = Path("artifacts") / f"wf_props_{market}_{start_season}_{end_season}.parquet"
    if cache.exists() and not refresh:
        print(f"Using cached predictions: {cache}")
        return pd.read_parquet(cache)

    seasons = list(range(ppf.FIRST_SEASON, end_season + 1))
    meta, features = build_player_dataset(seasons, market)
    baseline_col = f"lag_{spec['target_col']}"

    out = []
    for season in range(start_season, end_season + 1):
        tr = (meta["season"] < season).to_numpy()
        te = (meta["season"] == season).to_numpy()
        if te.sum() == 0:
            continue
        means = features.loc[tr].mean(numeric_only=True).fillna(0.0)
        x_tr = features.loc[tr].fillna(means)
        x_te = features.loc[te].fillna(means)
        y_tr = meta.loc[tr, "y"].to_numpy(dtype=float)

        model = CatBoostRegressor(**CATBOOST_PARAMS)
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)

        d = meta.loc[te, ["player_id", "player_display_name", "position", "team",
                          "opponent_team", "season", "week", "reliability_flag", "y"]].copy()
        d["pred"] = pred
        d["baseline"] = (x_te[baseline_col].to_numpy() if baseline_col in x_te.columns
                         else np.nan)
        out.append(d)
        print(f"  predicted {season}: {int(te.sum())} player-games")

    preds = pd.concat(out, ignore_index=True)
    cache.parent.mkdir(exist_ok=True)
    preds.to_parquet(cache)
    print(f"Cached predictions -> {cache}")
    return preds


def expanding_fold_std(market: str, start_season: int, end_season: int,
                       refresh: bool = False,
                       min_sample: int = MIN_RESIDUAL_SAMPLE) -> pd.DataFrame:
    """`walk_forward` predictions for `market`, filtered to reliability-eligible rows
    (drops `DROP_FLAGS`), with a leak-free expanding-window residual std attached per
    season (`fold_std`): each season's std is estimated from only strictly-prior
    seasons' residuals, gated at `min_sample` - the same spirit as `train()` estimating
    spread from a single holdout season, just an expanding window instead of one fixed
    split. Seasons without enough prior history are dropped from the result.

    Shared by `parlay_optimizer.py` (grading backtest prop legs) and
    `leg_correlation.py` (standardizing prop outcomes to z-scores for correlation
    estimation) so both use one definition of "how much prior history is enough."
    """
    wf = walk_forward(market, start_season, end_season, refresh)
    wf = wf.dropna(subset=["baseline", "y"]).reset_index(drop=True)
    flagged = wf["reliability_flag"].fillna("").str.contains("|".join(DROP_FLAGS), regex=True)
    wf = wf[~flagged].reset_index(drop=True)

    resid = (wf["pred"] - wf["y"]).to_numpy()
    seasons = wf["season"].to_numpy()
    fold_std = np.full(len(wf), np.nan)
    for s in sorted(wf["season"].unique()):
        prior = resid[seasons < s]
        if len(prior) >= min_sample:
            fold_std[seasons == s] = float(prior.std())
    return wf.assign(fold_std=fold_std).dropna(subset=["fold_std"]).reset_index(drop=True)


def _reliability(g: pd.DataFrame) -> dict:
    valid = g.dropna(subset=["baseline"])
    model_mae = (valid["pred"] - valid["y"]).abs().mean()
    baseline_mae = (valid["baseline"] - valid["y"]).abs().mean()
    model_rmse = float(np.sqrt(((valid["pred"] - valid["y"]) ** 2).mean()))
    improvement = (baseline_mae - model_mae) / baseline_mae if baseline_mae else np.nan
    return {
        "n": len(valid), "model_mae": model_mae, "baseline_mae": baseline_mae,
        "model_rmse": model_rmse, "improvement_pct": improvement,
        "reliable": bool(pd.notna(improvement) and improvement >= MIN_IMPROVEMENT),
    }


def print_backtest(market: str, preds: pd.DataFrame) -> dict:
    spec = MARKETS[market]
    valid = preds.dropna(subset=["baseline"])
    print(f"\n{len(preds)} player-games predicted, {len(valid)} with a naive baseline to "
          f"compare against  ({spec['label']})\n")

    print("  season     n   model MAE   baseline MAE   improvement")
    for season, g in valid.groupby("season"):
        m = _reliability(g)
        print(f"  {season}  {m['n']:5d}     {m['model_mae']:8.2f}    {m['baseline_mae']:10.2f}   "
              f"{m['improvement_pct']:+8.1%}")

    overall = _reliability(valid)
    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    print(f"{spec['label']}: model MAE {overall['model_mae']:.2f} vs naive-baseline "
          f"(player's own trailing average) MAE {overall['baseline_mae']:.2f}  "
          f"({overall['improvement_pct']:+.1%}, n={overall['n']})")
    if overall["reliable"]:
        print(f"  -> RELIABLE: beats the naive baseline by >= {MIN_IMPROVEMENT:.0%}. "
              "Safe to use `predict` for this market.")
    else:
        print(f"  -> NOT RELIABLE: doesn't beat just using the player's own recent average "
              f"by {MIN_IMPROVEMENT:.0%}. `predict` will still run but flags this market as "
              "unproven - treat its numbers as a coin flip vs. your own gut, not a projection.")
    return overall


def _update_bundle_reliability(market: str, overall: dict) -> None:
    """Stamp the just-computed backtest verdict onto the shipped bundle (if any)
    so `predict` reflects the latest reliability check instead of a stale one."""
    path = Path(MARKETS[market]["artifact"])
    if not path.exists():
        return
    bundle = joblib.load(path)
    bundle["reliability"] = {
        "reliable": overall["reliable"],
        "improvement_pct": overall["improvement_pct"],
        "model_mae": overall["model_mae"],
        "baseline_mae": overall["baseline_mae"],
        "n": overall["n"],
    }
    joblib.dump(bundle, path)
    print(f"Stamped reliability verdict onto {path}")


# --------------------------------------------------------------------------- #
# Train + ship a single bundle
# --------------------------------------------------------------------------- #
def train(market: str) -> None:
    spec = MARKETS[market]
    artifact_path = Path(spec["artifact"])
    train_through = nfl.get_current_season()
    seasons = list(range(ppf.FIRST_SEASON, train_through + 1))

    print(f"=== training props/{market} model -> {artifact_path} ===")
    meta, features = build_player_dataset(seasons, market)
    feature_columns = features.columns
    y = meta["y"].to_numpy(dtype=float)

    holdout = train_through
    is_ho = (meta["season"] == holdout).to_numpy()
    if is_ho.sum() == 0:
        raise RuntimeError(f"No player-games for holdout season {holdout}.")

    means = features.loc[~is_ho].mean(numeric_only=True).fillna(0.0)
    x_tr = features.loc[~is_ho].fillna(means)
    x_ho = features.loc[is_ho].fillna(means)
    y_tr, y_ho = y[~is_ho], y[is_ho]

    print(f"\nEvaluating: train {seasons[0]}-{holdout - 1} ({len(y_tr)}), "
          f"holdout {holdout} ({len(y_ho)})")
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(x_tr, y_tr)
    pred_ho = model.predict(x_ho)

    baseline_col = f"lag_{spec['target_col']}"
    baseline_ho = (x_ho[baseline_col].to_numpy() if baseline_col in x_ho.columns
                  else np.full_like(y_ho, np.nan))
    mae = float(np.abs(pred_ho - y_ho).mean())
    rmse = float(np.sqrt(((pred_ho - y_ho) ** 2).mean()))
    baseline_mae = float(np.nanmean(np.abs(baseline_ho - y_ho)))
    print(f"  holdout MAE  : {mae:.2f}  ({spec['label']})")
    print(f"  holdout RMSE : {rmse:.2f}")
    print(f"  naive baseline (player's own trailing average) MAE: {baseline_mae:.2f}")
    print("  -> run `backtest` for the full walk-forward reliability verdict before trusting this.")

    print(f"\nRefitting on {seasons[0]}-{train_through} ...")
    final_means = features.mean(numeric_only=True).fillna(0.0)
    x_all = features.fillna(final_means)
    final_model = CatBoostRegressor(**CATBOOST_PARAMS)
    final_model.fit(x_all, y)
    residual_std = float((pred_ho - y_ho).std())

    artifact_path.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": final_model,
        "feature_columns": list(feature_columns),
        "impute_means": final_means,
        "market": market,
        "target_col": spec["target_col"],
        "positions": spec["positions"],
        "trained_through_season": train_through,
        "first_season": ppf.FIRST_SEASON,
        "residual_std": residual_std,
        "holdout_mae": mae,
        "holdout_baseline_mae": baseline_mae,
        "reliability": None,  # populated by `backtest`
    }, artifact_path)
    print(f"Saved {artifact_path}")


# --------------------------------------------------------------------------- #
# Predict one upcoming week
# --------------------------------------------------------------------------- #
def predict_week(market: str, season: int, week: int, bundle: dict) -> pd.DataFrame:
    spec = MARKETS[market]
    out, features = build_inference_frame(market, season, week, bundle["feature_columns"])
    X = features.fillna(bundle["impute_means"]).fillna(0.0)
    pred = bundle["model"].predict(X)
    std = bundle["residual_std"]

    # The player's own trailing average - already a feature (`lag_<target_col>`) and
    # the same baseline the reliability gate compares against. Doubles as a synthetic
    # "line" for markets with no real sportsbook line (see `player_props_common.py`
    # module docstring / parlay_optimizer.py).
    baseline_col = f"lag_{spec['target_col']}"
    baseline_vals = (X[baseline_col].to_numpy() if baseline_col in X.columns
                     else np.full(len(X), np.nan))

    reliability = bundle.get("reliability")
    if reliability is None:
        tier = "UNKNOWN (run `backtest` first)"
    elif reliability["reliable"]:
        tier = f"RELIABLE (+{reliability['improvement_pct']:.0%} vs baseline)"
    else:
        tier = f"NOT RELIABLE ({reliability['improvement_pct']:+.0%} vs baseline)"

    out = out.reset_index(drop=True)
    rows = []
    for i, r in out.iterrows():
        p = max(0.0, float(pred[i]))
        rows.append({
            "player": r["player_display_name"], "position": r["position"], "team": r["team"],
            "opponent": r["opponent_team"], "season": season, "week": week,
            "market": spec["label"], "projected": round(p, 1),
            "low": round(max(0.0, p - INTERVAL_Z * std), 1),
            "high": round(p + INTERVAL_Z * std, 1),
            "baseline": (round(float(baseline_vals[i]), 1)
                        if pd.notna(baseline_vals[i]) else np.nan),
            "reliability_tier": tier,
            "flags": r["reliability_flag"],
        })
    table = pd.DataFrame(rows).sort_values("projected", ascending=False).reset_index(drop=True)
    return table


def add_line_comparison(table: pd.DataFrame, bundle: dict, lines: pd.DataFrame) -> pd.DataFrame:
    """Merge a user-supplied `player,line` frame onto the projection table and
    compute the model's own implied P(over) from its predicted distribution -
    no odds feed involved, just the model comparing its point estimate + the
    backtested residual spread against whatever number the user typed in."""
    std = bundle["residual_std"]
    key = table["player"].str.strip().str.lower()
    lines = lines.copy()
    lines["_key"] = lines["player"].str.strip().str.lower()
    merged = table.assign(_key=key).merge(
        lines[["_key", "line"]], on="_key", how="left").drop(columns=["_key"])
    prob_over = 1 - norm.cdf((merged["line"] - merged["projected"]) / std)
    merged["prob_over_line"] = np.where(merged["line"].notna(), prob_over.round(3), np.nan)
    return merged


def header(market: str, season: int, week: int, bundle: dict) -> str:
    spec = MARKETS[market]
    return (f"{season} week {week}  ({spec['label']} model, "
            f"trained {bundle['first_season']}-{bundle['trained_through_season']})")


def load_bundle(market: str, path: str | None) -> dict:
    p = Path(path) if path else Path(MARKETS[market]["artifact"])
    if not p.exists():
        raise SystemExit(f"{p} not found - run `uv run python player_props_model.py "
                         f"train --market {market}` first.")
    return joblib.load(p)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_train(args) -> None:
    train(args.market)


def _cmd_backtest(args) -> None:
    end_season = args.end_season or nfl.get_current_season()
    print(f"props/{args.market} backtest: folds {args.start_season}-{end_season}\n")
    preds = walk_forward(args.market, args.start_season, end_season, args.refresh)
    overall = print_backtest(args.market, preds)
    _update_bundle_reliability(args.market, overall)
    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        preds.to_csv(out / f"props_{args.market}_backtest.csv", index=False)
        print(f"\nWrote {out / f'props_{args.market}_backtest.csv'}")


def _cmd_predict(args) -> None:
    bundle = load_bundle(args.market, args.model)
    season, week = args.season, args.week
    if season is None or week is None:
        sched_season = season or nfl.get_current_season()
        sched = nff.load_schedules([sched_season])
        unplayed = sched[sched["result"].isna()]
        if season is None:
            season = sched_season if not unplayed.empty else sched_season + 1
            unplayed = nff.load_schedules([season])
            unplayed = unplayed[unplayed["result"].isna()]
        if week is None:
            if unplayed.empty:
                raise SystemExit(f"Every {season} game already has a result - nothing to predict.")
            week = int(unplayed["week"].min())

    print(f"Predicting {header(args.market, season, week, bundle)}\n")
    table = predict_week(args.market, season, week, bundle)

    if args.line_file:
        lines = pd.read_csv(args.line_file)
        table = add_line_comparison(table, bundle, lines)

    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(table.to_string(index=False))

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"props_{args.market}_{season}_wk{week:02d}.csv"
        table.to_csv(path, index=False)
        print(f"\nWrote {path}")


def run_cli() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train + ship one market's model")
    t.add_argument("--market", required=True, choices=list(MARKETS))
    t.set_defaults(func=_cmd_train)

    b = sub.add_parser("backtest", help="walk-forward reliability backtest")
    b.add_argument("--market", required=True, choices=list(MARKETS))
    b.add_argument("--start-season", type=int, default=2015)
    b.add_argument("--end-season", type=int, default=None)
    b.add_argument("--refresh", action="store_true", help="ignore the cached predictions")
    b.add_argument("--output", type=str, default=None, help="dir for the backtest CSV")
    b.set_defaults(func=_cmd_backtest)

    p = sub.add_parser("predict", help="project one upcoming week")
    p.add_argument("--market", required=True, choices=list(MARKETS))
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--week", type=int, default=None)
    p.add_argument("--model", type=str, default=None, help="bundle path (default the market's artifact)")
    p.add_argument("--line-file", type=str, default=None,
                   help="CSV with columns player,line - adds prob_over_line, computed "
                        "locally from the model's own projection + residual spread")
    p.add_argument("--out-dir", type=str, default=None)
    p.set_defaults(func=_cmd_predict)

    args = ap.parse_args()
    args.func(args)
