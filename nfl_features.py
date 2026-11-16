"""Shared feature engineering for the NFL game predictor.

Both training (`train.py`) and inference (`predict.py`) build their feature
matrix through these functions so the two pipelines cannot drift apart.

Data source: nflverse, via the `nflreadpy` package. `load_schedules` carries the
pre-game context we use as-is (rest days, betting lines) plus `result`; team
box-score stats come from `load_team_stats(summary_level="week")` and are turned
into *lagged* rolling averages so no post-game information leaks into a game's
own feature row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import nflreadpy as nfl

# Columns taken straight off the schedule (known before kickoff).
SCHEDULE_FEATURES = [
    "game_id", "week", "season", "home_team", "away_team",
    "home_rest", "away_rest", "spread_line", "total_line", "result",
]

# Rolling window (in games) for the lagged team-stat averages.
LAG_WINDOW = 4

# Earliest season with usable nflverse team stats.
FIRST_SEASON = 2000


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    """Return the nflverse schedule for the given seasons as a pandas frame."""
    return nfl.load_schedules(seasons=list(seasons)).to_pandas()


def load_team_stats(seasons: list[int]) -> pd.DataFrame:
    """Weekly team stats for the given seasons.

    Seasons with no games played yet (e.g. an upcoming season) 404 on the
    nflverse release and are skipped rather than raising.
    """
    frames = []
    for year in seasons:
        try:
            frames.append(
                nfl.load_team_stats(seasons=year, summary_level="week").to_pandas()
            )
        except Exception as exc:  # noqa: BLE001 - nflreadpy raises bare ConnectionError on 404
            if "404" in str(exc):
                print(f"  (no team stats published for {year} yet - skipping)")
                continue
            raise
    if not frames:
        raise RuntimeError("No team stats could be loaded for any requested season.")
    return pd.concat(frames, ignore_index=True)


def lag_team_frame(df: pd.DataFrame, window: int = LAG_WINDOW) -> pd.DataFrame:
    """Per team-season rolling mean of a frame's *previous* rows.

    `df` must have `team`, `season`, `week` columns plus numeric stat columns.
    `shift()` drops the current game before the rolling average, so row N holds
    the mean of that team-season's games N-window .. N-1 only. This is the
    canonical leak-free lag; every rolling feature builder goes through it.
    """
    df = df.sort_values(by=["team", "season", "week"])
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    lagged = df.groupby(["team", "season"])[numeric_cols].transform(
        lambda x: x.shift().rolling(window=window, min_periods=1).mean()
    )
    for k in ("team", "week", "season"):
        lagged[k] = df[k].values
    return lagged


def build_lagged_team_stats(team_stats: pd.DataFrame, window: int = LAG_WINDOW) -> pd.DataFrame:
    """Lagged rolling box-score stats (the base model's team-form block)."""
    return lag_team_frame(team_stats, window)


# --------------------------------------------------------------------------- #
# Optional feature builders (variants). Each returns a frame keyed by `game_id`
# with its columns already home_/away_ prefixed where a team split applies. The
# base model never calls any of these.
# --------------------------------------------------------------------------- #
def load_pbp(seasons: list[int]) -> pd.DataFrame:
    """Play-by-play for the given seasons; seasons not yet published are skipped."""
    frames = []
    for year in seasons:
        try:
            frames.append(nfl.load_pbp(seasons=year).to_pandas())
        except Exception as exc:  # noqa: BLE001
            if "404" in str(exc):
                print(f"  (no pbp published for {year} yet - skipping)")
                continue
            raise
    if not frames:
        raise RuntimeError("No pbp could be loaded for any requested season.")
    return pd.concat(frames, ignore_index=True)


def _inject_target_rows(team_frame: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Add NaN (team, season, week) rows for games in `schedules` not already in
    `team_frame`, so the subsequent lag emits a value for an upcoming week.

    Harmless in training (every played game with data is already present); for
    inference this is what lets a future week pick up each team's prior form.
    """
    want = pd.concat([
        schedules[["home_team", "season", "week"]].rename(columns={"home_team": "team"}),
        schedules[["away_team", "season", "week"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True).drop_duplicates()
    combined = pd.concat([team_frame, want], ignore_index=True)
    return combined.drop_duplicates(["team", "season", "week"], keep="first")


def _merge_team_lagged(schedules: pd.DataFrame, lagged: pd.DataFrame) -> pd.DataFrame:
    """Attach a lagged team frame to games as home_*/away_* columns, keyed by game_id."""
    out = schedules[["game_id", "home_team", "away_team", "week", "season"]].copy()
    val_cols = [c for c in lagged.columns if c not in ("team", "week", "season")]
    for side in ("home", "away"):
        side_l = lagged.rename(columns={"team": f"{side}_team",
                                        **{c: f"{side}_{c}" for c in val_cols}})
        out = out.merge(side_l, on=[f"{side}_team", "week", "season"], how="left")
    return out.drop(columns=["home_team", "away_team", "week", "season"])


def build_pbp_features(schedules: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Lagged play-by-play efficiency: EPA/play, success rate, splits, explosive
    rate - for a team's offense and (via defteam) the defense it allowed."""
    p = pbp[((pbp["pass"] == 1) | (pbp["rush"] == 1)) & pbp["epa"].notna()].copy()
    p["early"] = p["down"].isin([1, 2])
    p["explosive"] = (p["epa"] > 0.75).astype(float)

    def side_agg(unit_col: str, prefix: str) -> pd.DataFrame:
        g = p.groupby([unit_col, "season", "week"])
        out = g.agg(
            **{f"{prefix}epa": ("epa", "mean"),
               f"{prefix}success": ("success", "mean"),
               f"{prefix}explosive": ("explosive", "mean"),
               f"{prefix}pass_rate": ("pass", "mean"),
               f"{prefix}air_yards": ("air_yards", "mean")}
        )
        out[f"{prefix}pass_epa"] = g.apply(
            lambda d: d.loc[d["pass"] == 1, "epa"].mean(), include_groups=False)
        out[f"{prefix}rush_epa"] = g.apply(
            lambda d: d.loc[d["rush"] == 1, "epa"].mean(), include_groups=False)
        out[f"{prefix}early_epa"] = g.apply(
            lambda d: d.loc[d["early"], "epa"].mean(), include_groups=False)
        return out.reset_index().rename(columns={unit_col: "team"})

    off = side_agg("posteam", "pbp_off_")
    deff = side_agg("defteam", "pbp_def_")
    team_game = off.merge(deff, on=["team", "season", "week"], how="outer")
    team_game = _inject_target_rows(team_game, schedules)
    return _merge_team_lagged(schedules, lag_team_frame(team_game))


def build_qb_features(schedules: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Starting-QB form: each listed starter's rolling qb_epa / cpoe (over their
    own last 8 games, leak-free), a career-sample count (proxy for backup /
    inexperience), and a flag for a QB change from the team's previous game."""
    db = pbp[(pbp["qb_dropback"] == 1) if "qb_dropback" in pbp.columns
             else ((pbp["pass"] == 1) | (pbp["sack"] == 1))]
    db = db[db["passer_player_id"].notna() & db["qb_epa"].notna()]
    qg = db.groupby(["passer_player_id", "season", "week"]).agg(
        e=("qb_epa", "mean"), c=("cpoe", "mean"),
    ).reset_index()
    # inject each game's listed starter for the target week so the roll emits a value
    listed = pd.concat([
        schedules[["home_qb_id", "season", "week"]].rename(columns={"home_qb_id": "passer_player_id"}),
        schedules[["away_qb_id", "season", "week"]].rename(columns={"away_qb_id": "passer_player_id"}),
    ], ignore_index=True).dropna(subset=["passer_player_id"])
    qg = pd.concat([qg, listed], ignore_index=True).drop_duplicates(
        ["passer_player_id", "season", "week"], keep="first"
    ).sort_values(["passer_player_id", "season", "week"])
    roll = qg.groupby("passer_player_id")[["e", "c"]].transform(
        lambda x: x.shift().rolling(8, min_periods=1).mean())
    qb_form = qg[["passer_player_id", "season", "week"]].assign(
        qb_epa=roll["e"], qb_cpoe=roll["c"],
        qb_games=qg.groupby("passer_player_id").cumcount().values,
    )

    # each team's starter in its previous game, for the "QB changed" flag
    starts = pd.concat([
        schedules[["season", "week", "home_team", "home_qb_id"]].rename(
            columns={"home_team": "team", "home_qb_id": "qb_id"}),
        schedules[["season", "week", "away_team", "away_qb_id"]].rename(
            columns={"away_team": "team", "away_qb_id": "qb_id"}),
    ], ignore_index=True).dropna(subset=["qb_id"]).sort_values(["team", "season", "week"])
    starts["prev_qb_id"] = starts.groupby("team")["qb_id"].shift()

    out = schedules[["game_id", "season", "week", "home_team", "away_team",
                     "home_qb_id", "away_qb_id"]].copy()
    for side in ("home", "away"):
        s = starts.rename(columns={"team": f"{side}_team", "qb_id": f"{side}_qb_id"})
        out = out.merge(s[[f"{side}_team", "season", "week", f"{side}_qb_id", "prev_qb_id"]],
                        on=[f"{side}_team", "season", "week", f"{side}_qb_id"], how="left")
        f = qb_form.rename(columns={"passer_player_id": f"{side}_qb_id",
                                    "qb_epa": f"{side}_qb_epa", "qb_cpoe": f"{side}_qb_cpoe",
                                    "qb_games": f"{side}_qb_games"})
        out = out.merge(f, on=[f"{side}_qb_id", "season", "week"], how="left")
        out[f"{side}_qb_changed"] = (
            out["prev_qb_id"].notna() & (out[f"{side}_qb_id"] != out["prev_qb_id"])
        ).astype(float)
        out = out.drop(columns=["prev_qb_id"])
    return out[["game_id"] + [c for c in out.columns
                              if c.startswith(("home_qb_epa", "home_qb_cpoe", "home_qb_games",
                                               "home_qb_changed", "away_qb_epa", "away_qb_cpoe",
                                               "away_qb_games", "away_qb_changed"))]]


# Rough home time zone per franchise (US mainland offsets; body-clock effect).
TEAM_TZ = {
    "SEA": -8, "SF": -8, "LA": -8, "LAC": -8, "LV": -8, "ARI": -7, "DEN": -7,
    "DAL": -6, "HOU": -6, "KC": -6, "MIN": -6, "GB": -6, "CHI": -6, "NO": -6, "TEN": -6,
    "DET": -5, "IND": -5, "CIN": -5, "CLE": -5, "PIT": -5, "BUF": -5, "NYG": -5,
    "NYJ": -5, "NE": -5, "PHI": -5, "WAS": -5, "BAL": -5, "MIA": -5, "JAX": -5,
    "ATL": -5, "CAR": -5, "TB": -5,
    # historical abbreviations
    "OAK": -8, "SD": -8, "STL": -6,
}


def build_context_features(schedules: pd.DataFrame) -> pd.DataFrame:
    """Cheap situational context, all known before kickoff, keyed by game_id."""
    s = schedules
    roof = s["roof"].fillna("").str.lower()
    wd = s["weekday"].fillna("")
    gt = s["gametime"].fillna("13:00").str.slice(0, 2).replace("", "13").astype(int)
    out = pd.DataFrame({
        "game_id": s["game_id"],
        "ctx_div_game": s.get("div_game", 0).fillna(0).astype(float),
        "ctx_is_dome": roof.isin(["dome", "closed"]).astype(float),
        "ctx_is_primetime": (wd.isin(["Thursday", "Monday"]) | (gt >= 19)).astype(float),
        "ctx_home_off_bye": (s["home_rest"] >= 13).astype(float),
        "ctx_away_off_bye": (s["away_rest"] >= 13).astype(float),
        "ctx_home_short_week": (s["home_rest"] <= 4).astype(float),
        "ctx_away_short_week": (s["away_rest"] <= 4).astype(float),
        "ctx_rest_diff": (s["home_rest"] - s["away_rest"]).astype(float),
        "ctx_temp": np.where(roof.isin(["dome", "closed"]), 68.0, s["temp"].fillna(60.0)),
        "ctx_wind": np.where(roof.isin(["dome", "closed"]), 0.0, s["wind"].fillna(0.0)),
        "ctx_away_tz_shift": [
            abs(TEAM_TZ.get(h, -6) - TEAM_TZ.get(a, -6))
            for h, a in zip(s["home_team"], s["away_team"])
        ],
    })
    return out


BUILDERS = {
    "pbp": ("pbp", build_pbp_features),
    "qb": ("pbp", build_qb_features),
    "context": (None, build_context_features),
}


def build_extra_features(names: list[str], schedules: pd.DataFrame,
                         seasons: list[int]) -> list[pd.DataFrame]:
    """Run the named builders and return their game_id-keyed frames."""
    if not names:
        return []
    pbp = load_pbp(seasons) if any(BUILDERS[n][0] == "pbp" for n in names) else None
    frames = []
    for n in names:
        need, fn = BUILDERS[n]
        frames.append(fn(schedules, pbp) if need == "pbp" else fn(schedules))
    return frames


def assemble_matrix(
    schedules: pd.DataFrame,
    lagged_team_stats: pd.DataFrame,
    feature_columns: pd.Index | list[str] | None = None,
    extra_features: list[pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join schedule context + home/away lagged stats into a numeric matrix.

    Returns `(meta, features)` where `meta` keeps identifying columns
    (`game_id`, teams, `result`) and `features` is numeric-only.

    When `feature_columns` is given (inference), `features` is reindexed to
    exactly those columns so the model always sees the training layout.

    `extra_features` is an optional list of game_id-keyed frames (from the
    variant builders) merged in after the team-form block. The base model
    passes nothing, so its behaviour is unchanged.
    """
    cols = [c for c in SCHEDULE_FEATURES if c in schedules.columns]
    pre_game = schedules[cols].copy()

    home = lagged_team_stats.add_prefix("home_")
    away = lagged_team_stats.add_prefix("away_")

    merged = pd.merge(
        pre_game, home,
        left_on=["home_team", "week", "season"],
        right_on=["home_team", "home_week", "home_season"],
        how="left",
    )
    merged = pd.merge(
        merged, away,
        left_on=["away_team", "week", "season"],
        right_on=["away_team", "away_week", "away_season"],
        how="left",
    )
    merged = merged.loc[:, ~merged.columns.duplicated()]

    for extra in (extra_features or []):
        merged = merged.merge(extra, on="game_id", how="left", suffixes=("", "_dup"))
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_dup")])

    meta_cols = ["game_id", "season", "week", "home_team", "away_team", "result"]
    meta = merged[[c for c in meta_cols if c in merged.columns]].copy()
    if "result" in merged.columns:
        played = merged["result"].notna()
        meta["home_win"] = pd.array(
            np.where(played, merged["result"] > 0, None), dtype="boolean"
        ).astype("Int64")

    features = merged.select_dtypes(include=np.number)
    features = features.drop(columns=[c for c in ("result", "home_win") if c in features.columns])
    features = features.loc[:, ~features.columns.duplicated()]

    if feature_columns is not None:
        features = features.reindex(columns=list(feature_columns))

    return meta, features


def fit_imputation(features: pd.DataFrame) -> pd.Series:
    """Column means for numeric imputation; all-null columns map to 0."""
    means = features.mean(numeric_only=True)
    return means.fillna(0.0)


def apply_imputation(features: pd.DataFrame, means: pd.Series) -> pd.DataFrame:
    """Fill NaNs with the supplied (training) means, then 0 as a last resort."""
    return features.fillna(means).fillna(0.0)
