"""Shared feature engineering for the NFL player-props models.

Player-level data source: nflreadpy. `load_player_stats(summary_level="week")`
is both the feature source (lagged) and the label. Snap counts key on PFR
player ids while everything else keys on gsis ids, so `load_id_bridge` (via
`load_rosters_weekly`, which carries both) is what lets `attach_snap_share`
join them. `load_injuries` flags players unlikely to suit up. Team-level game
context (implied point total from the closing spread/total) is derived from
`nfl_features.load_schedules` so props share the same market-context prior as
the game-level models.

No player-prop sportsbook lines exist in any data source available to this
project (nflverse/sportsdataverse carry none) - there is nothing to train an
"edge vs market" against. These are point projections; `player_props_common.py`'s
backtest decides whether a market's projections are reliable enough to trust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import nflreadpy as nfl

# nflverse snap-count data (the usage/reliability signal) only goes back to
# 2012; player_stats/injuries/rosters cover more, but capping here keeps the
# same feature set available across the whole training range.
FIRST_SEASON = 2012

LAG_WINDOW = 5

STAT_COLUMNS = [
    "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "target_share", "air_yards_share",
]

PLAYER_STAT_COLS = [
    "player_id", "player_display_name", "position", "position_group",
    "season", "week", "season_type", "game_id", "team", "opponent_team",
] + STAT_COLUMNS


def load_player_stats(seasons: list[int]) -> pd.DataFrame:
    """Weekly player stats, regular season only (playoff usage/roles are a
    different distribution and would distort the rolling averages)."""
    frames = []
    for year in seasons:
        try:
            df = nfl.load_player_stats(seasons=year, summary_level="week").to_pandas()
        except Exception as exc:  # noqa: BLE001 - nflreadpy raises bare exceptions on 404
            if "404" in str(exc):
                print(f"  (no player stats published for {year} yet - skipping)")
                continue
            raise
        frames.append(df[df["season_type"] == "REG"])
    if not frames:
        raise RuntimeError("No player stats could be loaded for any requested season.")
    out = pd.concat(frames, ignore_index=True)
    return out[[c for c in PLAYER_STAT_COLS if c in out.columns]].copy()


def load_snap_counts(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for year in seasons:
        try:
            frames.append(nfl.load_snap_counts(seasons=year).to_pandas())
        except Exception as exc:  # noqa: BLE001
            if "404" in str(exc) or "must be between" in str(exc):
                print(f"  (no snap counts published for {year} yet - skipping)")
                continue
            raise
    if not frames:
        raise RuntimeError("No snap counts could be loaded for any requested season.")
    return pd.concat(frames, ignore_index=True)


def _bypass_season_gate(path_template: str, season: int) -> pd.DataFrame:
    """nflreadpy's own loaders reject any season past their internal
    Thursday-after-Labor-Day estimate of "current season" outright, even once
    that season's data is actually published on nflverse-data - verified
    directly against the release, this can lag the real schedule by a few days
    right at kickoff (e.g. a season whose real opener is a day before the
    formula's cutoff). Hit the exact same download path the rejecting loader
    would have, bypassing only the date check - used exclusively as a fallback
    once the loader has already refused the season."""
    from nflreadpy.downloader import get_downloader
    return get_downloader().download(
        "nflverse-data", path_template.format(season=season), season=season).to_pandas()


def load_rosters_weekly_safe(season: int) -> pd.DataFrame:
    try:
        return nfl.load_rosters_weekly(seasons=season).to_pandas()
    except Exception as exc:  # noqa: BLE001
        if "must be between" in str(exc):
            return _bypass_season_gate("weekly_rosters/roster_weekly_{season}", season)
        raise


def load_injuries(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for year in seasons:
        try:
            frames.append(nfl.load_injuries(seasons=year).to_pandas())
        except Exception as exc:  # noqa: BLE001
            if "404" in str(exc):
                print(f"  (no injury reports published for {year} yet - skipping)")
                continue
            if "must be between" in str(exc):
                try:
                    frames.append(_bypass_season_gate("injuries/injuries_{season}", year))
                    continue
                except Exception:  # noqa: BLE001
                    print(f"  (no injury reports published for {year} yet - skipping)")
                    continue
            raise
    if not frames:
        # A season with no injury reports yet is a legitimate state (e.g. the
        # only season requested is the one about to start), not a failure -
        # `attach_injury_status` merges an empty frame into all-NaN status fine.
        return pd.DataFrame(columns=["gsis_id", "season", "week", "report_status"])
    return pd.concat(frames, ignore_index=True)


def load_id_bridge(seasons: list[int]) -> pd.DataFrame:
    """gsis_id <-> pfr_id map (snap counts key on PFR ids; everything else uses gsis)."""
    frames = []
    for year in seasons:
        try:
            frames.append(load_rosters_weekly_safe(year))
        except Exception as exc:  # noqa: BLE001
            if "404" in str(exc):
                continue
            raise
    ros = pd.concat(frames, ignore_index=True)
    return (ros[["gsis_id", "pfr_id"]].dropna().drop_duplicates("gsis_id")
            .rename(columns={"gsis_id": "player_id"}))


def attach_snap_share(player_frame: pd.DataFrame, snaps: pd.DataFrame,
                      id_bridge: pd.DataFrame) -> pd.DataFrame:
    """Add `offense_pct` onto the player-week frame, bridged pfr_player_id -> player_id."""
    bridged = snaps.merge(id_bridge, left_on="pfr_player_id", right_on="pfr_id", how="left")
    bridged = bridged[["player_id", "season", "week", "offense_pct"]].dropna(subset=["player_id"])
    return player_frame.merge(bridged, on=["player_id", "season", "week"], how="left")


def attach_injury_status(player_frame: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Add that week's `report_status` (Out / Doubtful / Questionable / NaN=no report)."""
    inj = injuries.rename(columns={"gsis_id": "player_id"})[
        ["player_id", "season", "week", "report_status"]
    ].drop_duplicates(["player_id", "season", "week"], keep="last")
    return player_frame.merge(inj, on=["player_id", "season", "week"], how="left")


# A player on the field for at least this share of offensive snaps "played" -
# a game with 0 targets/attempts then is a real 0 (a lost over, a won under), not
# a non-event. Below it (a 2-snap goal-line package) books treat the player as
# barely in the market anyway.
MIN_PLAYED_SNAP_PCT = 0.10

# Snap counts and schedules use era team codes; player_stats uses current ones.
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def load_played_frame(seasons: list[int], positions: list[str],
                      participation_col: str) -> pd.DataFrame:
    """Player-weeks at `positions` where the player recorded `participation_col`
    OR played >= MIN_PLAYED_SNAP_PCT of offensive snaps, with `offense_pct` and
    that week's `report_status` attached.

    Games a player suited up for but got 0 targets/attempts are kept as zeros:
    for a prop they settle as a loss on the over, so dropping them (as a
    participation > 0 filter does) makes every over look likelier than it is -
    ~5% of WR/TE games at >=50% snap share. player_stats has no row at all for
    many of those, so they're rebuilt from snap counts."""
    stats = load_player_stats(seasons)
    stats = stats[stats["position"].isin(positions)]
    snaps = load_snap_counts(seasons)
    snaps = snaps[snaps["game_type"] == "REG"]
    id_bridge = load_id_bridge(seasons)
    stats = attach_snap_share(stats, snaps, id_bridge)

    sn = snaps[snaps["position"].isin(positions)
               & (snaps["offense_pct"] >= MIN_PLAYED_SNAP_PCT)]
    sn = sn.merge(id_bridge, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    key = ["player_id", "season", "week"]
    sn = sn.drop_duplicates(key)
    missing = sn.merge(stats[key], on=key, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"]
    extra = pd.DataFrame({
        "player_id": missing["player_id"], "player_display_name": missing["player"],
        "position": missing["position"], "position_group": missing["position"],
        "season": missing["season"], "week": missing["week"], "season_type": "REG",
        "game_id": missing["game_id"], "team": missing["team"].replace(TEAM_FIX),
        "opponent_team": missing["opponent"].replace(TEAM_FIX),
        "offense_pct": missing["offense_pct"],
    })
    for c in STAT_COLUMNS:
        extra[c] = 0.0
    stats = pd.concat([stats, extra], ignore_index=True)
    stats[STAT_COLUMNS] = stats[STAT_COLUMNS].astype(float).fillna(0.0)

    played = (stats[participation_col] > 0) | (stats["offense_pct"].fillna(0) >= MIN_PLAYED_SNAP_PCT)
    return attach_injury_status(stats[played].copy(), load_injuries(seasons))


def lag_player_frame(df: pd.DataFrame, stat_cols: list[str],
                     window: int = LAG_WINDOW) -> pd.DataFrame:
    """Per-player rolling mean of a stat's *previous* games (leak-free), plus a
    career-games-played counter used for the LOW_SAMPLE reliability flag.

    Rolls across season boundaries deliberately - a player's recent form
    carries over the off-season better than resetting it like team stats
    (whose opponents/scheme reset with the roster each year).
    """
    df = df.sort_values(["player_id", "season", "week"])
    lagged = df.groupby("player_id")[stat_cols].transform(
        lambda x: x.shift().rolling(window=window, min_periods=1).mean()
    )
    lagged.columns = [f"lag_{c}" for c in lagged.columns]
    lagged["games_played"] = df.groupby("player_id").cumcount().values
    for k in ("player_id", "season", "week"):
        lagged[k] = df[k].values
    return lagged


def build_team_implied_totals(schedules: pd.DataFrame) -> pd.DataFrame:
    """Each team's game-implied point total from the closing spread + total -
    a strong prior for player yardage/TD volume even with no player-specific line."""
    schedules = schedules.assign(home_team=schedules["home_team"].replace(TEAM_FIX),
                                 away_team=schedules["away_team"].replace(TEAM_FIX))
    home = pd.DataFrame({
        "team": schedules["home_team"], "season": schedules["season"], "week": schedules["week"],
        "opponent": schedules["away_team"],
        "team_implied_total": schedules["total_line"] / 2 + schedules["spread_line"] / 2,
        "is_home": 1.0,
    })
    away = pd.DataFrame({
        "team": schedules["away_team"], "season": schedules["season"], "week": schedules["week"],
        "opponent": schedules["home_team"],
        "team_implied_total": schedules["total_line"] / 2 - schedules["spread_line"] / 2,
        "is_home": 0.0,
    })
    out = pd.concat([home, away], ignore_index=True)
    opp = out[["team", "season", "week", "team_implied_total"]].rename(
        columns={"team": "opponent", "team_implied_total": "opp_implied_total"})
    return out.merge(opp, on=["opponent", "season", "week"], how="left")


REPORT_STATUS_FLAG = {"Out": "OUT", "Doubtful": "DOUBTFUL", "Questionable": "QUESTIONABLE"}
LOW_SAMPLE_GAMES = 3
BACKUP_SNAP_SHARE = 0.35


def add_reliability_flags(df: pd.DataFrame) -> pd.Series:
    """One human-readable '+'-joined flag string per row (empty = no concern)."""
    low_sample = np.where(df["games_played"] < LOW_SAMPLE_GAMES, "LOW_SAMPLE", "")
    backup = np.where(df["lag_offense_pct"].fillna(0) < BACKUP_SNAP_SHARE, "BACKUP_ROLE", "")
    status = df.get("report_status", pd.Series("", index=df.index)).map(
        REPORT_STATUS_FLAG).fillna("").to_numpy()
    stacked = np.stack([low_sample, backup, status], axis=1)
    return pd.Series(["+".join(p for p in row if p) for row in stacked], index=df.index)
