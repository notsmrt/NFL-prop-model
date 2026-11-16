# NFL Player Props

CatBoost models that project NFL player stat lines for the upcoming week, one
model per market. The data comes from nflverse and gets pulled automatically
through `nflreadpy`, so there's no API key or dataset to download.

Markets: `pass_yards`, `pass_tds`, `pass_interceptions`, `rush_yards`,
`rush_tds`, `receptions`, `rec_yards`, `rec_tds`

## Setup

I'm on Python 3.12.

```
pip install nflreadpy pandas numpy scipy catboost joblib pyarrow
```

## Usage

Train a market, then predict it:

```
python player_props_model.py train --market pass_yards
python player_props_model.py predict --market pass_yards
```

Training takes around 20 seconds per market and saves to `artifacts/`. Predict
prints every player for the upcoming week:

```
         player  team  opponent  projected    low   high  baseline
     Joe Burrow   CIN       PIT      276.6  171.7  381.6     328.6
    Brock Purdy    SF       DAL      274.1  169.1  379.0     238.2
   Dak Prescott   DAL        SF      266.6  161.7  371.6     263.2
```

`baseline` is the player's own recent average. There's also a `flags` column
for things like `LOW_SAMPLE`, `BACKUP_ROLE` and `OUT`. Use `--season` and
`--week` if you want a specific week.

To check a projection against a line at your book, make a csv:

```
player,line
Drake Maye,234.5
Josh Allen,229.5
```

```
python player_props_model.py predict --market pass_yards --line-file lines.csv
```

That adds `prob_over_line`, the model's chance of the over.

## Testing

```
python player_props_model.py backtest --market pass_yards
```

This goes season by season from 2015, trains on the seasons before, predicts
the one it hasn't seen, and compares the error to just using each player's
recent average. A market gets marked RELIABLE if the model is at least 3%
better, and predict shows that label next to every projection.

Where each market stood at week 10 of 2026:

```
pass_yards           +6.1%   reliable
pass_interceptions   +3.9%   reliable
pass_tds             +3.7%   reliable
rec_yards            +2.8%
receptions           +2.8%
rush_yards           +1.7%
rush_tds             -2.2%
rec_tds              -2.1%
```

So the passing markets are the ones worth using. The TD markets for rushers and
receivers do worse than a plain average.

There's no free source of historical prop lines, so the backtest can't tell you
whether any of this beats a sportsbook. The `--line-file` option is there so you
can track that yourself going forward.

## Files

- `player_props_model.py` the script you run
- `player_props_common.py` training, backtest and predict logic
- `player_props_features.py` player features
- `nfl_features.py` schedule and team data loading
