# GTA VI trophy rarity forecast

Entry for the make.fun bounty `gta6-trophy-rarity-forecast-01`: forecast
each PlayStation 5 trophy's Exophase earn rate at 2026-11-26T00:00:00Z as
five quantiles, scored by mean pinball loss on the logit scale.

## Program

`main.py` (Python 3.13, standard library only) reads every `*.in` case in
`$MAKEFUN_INPUT_DIR` (default `/input`) and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (default `/output`).

## Model

A trophy's rate is `earners / players_tracked`, and both counts move only
when Exophase refreshes the page. So the forecaster tracks the two counts
separately and replays the refresh mechanism forward:

- **Players** grow by an influx of newly syncing players that decays
  exponentially after launch (rate and decay fitted from the observed
  growth; a launch-night prior from the aggregates the rules publish
  when the history is still flat).
- **At each refresh**, new players bring trophies with them (a new cohort
  earns trophy *j* with probability ~ *c* × its current rate, *c* fitted
  from the hard trophies, where already-tracked players barely earn),
  and tracked players keep earning toward the trophy's natural ceiling:
  earning is proportional to the headroom between the trophy's eventual
  rate *R_j* (fitted from past refreshes, shrunk toward its tier) and
  its current earners, so a platinum nobody can finish in a week
  saturates low while an early story trophy climbs. The earning pace
  decays at the same rate as the influx.
- **Refreshes** arrive about every `interval` hours (median observed gap,
  launch-night prior otherwise), the page is already `staleness` hours
  overdue one, and the interval itself is uncertain, which gives a
  distribution over how many refreshes land before the target.

The forecast is a mixture over refresh counts: scenario *m* replays *m*
refreshes, spaced evenly over the horizon (the first one also dumps the
backlog accumulated since the last refresh), and the five quantiles are
the mixture's quantiles on the logit scale. With no refresh yet seen and
a stale page, the low quantiles already price in the dilution a refresh
brings, while the high ones allow for no refresh at all; over a long
horizon the scenarios concentrate on the integral path and the spread
comes from the random walk and parameter uncertainty.

## Local validation

Against practice answers reconstructed from the aggregates the rules
publish (04:30Z read: 3,868 players tracked, 4,276 trophies earned,
every rate but the 100% one lower), mean pinball loss over p00/p01:

| answer hypothesis | this model | persistence baseline |
|---|---:|---:|
| proportional dilution | 0.31 | 1.40 |
| easiest trophies first | 0.42 | 1.52 |
| even spread | 0.31 | 1.39 |
| pure dilution | 0.47 | 1.69 |

On synthetic final-like cases (two days of reads, refreshes every few
hours, decaying influx, saturating earning), the model also beats
persistence (0.31 vs 0.37, and 0.33 vs 0.54 under a second generating
process). Runtime is ~0.2 s per practice case and ~1 s per final-like
case; output is deterministic.

## License

MIT — see `LICENSE`.
