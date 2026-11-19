"""GTA VI trophy-rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case target as five
quantiles, from the house's recorded reads of the trophy page.

Model (built from the case data alone, standard library only):

  rate = earners / players_tracked, and both move only when Exophase
  refreshes the page.  So instead of extrapolating the rate itself, the
  forecaster tracks the two counts separately and replays the refresh
  mechanism forward:

  * players_tracked grows by an influx of newly syncing players that
    decays exponentially after launch (rate and decay fitted from the
    observed growth; a launch-night prior when the history is flat);
  * at each refresh, new players bring trophies with them (a new cohort
    earns trophy j with probability ~ c * its current rate, c fitted
    from the hard trophies, where already-tracked players barely earn),
    and players already tracked keep earning toward the trophy's
    natural ceiling: earning is proportional to the headroom between
    the trophy's eventual rate R_j (fitted from past refreshes,
    shrunk toward its tier) and its current earners, so a platinum
    nobody can finish in a week saturates low while an early story
    trophy climbs; the earning pace decays at the same rate as the
    influx;
  * refreshes arrive about every `interval` hours (median observed gap,
    launch-night prior otherwise), the page is already `staleness`
    hours overdue one, and the interval itself is uncertain, which
    gives a distribution over how many refreshes land before the
    target.

  The forecast is a mixture over refresh counts: scenario m replays m
  refreshes, spaced evenly over the horizon (the first one also dumps
  the backlog accumulated since the last refresh), and the five
  quantiles are the mixture's quantiles on the logit scale.  That gives
  an honest spread: with no refresh yet seen and a stale page, the low
  quantiles already price in the dilution a refresh brings, while the
  high ones allow for no refresh at all; over a long horizon the
  scenarios concentrate on the integral path and the spread comes from
  the random walk and parameter uncertainty.

  Everything is deterministic: same input bytes give same output bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# --- launch-night priors, taken from the aggregates the rules publish
# (page frozen at 638 players / 943 trophies earned, then refreshed to
# 3,868 players / 4,276 trophies: 3,230 new players in ~4.7 h, of which
# all but 103 trophies were the one trophy everybody holds).
PRIOR_INFLUX_PER_HOUR = 3230.0 / 4.7        # new players per hour
PRIOR_REFRESH_INTERVAL_H = 4.7               # hours between refreshes
PRIOR_NEW_COHORT_SCALE = 0.067              # new cohort earns j at
#                                            # 0.067 * its current rate
PRIOR_INTERVAL_LOG_SD = 0.35                # uncertainty of the
#                                            # refresh interval (log)

TRIVIAL_RATE = 99.9   # a trophy (nearly) every tracked player holds


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


class CaseModel:
    """Everything the forecast needs, fitted from one case's history."""

    def __init__(self, case):
        self.case = case
        self.trophies = case["trophies"]
        self.n = len(self.trophies)
        self.history = case.get("history") or []
        self.horizon_h = case["horizon_minutes"] / 60.0
        release = case.get("game", {}).get("release_date", "2026-11-19")
        self.epoch = parse_time(release + "T00:00:00Z")
        self._build_series()
        self._find_refreshes()
        self._fit_params()

    # -- data ---------------------------------------------------------
    def _hours(self, iso):
        return (parse_time(iso) - self.epoch).total_seconds() / 3600.0

    def _build_series(self):
        self.times = []
        self.players = []
        self.rates = []          # per read: list of rates (float or None)
        for read in self.history:
            self.times.append(self._hours(read["at"]))
            self.players.append(float(read["players_tracked"]))
            self.rates.append(list(read["rates"]))
        if not self.times:
            self.times = [0.0]
            self.players = [1.0]
            self.rates = [[None] * self.n]
        self.last_idx = len(self.times) - 1
        self.t_last = self.times[self.last_idx]
        self.p_last = self.players[self.last_idx]
        self.last_rates = []
        for j in range(self.n):
            rate = None
            for i in range(self.last_idx, -1, -1):
                if self.rates[i][j] is not None:
                    rate = float(self.rates[i][j])
                    break
            self.last_rates.append(rate if rate is not None else 1.0)
        self.earners_last = [r * self.p_last / 100.0 for r in self.last_rates]

    # -- refreshes ----------------------------------------------------
    def _find_refreshes(self):
        """Indices i where read i shows a refreshed page vs read i-1."""
        self.refresh_idx = []
        for i in range(1, len(self.times)):
            if self.players[i] != self.players[i - 1]:
                self.refresh_idx.append(i)
                continue
            changed = any(
                self.rates[i][j] is not None
                and self.rates[i - 1][j] is not None
                and self.rates[i][j] != self.rates[i - 1][j]
                for j in range(self.n))
            if changed:
                self.refresh_idx.append(i)
        if self.refresh_idx:
            self.t_last_refresh = self.times[self.refresh_idx[-1]]
            gaps = [self.times[self.refresh_idx[k]]
                    - self.times[self.refresh_idx[k - 1]]
                    for k in range(1, len(self.refresh_idx))]
            self.interval = median(gaps) if gaps else PRIOR_REFRESH_INTERVAL_H
        else:
            self.t_last_refresh = self.times[0]
            self.interval = PRIOR_REFRESH_INTERVAL_H
        self.interval = min(48.0, max(0.5, self.interval))
        self.staleness = self.t_last - self.t_last_refresh
        if not self.refresh_idx:
            # Page was already stale at the first read; allow half an
            # hour of pre-history staleness on top of the observed span.
            self.staleness += 0.5

    # -- parameters ---------------------------------------------------
    def _fit_params(self):
        n_ref = len(self.refresh_idx)
        span = self.t_last - self.times[0]
        # Influx of new players per hour, recent window, and its decay.
        self.tau = float("inf")      # exponential decay time, hours
        if self.players[-1] > self.players[0] and span > 0:
            def window_rate(hours_back, window):
                cut = self.t_last - hours_back
                lo = None
                for i, t in enumerate(self.times):
                    if t >= cut - window:
                        lo = i
                        break
                if lo is None:
                    return None, 0.0
                hi = lo
                for i, t in enumerate(self.times):
                    if t <= cut:
                        hi = i
                wspan = self.times[hi] - self.times[lo]
                if wspan < 1.0:
                    return None, 0.0
                return (self.players[hi] - self.players[lo]) / wspan, wspan
            recent, _ = window_rate(0.0, 24.0)
            if recent is None:
                recent = (self.p_last - self.players[0]) / span
            self.influx = max(0.0, recent)
            if span >= 48.0:
                older, _ = window_rate(24.0, 24.0)
                if older and recent and older > recent > 0:
                    tau = 24.0 / math.log(older / recent)
                    self.tau = min(720.0, max(12.0, tau))
        else:
            self.influx = PRIOR_INFLUX_PER_HOUR

        # New-cohort scale c and the earning hazard h with per-trophy
        # ceilings R_j.
        #   dE_j = c * (r_j/100) * dP + h * (R_j/100 * P - E_j) * dt
        # The two terms are not jointly identified from a pooled fit
        # (dP and P*dt move together), so estimate them in turn: c
        # from the hard trophies, where tracked players barely earn
        # anything new, then (h, R_j) from the residual earning: h by
        # a grid search, R_j per trophy by least squares at that h.
        # Without a ceiling the hazard would claim every player
        # eventually earns every trophy, which no trophy does.
        w = n_ref / (n_ref + 2.0) if n_ref else 0.0
        c_hat = PRIOR_NEW_COHORT_SCALE
        self.hazard = 0.0
        self.ceiling = list(self.last_rates)
        if n_ref:
            hard = [j for j in range(self.n)
                    if self.last_rates[j] < 2.0
                    and self.last_rates[j] < TRIVIAL_RATE]
            if not hard:
                hard = [j for j in range(self.n)
                        if self.last_rates[j] < TRIVIAL_RATE]
            num = den = 0.0
            prev = 0
            for i in self.refresh_idx:
                dp = self.players[i] - self.players[prev]
                for j in hard:
                    rb = self.rates[prev][j]
                    ra = self.rates[i][j]
                    if rb is None or ra is None:
                        continue
                    num += (ra * self.players[i] - rb * self.players[prev]) / 100.0
                    den += (rb / 100.0) * dp
                prev = i
            if den > 0:
                c_hat = num / den
        self.cohort_scale = min(0.25, max(0.0,
            w * c_hat + (1.0 - w) * PRIOR_NEW_COHORT_SCALE))
        if n_ref:
            self._fit_ceiling(w)

        # Time to the next refresh, and the refresh-count distribution.
        # Interval ~ lognormal(median, PRIOR_INTERVAL_LOG_SD): count m
        # lands when the m-th refresh falls inside the horizon.
        self.delta = max(0.25, self.interval - min(self.staleness,
                                                    self.interval))
        self.count_weights = self._count_weights()

    def _fit_ceiling(self, w):
        """Fit the earning hazard h and each trophy's ceiling rate R_j.

        At each refresh, the residual earning (after the new cohort's
        share) is h * (R_j/100 * P - E_j) * dt.  For a fixed h this is
        linear in R_j, so R_j comes from least squares per trophy and
        h from a grid search on the total squared error.  Ceilings are
        shrunk toward their tier's median and floored at the current
        rate (a ceiling cannot sit below what is already earned).
        """
        tier_prior = {"platinum": 0.8, "gold": 6.0, "silver": 18.0,
                      "bronze": 40.0}
        obs = []   # (j, dt, p_prev, e_before, residual_dE)
        prev = 0
        for i in self.refresh_idx:
            dp = self.players[i] - self.players[prev]
            dt = self.times[i] - self.times[prev]
            p_prev = self.players[prev]
            for j in range(self.n):
                rb = self.rates[prev][j]
                ra = self.rates[i][j]
                if rb is None or ra is None or rb >= TRIVIAL_RATE:
                    continue
                e_before = rb * p_prev / 100.0
                de = (ra * self.players[i] - rb * p_prev) / 100.0
                resid = de - self.cohort_scale * (rb / 100.0) * dp
                obs.append((j, dt, p_prev, e_before, resid))
            prev = i
        if not obs:
            return
        obs_by_j = {}
        for o in obs:
            obs_by_j.setdefault(o[0], []).append(o)
        best = None
        for h in (0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032):
            ceilings = {}
            sse = 0.0
            for j in range(self.n):
                rows = obs_by_j.get(j, [])
                if not rows or h == 0.0:
                    ceilings[j] = self.last_rates[j]
                    continue
                num = den = 0.0
                for (_, dt, p_prev, e_before, resid) in rows:
                    x = h * p_prev * dt / 100.0
                    num += x * (resid + h * e_before * dt)
                    den += x * x
                ceilings[j] = num / den if den > 0 else self.last_rates[j]
            for (j, dt, p_prev, e_before, resid) in obs:
                pred = h * (ceilings[j] / 100.0 * p_prev - e_before) * dt
                sse += (resid - pred) ** 2
            if best is None or sse < best[0]:
                best = (sse, h, ceilings)
        _, h, ceilings = best
        self.hazard = h
        tier_med = {}
        for tier in tier_prior:
            vals = [ceilings[j] for j in range(self.n)
                    if self.trophies[j]["tier"] == tier]
            tier_med[tier] = median(vals) if vals else tier_prior[tier]
        for j in range(self.n):
            tier = self.trophies[j]["tier"]
            target = tier_med[tier]
            if not (0.0 < target < 100.0):
                target = tier_prior.get(tier, 40.0)
            r_fit = min(99.0, max(self.last_rates[j], ceilings[j]))
            self.ceiling[j] = w * (0.5 * r_fit + 0.5 * target) \
                + (1.0 - w) * tier_prior.get(tier, 40.0)
            if self.last_rates[j] >= TRIVIAL_RATE:
                self.ceiling[j] = 100.0

    def _count_weights(self):
        """P(exactly m refreshes before the target), m = 0..M.

        The interval I is lognormal around its median; for a given I
        the refreshes land at delta(I), delta(I)+I, ... with
        delta(I) = max(0.25, I - min(staleness, I)).  Integrate the
        count over the lognormal density on a fixed grid.
        """
        sd = PRIOR_INTERVAL_LOG_SD
        med = self.interval
        horizon = self.horizon_h
        counts = {}

        def count_for(interval):
            delta = max(0.25, interval - min(self.staleness, interval))
            if delta > horizon:
                return 0
            return 1 + int((horizon - delta) // interval)

        lo = math.log(med) - 4.0 * sd
        hi = math.log(med) + 4.0 * sd
        steps = 400
        total = 0.0
        for k in range(steps + 1):
            x = lo + (hi - lo) * k / steps
            interval = math.exp(x)
            # lognormal pdf in x (normal with mean log med, sd)
            pdf = math.exp(-0.5 * ((x - math.log(med)) / sd) ** 2)
            w = pdf * (1.0 if k in (0, steps) else 2.0)
            c = count_for(interval)
            counts[c] = counts.get(c, 0.0) + w
            total += w
        m_max = max(counts)
        weights = [counts.get(m, 0.0) / total for m in range(m_max + 1)]
        return weights

    # -- forward simulation -------------------------------------------
    def _influx_integral(self, a, b):
        """New players arriving between offsets a and b (hours from now)."""
        if b <= a:
            return 0.0
        if math.isinf(self.tau):
            return self.influx * (b - a)
        return self.influx * self.tau * (
            math.exp(-a / self.tau) - math.exp(-b / self.tau))

    def _decay_at(self, t):
        if math.isinf(self.tau):
            return 1.0
        return math.exp(-t / self.tau)

    def _scenario_rate(self, j, m):
        """Rate after replaying m refreshes, spaced over the horizon."""
        if m == 0:
            return self.last_rates[j]
        earners = self.earners_last[j]
        players = self.p_last
        rate = self.last_rates[j]
        horizon = self.horizon_h
        # Refresh times: first at delta, the rest evenly to the horizon.
        times = []
        if m == 1:
            times = [min(self.delta, horizon)]
        else:
            span = max(horizon - self.delta, 0.0)
            times = [self.delta + s * span / (m - 1) for s in range(m)]
        prev_t = -self.staleness  # backlog: first refresh covers the
        # accumulation since the last refresh, S + delta hours.
        for t in times:
            acc_start = prev_t
            acc_end = t
            dp = self._influx_integral(acc_start, acc_end)
            gap = acc_end - acc_start
            if rate >= TRIVIAL_RATE:
                new_earners = dp
            else:
                headroom = max(0.0, self.ceiling[j] / 100.0 * players
                               - earners)
                new_earners = (
                    self.cohort_scale * (rate / 100.0) * dp
                    + self.hazard * self._decay_at(t) * headroom * gap)
            earners += new_earners
            players += dp
            rate = 100.0 * earners / players if players > 0 else rate
            if rate >= 100.0:
                rate = 100.0
            prev_t = t
        return rate

    def forecast_trophy(self, j):
        if self.last_rates[j] >= TRIVIAL_RATE:
            return [100, 100, 100, 100, 100]
        weights = self.count_weights
        horizon_days = self.horizon_h / 24.0
        sigma = math.sqrt(0.25 * horizon_days + 0.09)
        mus = [logit(self._scenario_rate(j, m)) for m in range(len(weights))]
        out = []
        lo, hi = logit(0.01), logit(99.99)
        for tau in LEVELS:
            a, b = lo, hi
            for _ in range(80):
                mid = 0.5 * (a + b)
                cdf = sum(w * norm_cdf((mid - mu) / sigma)
                          for w, mu in zip(weights, mus))
                if cdf < tau:
                    a = mid
                else:
                    b = mid
            out.append(round4(clamp_pct(inv_logit(0.5 * (a + b)))))
        for i in range(1, 5):
            if out[i] < out[i - 1]:
                out[i] = out[i - 1]
        return out

    def forecast_case(self):
        rows = []
        for j, trophy in enumerate(self.trophies):
            rows.append({"id": trophy["id"],
                         "q": self.forecast_trophy(j)})
        return {"trophies": rows}


def forecast_case(case):
    return CaseModel(case).forecast_case()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
