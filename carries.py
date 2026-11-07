"""Carry measures from SPADL event data.

A "carry" is the SPADL ``dribble`` action: the ball moved between two consecutive events of the same team and the next actor is credited. It is inferred
from event data, so it says where the ball went with a player, not how fast or under what pressure. A "take-on" is the SPADL ``take_on`` action
(an attempt to beat an opponent), counted whether or not it succeeded (success rates drift across eras, so they are not used).
"""
import numpy as np
import pandas as pd

PROG_METRES = 10      # a progressive carry moves the ball this many metres closer to the goal centre
MIN_MINUTES = 450     # below this, per-90 measures are NaN
KEY = ['player_id', 'season', 'half']
LOGGED = ['car90', 'prog90', 'to90']


def cell_index(x, y):
    """Index into the flattened 12 x 8 xT grid (row-major, 8 rows of 12 columns)."""
    col = np.clip(np.floor(np.asarray(x, float) / 105 * 12), 0, 11).astype(int)
    row = np.clip(np.floor(np.asarray(y, float) / 68 * 8), 0, 7).astype(int)
    return row * 12 + col


def carry_sums(actions: pd.DataFrame, games: pd.DataFrame, parity: pd.DataFrame, minutes: pd.DataFrame, xt_flat: np.ndarray) -> pd.DataFrame:
    """One row per (player_id, season, half) with n, prog, xt, gain, takeon, minutes (sums). Away teams are flipped to attack toward x = 105."""
    a = actions[actions.type_name.isin(['dribble', 'take_on'])].merge(games[['game_id', 'home_team_id']], on='game_id')
    away = (a.team_id != a.home_team_id).to_numpy()
    for c, m in (('start_x', 105), ('end_x', 105), ('start_y', 68), ('end_y', 68)):
        a[c] = np.where(away, m - a[c].to_numpy(float), a[c].to_numpy(float))
    a = a.dropna(subset=['player_id']).assign(player_id=lambda d: d.player_id.astype('int64')).merge(parity, on=['player_id', 'season', 'game_id'])
    c = a[a.type_name == 'dribble'].dropna(subset=['start_x', 'start_y', 'end_x', 'end_y']).copy()
    gain = np.hypot(105 - c.start_x, 34 - c.start_y) - np.hypot(105 - c.end_x, 34 - c.end_y)
    c['prog'] = (gain >= PROG_METRES).astype(float)
    c['gain'] = gain.clip(lower=0)
    c['xt'] = xt_flat[cell_index(c.end_x, c.end_y)] - xt_flat[cell_index(c.start_x, c.start_y)]
    c['n'] = 1.0
    cs = c.groupby(KEY)[['n', 'prog', 'xt', 'gain']].sum()
    cs['takeon'] = a[a.type_name == 'take_on'].groupby(KEY).size()
    mn = minutes.merge(parity, on=['player_id', 'season', 'game_id']).groupby(KEY).minutes.sum()
    out = cs.join(mn, how='outer').fillna(0.0).reset_index()
    return out[KEY + ['n', 'prog', 'xt', 'gain', 'takeon', 'minutes']]


def carry_measures(sums: pd.DataFrame, half: int | None = None) -> pd.DataFrame:
    """car90, prog90, to90, xt90c, gain90 and to_per_carry (take-ons per carry: the dribbler index) per (player_id, season). half=None sums both halves; rows under the minutes floor are NaN."""
    s = sums if half is None else sums[sums.half == half]
    g = s.groupby(['player_id', 'season'])[['n', 'prog', 'xt', 'gain', 'takeon', 'minutes']].sum()
    mn = g.minutes.where(g.minutes > 0)
    ok = g.minutes >= (MIN_MINUTES if half is None else MIN_MINUTES / 2)
    out = pd.DataFrame({'car90': g.n / mn * 90, 'prog90': g.prog / mn * 90, 'to90': g.takeon / mn * 90, 'xt90c': g.xt / mn * 9000, 'gain90': g.gain / mn * 90, 'to_per_carry': g.takeon / g.n.clip(lower=1)})
    return out.where(ok)


def _z(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    r = np.log1p(ref[LOGGED].clip(lower=0)); d = np.log1p(df[LOGGED].clip(lower=0))
    return (d - r.mean()) / r.std()


def carry_composite(measures: pd.DataFrame, ref: pd.DataFrame) -> pd.Series:
    """Mean z-score of log1p(car90), log1p(prog90), log1p(to90); z-parameters from ref (the position group's full-season table)."""
    return _z(measures, ref).mean(axis=1, skipna=False)


def tier_cuts(composite: pd.Series) -> tuple[float, float]:
    """Terciles of the full-season composite."""
    q = np.nanquantile(composite.to_numpy(float), [1 / 3, 2 / 3])
    return float(q[0]), float(q[1])


def carry_tier(composite: pd.Series, cuts: tuple[float, float]) -> pd.Series:
    """'low' / 'mid' / 'high' by the cut points; NaN stays NaN."""
    t = pd.Series(np.where(composite < cuts[0], 'low', np.where(composite < cuts[1], 'mid', 'high')), index=composite.index, dtype=object)
    return t.where(composite.notna())
