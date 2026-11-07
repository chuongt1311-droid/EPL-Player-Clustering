import numpy as np
import pandas as pd
import pytest

import carries as cr

COLS = ['game_id', 'team_id', 'player_id', 'type_name', 'start_x', 'start_y', 'end_x', 'end_y', 'season']


def acts(rows):
    return pd.DataFrame(rows, columns=COLS)


GAMES = pd.DataFrame({'game_id': [1, 2], 'home_team_id': [10, 10]})
PARITY = pd.DataFrame({'player_id': [7, 7], 'season': ['2024-25'] * 2, 'game_id': [1, 2], 'half': [0, 1]})
MINUTES = pd.DataFrame({'player_id': [7, 7], 'season': ['2024-25'] * 2, 'game_id': [1, 2], 'minutes': [90.0, 80.0]})
ZERO = np.zeros(96)


def sums(rows, parity=PARITY, minutes=MINUTES, xt=ZERO):
    return cr.carry_sums(acts(rows), GAMES, parity, minutes, xt)


def test_away_carry_is_flipped():
    s = sums([(1, 20, 7, 'dribble', 80, 34, 60, 34, '2024-25')])  # team 20 is away
    assert s.loc[s.half == 0, 'prog'].iat[0] == 1
    assert s.loc[s.half == 0, 'gain'].iat[0] == pytest.approx(20)


def test_progressive_threshold():
    s = sums([(1, 10, 7, 'dribble', 30, 34, 39.9, 34, '2024-25'), (1, 10, 7, 'dribble', 30, 34, 40.0, 34, '2024-25')])
    assert s.loc[s.half == 0, 'prog'].iat[0] == pytest.approx(1)


def test_take_on_counted_regardless_of_result():
    s = sums([(1, 10, 7, 'take_on', 50, 30, 52, 30, '2024-25'), (1, 10, 7, 'take_on', 50, 30, 50, 30, '2024-25'), (1, 10, 7, 'pass', 50, 30, 60, 30, '2024-25')])
    assert s.loc[s.half == 0, 'takeon'].iat[0] == 2
    assert s.loc[s.half == 0, 'n'].iat[0] == 0


def test_halves_follow_parity():
    s = sums([(1, 10, 7, 'dribble', 30, 34, 45, 34, '2024-25'), (2, 10, 7, 'dribble', 30, 34, 45, 34, '2024-25'), (3, 10, 7, 'dribble', 30, 34, 45, 34, '2024-25')])
    assert sorted(s.half.tolist()) == [0, 1]
    assert s.set_index('half').n.to_dict() == {0: 1, 1: 1}


def test_xt_uses_flat_grid():
    g = np.zeros(96)
    g[0 * 12 + 11] = 0.5                                   # top row, last column
    s = sums([(1, 10, 7, 'dribble', 50, 34, 104, 1, '2024-25')], xt=g)
    assert s.loc[s.half == 0, 'xt'].iat[0] == pytest.approx(0.5)


def test_cell_index_matches_grid_layout():
    assert cr.cell_index(np.array([104.0]), np.array([1.0]))[0] == 11
    assert cr.cell_index(np.array([0.0]), np.array([67.9]))[0] == 7 * 12


def test_minutes_summed_per_half():
    s = sums([(1, 10, 7, 'dribble', 30, 34, 45, 34, '2024-25')])
    assert s.set_index('half').minutes.to_dict() == {0: 90.0, 1: 80.0}


def test_carry_without_minutes_row_gets_zero():
    s = sums([(1, 10, 7, 'dribble', 30, 34, 45, 34, '2024-25')], minutes=MINUTES.iloc[1:])
    assert s.set_index('half').minutes.to_dict()[0] == 0.0


def make_sums(minutes, n=18.0, prog=0.0, xt=0.0, gain=0.0, takeon=0.0, half=0):
    return pd.DataFrame({'player_id': [7], 'season': ['2024-25'], 'half': [half], 'n': [n], 'prog': [prog], 'xt': [xt], 'gain': [gain], 'takeon': [takeon], 'minutes': [minutes]})


def test_measures_per_90():
    m = cr.carry_measures(make_sums(900, n=18, prog=9, xt=0.2, takeon=4, gain=450))
    assert m.car90.iat[0] == pytest.approx(1.8)
    assert m.prog90.iat[0] == pytest.approx(0.9)
    assert m.to90.iat[0] == pytest.approx(0.4)
    assert m.xt90c.iat[0] == pytest.approx(2.0)
    assert m.gain90.iat[0] == pytest.approx(45.0)


def test_below_minutes_floor_is_nan():
    assert cr.carry_measures(make_sums(449)).isna().all(axis=None)
    assert cr.carry_measures(make_sums(450)).notna().all(axis=None)


def test_zero_carries_is_zero_not_nan():
    m = cr.carry_measures(make_sums(900, n=0, takeon=0))
    assert m.car90.iat[0] == 0.0 and m.to90.iat[0] == 0.0


def test_single_half_uses_half_floor():
    assert cr.carry_measures(make_sums(225), half=0).notna().all(axis=None)
    assert cr.carry_measures(make_sums(224), half=0).isna().all(axis=None)


def ref_table(n=300, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({'car90': r.gamma(5, 3, n), 'prog90': r.gamma(2, 1.5, n), 'to90': r.gamma(1.5, 1, n), 'xt90c': r.normal(2, 1, n), 'gain90': r.gamma(8, 10, n)})


def test_composite_nan_propagates():
    ref = ref_table(); probe = ref.iloc[:3].copy(); probe.iloc[1, 0] = np.nan
    c = cr.carry_composite(probe, ref)
    assert np.isnan(c.iat[1]) and not np.isnan(c.iat[0])


def test_tier_terciles():
    c = pd.Series(np.arange(9, dtype=float))
    t = cr.carry_tier(c, cr.tier_cuts(c))
    assert t.value_counts().to_dict() == {'low': 3, 'mid': 3, 'high': 3}


def test_tier_nan_stays_nan():
    c = pd.Series([0.0, 1.0, 2.0, np.nan])
    t = cr.carry_tier(c, cr.tier_cuts(c))
    assert pd.isna(t.iat[3])


def test_take_ons_per_carry_is_the_dribbler_index():
    m = cr.carry_measures(make_sums(900, n=20, takeon=5))
    assert m.to_per_carry.iat[0] == pytest.approx(0.25)
    assert cr.carry_measures(make_sums(900, n=0, takeon=0)).to_per_carry.iat[0] == 0.0
