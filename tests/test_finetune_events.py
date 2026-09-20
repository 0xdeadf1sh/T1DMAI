import numpy as np

import finetune_data as fd

NAN = np.nan


def _rows(**cols):
    n = len(cols['idx'])
    base = {c: np.full(n, NAN) for c in ('carbs', 'basal', 'bolus', 'insulin')}
    base['is_mdi'] = np.zeros(n, dtype=bool)
    return base | {k: np.asarray(v) for k, v in cols.items()}


def _events(n, r, bolus_type=None, basal_type=None):
    return fd._events_to_curves(
        n, r['idx'], r['carbs'], r['basal'], r['bolus'], r['insulin'],
        r['is_mdi'], bolus_type, basal_type)


def test_event_channels_match_the_layout():
    r = _rows(idx=[10], carbs=[40.0])
    *_, ev = _events(200, r)
    assert tuple(ev) == fd.EVENT_CHANNELS
    assert all(v.dtype == np.float32 and v.shape == (200,) for v in ev.values())


def test_doses_land_on_their_slot_and_conserve_the_curve_total():
    r = _rows(idx=[10, 10, 50], carbs=[40.0, NAN, NAN], bolus=[NAN, 4.0, 2.0])
    carb, ins, ev = _events(400, r)
    assert ev['carb_g'][10] == 40.0 and ev['carb_g'].sum() == 40.0
    assert ev['carb_gi'][10] == fd.CARB_GI_DEFAULT
    assert ev['bolus_u'][10] == 4.0 and ev['bolus_u'][50] == 2.0
    np.testing.assert_allclose(carb.sum(), 40.0, rtol=1e-3)
    np.testing.assert_allclose(ins.sum(), 6.0, rtol=1e-3)


def test_descriptors_are_zero_off_dose_and_dose_weighted_on_a_shared_slot():
    r = _rows(idx=[10, 10], bolus=[1.0, 9.0])
    *_, ev = _events(400, r)
    bv = fd.bolus_variant(None)
    peaks = [fd.gamma_peak_min(*fd.bolus_pk_for_dose(
        d, bv['gamma_k'], bv['gamma_theta'], bv['dia_base_hours'])[:2]) for d in (1.0, 9.0)]
    np.testing.assert_allclose(ev['bolus_peak_min'][10], 0.1 * peaks[0] + 0.9 * peaks[1],
                               rtol=1e-5)
    assert np.count_nonzero(ev['bolus_peak_min']) == 1


def test_pump_basal_carries_the_rapid_insulin_descriptors():
    r = _rows(idx=[0, 1, 2], basal=[0.1, 0.1, 0.1])
    *_, ev = _events(400, r, bolus_type='Fiasp')
    bv = fd.bolus_variant('Fiasp')
    np.testing.assert_allclose(ev['basal_u'][:3], 0.1, rtol=1e-6)
    np.testing.assert_allclose(ev['basal_dur_h'][:3], bv['dia_base_hours'], rtol=1e-6)
    assert ev['bolus_u'].sum() == 0.0


def test_mdi_injection_carries_the_long_acting_descriptors():
    r = _rows(idx=[5], basal=[20.0], is_mdi=[True])
    *_, ev = _events(400, r, basal_type='insulin detemir')
    av = fd.basal_variant('insulin detemir')
    assert ev['basal_u'][5] == 20.0
    np.testing.assert_allclose(ev['basal_dur_h'][5], av['action_hours'], rtol=1e-6)
    np.testing.assert_allclose(
        ev['basal_peak_min'][5], fd.bateman_peak_min(av['ka'], av['ke']), rtol=1e-5)
