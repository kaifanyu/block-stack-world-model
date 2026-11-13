import pytest

from assignment.experiments.distill_campaign import scene_bootstrap, validate_panel, wilson


def test_paired_clusters_preserve_duplicate_samples():
    rows = [dict(scene_seed=s, sample_seed=j, success=s == 1) for s in (1, 2) for j in (0, 1)]
    same = scene_bootstrap(rows, rows, draws=100)
    assert same['mean'] == 0 and same['ci95'] == [0, 0] and same['scenes'] == 2
    with pytest.raises(ValueError):
        scene_bootstrap(rows, rows[:-1])
    with pytest.raises(ValueError):
        scene_bootstrap(rows + rows[:1])


def test_wilson_extremes_and_protected_panels():
    assert wilson(0, 100)[0] == 0
    assert wilson(100, 100)[1] == 1
    with pytest.raises(ValueError):
        validate_panel(dict(scenes=[104000], sample_seeds=[0, 1], renderer='osmesa', mujoco='3.13.0'))
