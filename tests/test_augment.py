import numpy as np
from helpers import make_sequence

from augment import NORMAL, STRONG, augment
from landmarks import FRAME_FEATURES, SEQUENCE_LENGTH, to_sample

EXPECTED_KEYS = {"crop", "warp", "roll", "yaw", "amp", "noise", "drop", "mirror"}


def test_configs_have_expected_keys():
    assert EXPECTED_KEYS <= set(NORMAL)
    assert EXPECTED_KEYS <= set(STRONG)


def test_augment_output_shape_and_dtype():
    sample = to_sample(make_sequence(15, y_right=0.4))
    out = augment(sample, np.random.default_rng(0), NORMAL)
    assert out.shape == (SEQUENCE_LENGTH, FRAME_FEATURES)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_augment_is_deterministic_with_same_seed():
    sample = to_sample(make_sequence(15, y_right=0.4))
    a = augment(sample, np.random.default_rng(42), NORMAL)
    b = augment(sample, np.random.default_rng(42), NORMAL)
    assert np.array_equal(a, b)


def test_augment_on_empty_sample_does_not_crash():
    empty = np.zeros((SEQUENCE_LENGTH, FRAME_FEATURES), dtype=np.float32)
    out = augment(empty, np.random.default_rng(0), STRONG)
    assert out.shape == (SEQUENCE_LENGTH, FRAME_FEATURES)
    assert np.allclose(out, 0.0)
