import numpy as np
import pytest
from helpers import make_sequence

from features import FEATURE_SIZE, FEATURES_PER_FRAME, batch_features, canonicalize, sample_features, to_hands
from landmarks import SEQUENCE_LENGTH, to_sample


def _a_sample() -> np.ndarray:
    return to_sample(make_sequence(10, y_right=0.4))


def test_feature_size_is_consistent():
    assert FEATURE_SIZE == SEQUENCE_LENGTH * FEATURES_PER_FRAME


def test_sample_features_shape_and_finite():
    feats = sample_features(_a_sample())
    assert feats.shape == (FEATURE_SIZE,)
    assert np.isfinite(feats).all()


def test_sample_features_is_deterministic():
    sample = _a_sample()
    assert np.array_equal(sample_features(sample), sample_features(sample))


def test_sample_features_rejects_wrong_shape():
    with pytest.raises(ValueError):
        sample_features(np.zeros((SEQUENCE_LENGTH, 100), dtype=np.float32))


def test_batch_features_stacks_rows():
    samples = [_a_sample() for _ in range(3)]
    batch = batch_features(samples)
    assert batch.shape == (3, FEATURE_SIZE)


def test_canonicalize_keeps_shape():
    hands = to_hands(_a_sample())
    assert canonicalize(hands).shape == hands.shape
