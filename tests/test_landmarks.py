import numpy as np
import pytest
from helpers import make_hand, make_sequence

from landmarks import (
    FRAME_FEATURES,
    SAMPLE_SHAPE,
    SEQUENCE_LENGTH,
    hand_size,
    presence,
    resample_sequence,
    sample_at,
    to_sample,
)


def test_presence_detects_nonzero_hands():
    frames = make_sequence(5, y_right=0.5, y_left=None)  # só a direita presente
    pres = presence(frames)
    assert pres.shape == (5, 2)
    assert pres[:, 0].all()        # direita presente em todos os frames
    assert not pres[:, 1].any()    # esquerda ausente


def test_hand_size_matches_wrist_to_middle_distance():
    hand = make_hand(y=0.5, size=0.12)
    assert hand_size(hand) == pytest.approx(0.12, abs=1e-5)


def test_to_sample_has_canonical_shape():
    frames = make_sequence(10, y_right=0.4)
    sample = to_sample(frames)
    assert sample.shape == SAMPLE_SHAPE == (SEQUENCE_LENGTH, FRAME_FEATURES)
    assert np.isfinite(sample).all()


def test_resample_keeps_a_constant_sequence_constant():
    frames = make_sequence(12, y_right=0.3)
    out = resample_sequence(frames, length=SEQUENCE_LENGTH)
    assert out.shape[0] == SEQUENCE_LENGTH
    # A mão direita é constante, então toda posição reamostrada repete o mesmo frame.
    assert np.allclose(out[:, 0], frames[0, 0], atol=1e-5)
    # A mão esquerda nunca apareceu: continua zerada.
    assert np.allclose(out[:, 1], 0.0)


def test_sample_at_empty_returns_zeros_with_right_shape():
    empty = np.zeros((0, 2, 21, 3), dtype=np.float32)
    out = sample_at(empty, np.linspace(0, 0, SEQUENCE_LENGTH))
    assert out.shape == (SEQUENCE_LENGTH, 2, 21, 3)
    assert np.allclose(out, 0.0)
