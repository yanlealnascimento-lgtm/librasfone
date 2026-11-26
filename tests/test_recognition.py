import numpy as np
import pytest
from helpers import make_sequence

from features import FEATURE_SIZE
from recognition import predict_sign, validate_bundle


class FakeModel:
    """Modelo mínimo com a interface que o reconhecimento usa."""

    def __init__(self, probs, classes):
        self._probs = np.asarray(probs, dtype=float)
        self.classes_ = list(classes)

    def predict_proba(self, features):
        return np.tile(self._probs, (len(features), 1))


def _frames():
    return make_sequence(8, y_right=0.4)  # (8, 2, 21, 3), mão direita presente


def test_validate_bundle_accepts_a_consistent_bundle():
    labels = ["ola", "tchau", "_outro"]
    bundle = {"model": FakeModel([0.1, 0.1, 0.8], labels), "labels": labels, "feature_size": FEATURE_SIZE}
    assert validate_bundle(bundle) is bundle


def test_validate_bundle_rejects_wrong_feature_size():
    labels = ["ola", "tchau"]
    bundle = {"model": FakeModel([0.5, 0.5], labels), "labels": labels, "feature_size": FEATURE_SIZE + 1}
    with pytest.raises(ValueError):
        validate_bundle(bundle)


def test_validate_bundle_rejects_label_order_mismatch():
    labels = ["ola", "tchau"]
    model = FakeModel([0.5, 0.5], ["tchau", "ola"])  # ordem diferente de labels
    bundle = {"model": model, "labels": labels, "feature_size": FEATURE_SIZE}
    with pytest.raises(ValueError):
        validate_bundle(bundle)


def test_predict_sign_accepts_confident_known_word():
    labels = ["ola", "tchau", "_outro"]
    model = FakeModel([0.9, 0.05, 0.05], labels)
    pred = predict_sign(model, labels, _frames(), threshold=0.7)
    assert pred.label == "ola"
    assert pred.status == "accepted"
    assert pred.confidence == pytest.approx(0.9)


def test_predict_sign_marks_low_confidence_as_uncertain():
    labels = ["ola", "tchau", "_outro"]
    model = FakeModel([0.5, 0.45, 0.05], labels)
    pred = predict_sign(model, labels, _frames(), threshold=0.7)
    assert pred.status == "uncertain"


def test_predict_sign_marks_negative_class_as_unknown():
    labels = ["ola", "tchau", "_outro"]
    model = FakeModel([0.1, 0.2, 0.7], labels)  # maior prob é a classe de rejeição
    pred = predict_sign(model, labels, _frames(), threshold=0.5)
    assert pred.status == "unknown"
