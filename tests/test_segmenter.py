from helpers import make_frame

from segmenter import ABSENT_REST_LINE, SignSegmenter, calibrate_rest_line, top_of_hands


def test_top_of_hands_returns_smallest_y_of_present_hands():
    frame = make_frame(y_right=0.4, y_left=0.6)  # as duas presentes
    top = top_of_hands(frame, [True, True])
    # make_hand coloca o dedo médio em y - size (acima do punho), então o topo
    # é menor que o y do punho; o importante é que seja o menor dos dois.
    assert top < 0.4


def test_calibrate_with_visible_resting_hands_returns_line_in_range():
    frames = [(make_frame(y_right=0.85), [True, False]) for _ in range(30)]
    line, reason = calibrate_rest_line(frames)
    assert 0.2 <= line <= ABSENT_REST_LINE
    assert "câmera" in reason or "visíveis" in reason


def test_calibrate_with_no_hands_falls_back_to_absent_line():
    frames = [(make_frame(y_right=None, y_left=None), [False, False]) for _ in range(30)]
    line, reason = calibrate_rest_line(frames)
    assert line == ABSENT_REST_LINE


def test_segmenter_captures_a_sign_when_hands_rise_then_lower():
    seg = SignSegmenter(rest_line=0.5)
    result = None
    t = 0.0
    dt = 0.03
    # Mãos acima da linha (y=0.3 < 0.5) por ~0.5 s: forma um sinal.
    for _ in range(17):
        out = seg.update(make_frame(y_right=0.3), [True, False], t)
        result = out or result
        t += dt
    # Mãos abaixo da linha (y=0.8) por mais de END_TIME: encerra o sinal.
    for _ in range(12):
        out = seg.update(make_frame(y_right=0.8), [True, False], t)
        result = out or result
        t += dt
    assert result is not None, "o segmentador deveria ter fechado um sinal"
    assert len(result) >= 3


def test_segmenter_ignores_a_too_short_blip():
    seg = SignSegmenter(rest_line=0.5)
    result = None
    t = 0.0
    dt = 0.03
    for _ in range(3):  # sobe por ~0.09 s, bem abaixo de MIN_SIGN_TIME (0.3 s)
        result = seg.update(make_frame(y_right=0.3), [True, False], t) or result
        t += dt
    for _ in range(12):
        result = seg.update(make_frame(y_right=0.8), [True, False], t) or result
        t += dt
    assert result is None
