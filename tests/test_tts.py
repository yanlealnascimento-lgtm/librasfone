import wave

import numpy as np

from tts import _trim_silence, join_phrase, read_wav


def test_read_wav_round_trips_int16(tmp_path):
    path = tmp_path / "amostra.wav"
    samples = np.array([0, 16384, -16384, 32767, -32768], dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())

    audio, rate = read_wav(path)
    assert rate == 16000
    assert audio.dtype == np.float32
    assert len(audio) == len(samples)
    assert audio[0] == 0
    assert audio[1] == np.float32(0.5)
    assert abs(audio[3] - 1.0) < 1e-3  # 32767/32768 ≈ 1


def test_join_phrase_concatenates_with_a_gap():
    rate = 16000
    clip = np.ones(1000, dtype=np.float32)
    voices = {"a": (clip, rate)}
    out = join_phrase(["a", "a"], voices, gap=0.08)
    assert out is not None
    joined, out_rate = out
    assert out_rate == rate
    gap_len = int(0.08 * rate)
    assert len(joined) == 2 * len(clip) + gap_len


def test_join_phrase_returns_none_for_unknown_words():
    voices = {"a": (np.ones(10, dtype=np.float32), 16000)}
    assert join_phrase(["desconhecida"], voices) is None


def test_trim_silence_removes_leading_and_trailing_quiet():
    audio = np.zeros(5000, dtype=np.float32)
    audio[2000:2100] = 0.8  # trecho audível no meio
    trimmed = _trim_silence(audio, threshold=0.01, pad=10)
    assert len(trimmed) < len(audio)
    assert np.max(np.abs(trimmed)) >= 0.8
