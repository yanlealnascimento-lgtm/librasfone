"""Funções de apoio aos testes: constroem amostras de mãos sintéticas.

Nenhum teste depende de câmera, modelo treinado ou dependências pesadas
(mediapipe, opencv, scikit-learn). Só NumPy, para rodar rápido no CI.
"""
import numpy as np

from landmarks import MIDDLE_MCP, NUM_LANDMARKS, WRIST


def make_hand(y: float = 0.5, size: float = 0.1, x: float = 0.5) -> np.ndarray:
    """Uma mão (21, 3) com o punho em (x, y) e o resto espalhado por `size`."""
    hand = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    hand[:, 0] = x
    hand[:, 1] = y
    hand[WRIST] = [x, y, 0.0]
    hand[MIDDLE_MCP] = [x, y - size, 0.0]  # base do dedo médio, acima do punho
    # Espalha os demais pontos para a mão não ficar degenerada.
    for i in range(NUM_LANDMARKS):
        if i not in (WRIST, MIDDLE_MCP):
            hand[i] = [x + 0.01 * (i % 3), y - size * (i / NUM_LANDMARKS), 0.0]
    return hand


def make_frame(y_right: float | None = 0.5, y_left: float | None = None) -> np.ndarray:
    """Um frame (2, 21, 3): slot 0 = mão direita, slot 1 = esquerda.

    Um `y` None deixa aquela mão ausente (toda zerada).
    """
    frame = np.zeros((2, NUM_LANDMARKS, 3), dtype=np.float32)
    if y_right is not None:
        frame[0] = make_hand(y_right)
    if y_left is not None:
        frame[1] = make_hand(y_left)
    return frame


def make_sequence(length: int, y_right: float | None = 0.5, y_left: float | None = None) -> np.ndarray:
    """Uma sequência (length, 2, 21, 3) com a mesma mão em todos os frames."""
    return np.stack([make_frame(y_right, y_left) for _ in range(length)])
