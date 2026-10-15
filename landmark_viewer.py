import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

CAMERA_INDEX = 0
MAX_HANDS = 2
WINDOW_NAME = "LibrasFone - Landmarks (Q para sair)"


def ensure_model() -> Path:
    """Baixa o modelo do MediaPipe na primeira execução."""
    if not MODEL_PATH.exists():
        print(f"Baixando modelo para {MODEL_PATH} ...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = MODEL_PATH.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        tmp_path.replace(MODEL_PATH)
    return MODEL_PATH


def create_landmarker(num_hands: int = MAX_HANDS) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        # Buffer em vez de caminho: evita problemas com acentos no caminho no Windows.
        base_options=BaseOptions(model_asset_buffer=ensure_model().read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_hands(frame, result: vision.HandLandmarkerResult) -> None:
    for hand_landmarks in result.hand_landmarks:
        vision.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style(),
        )


def main() -> int:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Erro: não foi possível abrir a webcam.", file=sys.stderr)
        return 1

    start = time.monotonic()
    last_timestamp_ms = -1

    try:
        with create_landmarker() as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Erro: falha ao ler frame da webcam.", file=sys.stderr)
                    return 1

                # O modo VIDEO exige timestamps estritamente crescentes.
                timestamp_ms = max(
                    int((time.monotonic() - start) * 1000), last_timestamp_ms + 1
                )
                last_timestamp_ms = timestamp_ms

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                draw_hands(frame, result)

                # Espelha só na exibição (efeito "selfie"); o texto vem depois para não inverter.
                frame = cv2.flip(frame, 1)
                cv2.putText(
                    frame,
                    f"Maos detectadas: {len(result.hand_landmarks)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
