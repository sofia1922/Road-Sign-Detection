"""YOLO inference and image validation, independent of Streamlit."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps
from PIL.Image import open as open_image

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"
DEFAULT_CONFIDENCE = 0.37
IMAGE_SIZE = 640
CLASS_NAMES = {0: "trafficlight", 1: "stop", 2: "speedlimit", 3: "crosswalk"}
CLASS_LABELS = {
    "trafficlight": "Світлофор",
    "stop": "Знак STOP",
    "speedlimit": "Обмеження швидкості",
    "crosswalk": "Пішохідний перехід",
}


@dataclass(frozen=True)
class Detection:
    name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


def load_model(path=MODEL_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Модель не знайдено: {path}")
    from ultralytics import YOLO

    model = YOLO(str(path))
    if dict(model.names) != CLASS_NAMES:
        raise ValueError("Класи моделі не відповідають Road Sign Detection.")
    return model


def read_image(file):
    with open_image(file) as image:
        if image.width * image.height > 25_000_000:
            raise ValueError("Зображення завелике (максимум 25 Мп).")
        return ImageOps.exif_transpose(image).convert("RGB")


def detect(model, image, confidence=DEFAULT_CONFIDENCE):
    if not 0 <= confidence <= 1:
        raise ValueError("Поріг має бути в діапазоні від 0 до 1.")
    result = model.predict(image, imgsz=IMAGE_SIZE, conf=confidence, verbose=False)[0]
    detections = [
        Detection(
            name=model.names[int(box.cls[0])],
            confidence=float(box.conf[0]),
            xyxy=tuple(float(value) for value in box.xyxy[0].tolist()),
        )
        for box in result.boxes
    ]
    annotated = Image.fromarray(result.plot()[:, :, ::-1].copy())
    return annotated, sorted(detections, key=lambda item: item.confidence, reverse=True)
