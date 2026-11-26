import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image, UnidentifiedImageError

from inference import DEFAULT_CONFIDENCE, IMAGE_SIZE, detect, load_model, read_image


class InferenceTests(unittest.TestCase):
    def test_read_image_converts_rgb(self):
        buffer = BytesIO()
        Image.new("L", (20, 30)).save(buffer, format="PNG")
        buffer.seek(0)
        self.assertEqual(read_image(buffer).mode, "RGB")

    def test_invalid_image(self):
        with self.assertRaises(UnidentifiedImageError):
            read_image(BytesIO(b"not an image"))

    def test_missing_model(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_model(Path(directory) / "missing.pt")

    def test_invalid_confidence(self):
        with self.assertRaises(ValueError):
            detect(None, None, 1.1)

    def test_model_parameters(self):
        self.assertEqual(DEFAULT_CONFIDENCE, 0.37)
        self.assertEqual(IMAGE_SIZE, 640)

    def test_detection_fields_and_rgb_conversion(self):
        box = SimpleNamespace(cls=[1], conf=[0.92], xyxy=np.array([[1, 2, 10, 20]]))
        result = Mock(boxes=[box])
        result.plot.return_value = np.array([[[0, 0, 255]]], dtype=np.uint8)
        model = Mock(names={1: "stop"})
        model.predict.return_value = [result]
        image = Image.new("RGB", (32, 32))
        annotated, detections = detect(model, image)
        model.predict.assert_called_once_with(image, imgsz=640, conf=0.37, verbose=False)
        self.assertEqual(annotated.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(detections[0].name, "stop")
        self.assertAlmostEqual(detections[0].confidence, 0.92)
        self.assertEqual(detections[0].xyxy, (1, 2, 10, 20))

    def test_empty_detections(self):
        result = Mock(boxes=[])
        result.plot.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
        model = Mock()
        model.predict.return_value = [result]
        _, detections = detect(model, Image.new("RGB", (2, 2)))
        self.assertEqual(detections, [])


if __name__ == "__main__":
    unittest.main()
