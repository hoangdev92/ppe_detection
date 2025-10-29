import argparse
import os
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def load_model(weights_path: str) -> YOLO:
    """Load YOLO model from given weights path."""
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    return YOLO(weights_path)


def draw_detections(
    image_bgr: np.ndarray,
    results,
    class_names: dict,
    conf_threshold: float,
) -> np.ndarray:
    """Draw bounding boxes, labels and confidences on the image.

    Args:
        image_bgr: Input image in BGR format.
        results: Ultralytics results list.
        class_names: Mapping of class index to class name.
        conf_threshold: Minimum confidence to draw.

    Returns:
        Annotated image (BGR).
    """
    annotated = image_bgr.copy()
    if results is None:
        return annotated

    # Ultralytics returns a list-like of results
    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        for box in r.boxes:
            # xyxy is (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
            cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1

            if conf < conf_threshold:
                continue

            color = (0, 255, 0)
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.rectangle(annotated, pt1, pt2, color, 2)

            label = class_names.get(cls_id, str(cls_id))
            text = f"{label} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (pt1[0], pt1[1] - th - baseline - 4),
                (pt1[0] + tw + 4, pt1[1]),
                color,
                thickness=-1,
            )
            cv2.putText(
                annotated,
                text,
                (pt1[0] + 2, pt1[1] - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return annotated


def run_inference(
    image_path: str,
    weights_path: str,
    output_path: str,
    imgsz: int,
    conf: float,
    show: bool,
) -> Tuple[str, int, int]:
    """Run inference on a single image and save annotated output.

    Returns output path and resulting image dimensions.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = load_model(weights_path)

    # Read BGR image
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to read image. Ensure the path is correct and image is valid.")

    # Ultralytics accepts numpy arrays in BGR
    results = model.predict(image_bgr, imgsz=imgsz, conf=conf, verbose=False)

    class_names = getattr(model, "names", {}) or {}
    annotated = draw_detections(image_bgr, results, class_names, conf)

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, annotated)

    if show:
        # Convert to RGB for display window if preferred; here keep BGR for cv2.imshow
        cv2.imshow("Detections", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    h, w = annotated.shape[:2]
    return output_path, w, h


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test YOLO model on a single image and save annotated result",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image (jpg/png)",
    )
    parser.add_argument(
        "--weights",
        default=os.path.join(os.path.dirname(__file__), "..", "ppe_yolov11", "weights", "best.pt"),
        help="Path to YOLO weights (.pt). Default: ppe_yolov11/weights/best.pt",
    )
    parser.add_argument(
        "--output",
        default="output/annotated.jpg",
        help="Path to save annotated image",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show an OpenCV window with the annotated image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path, w, h = run_inference(
        image_path=args.image,
        weights_path=args.weights,
        output_path=args.output,
        imgsz=args.imgsz,
        conf=args.conf,
        show=args.show,
    )
    print(f"Saved annotated image to: {output_path} ({w}x{h})")


if __name__ == "__main__":
    main()


