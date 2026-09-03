#!/usr/bin/env python3
"""Detect an 80 cm red marker with a white X using OpenCV.

The detector is intentionally classical CV: it is light enough for Orin,
does not need training data, and gives explainable intermediate masks.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


MARKER_SIZE_M = 0.80


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    quad: np.ndarray
    center: tuple[float, float]
    score: float
    red_ratio: float
    white_ratio: float
    x_score: float
    estimated_range_m: Optional[float]


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def order_points(points: np.ndarray) -> np.ndarray:
    points = points.reshape(4, 2).astype(np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def make_x_template(size: int = 160, thickness_ratio: float = 0.26) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    thickness = max(5, int(size * thickness_ratio))
    margin = int(size * 0.18)
    cv2.line(mask, (margin, margin), (size - margin, size - margin), 255, thickness)
    cv2.line(mask, (size - margin, margin), (margin, size - margin), 255, thickness)
    return mask


X_TEMPLATE = make_x_template()


def red_mask_bgr(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 50, 45], dtype=np.uint8)
    upper_red_1 = np.array([12, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([168, 50, 45], dtype=np.uint8)
    upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask |= cv2.inRange(hsv, lower_red_2, upper_red_2)
    b, g, r = cv2.split(frame)
    # Phone screens and auto exposure can shift saturated red toward pink or
    # orange.  Excess-red catches those cases without trusting hue alone.
    excess_red = (
        (r.astype(np.int16) - np.maximum(b, g).astype(np.int16) > 28)
        & (r > 115)
    ).astype(np.uint8) * 255
    mask |= excess_red
    open_kernel = np.ones((5, 5), np.uint8)
    close_kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    # The white X can split the red background into several islands.
    # A stronger close makes the detector reason about the full square marker.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    return mask


def white_mask_bgr(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 125], dtype=np.uint8)
    upper_white = np.array([179, 115, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -5,
    )
    # Keep only bright adaptive regions so highlights on dark windows do not
    # dominate the marker template.
    mask |= cv2.bitwise_and(adaptive, cv2.inRange(gray, 135, 255))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def square_quad_from_contour(contour: np.ndarray) -> Optional[np.ndarray]:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return order_points(approx)

    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]
    if min(width, height) <= 1:
        return None
    aspect = max(width, height) / min(width, height)
    if aspect > 1.65:
        return None
    return order_points(cv2.boxPoints(rect))


def warp_quad(frame: np.ndarray, quad: np.ndarray, size: int = 160) -> np.ndarray:
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(frame, matrix, (size, size))


def x_template_score(white_mask: np.ndarray) -> float:
    white = (white_mask > 0).astype(np.uint8)
    template = (X_TEMPLATE > 0).astype(np.uint8)
    intersection = np.logical_and(white, template).sum()
    union = np.logical_or(white, template).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def x_hough_score(white_mask: np.ndarray) -> float:
    edges = cv2.Canny(white_mask, 60, 160)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=24,
        minLineLength=28,
        maxLineGap=8,
    )
    if lines is None:
        return 0.0

    diag_a = 0.0
    diag_b = 0.0
    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(float, line)
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle > 90:
            angle = 180 - angle
        length = math.hypot(x2 - x1, y2 - y1)
        if 28 <= angle <= 62:
            diag_a += length
        elif 118 <= abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) <= 152:
            diag_b += length
        elif 28 <= 180 - abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) <= 62:
            diag_b += length

    needed = white_mask.shape[0] * 0.7
    return float(min(diag_a / needed, 1.0) * min(diag_b / needed, 1.0))


def estimate_range(marker_px: float, focal_px: Optional[float]) -> Optional[float]:
    if not focal_px or marker_px <= 1:
        return None
    return MARKER_SIZE_M * focal_px / marker_px


def detect_markers(
    frame: np.ndarray,
    *,
    min_area: int = 700,
    focal_px: Optional[float] = None,
) -> list[Detection]:
    mask = red_mask_bgr(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        quad = square_quad_from_contour(contour)
        if quad is None:
            continue

        warped = warp_quad(frame, quad)
        red = red_mask_bgr(warped)
        white = white_mask_bgr(warped)
        red_ratio = cv2.countNonZero(red) / float(red.size)
        white_ratio = cv2.countNonZero(white) / float(white.size)

        # Strong sunlight, phone screens, and D455 auto exposure can bloom the
        # white X so it occupies more of the warped marker.  Keep the lower
        # gate, but allow a larger white area and rely on the X template score
        # below to reject plain red/white clutter.
        if red_ratio < 0.18 or not 0.10 <= white_ratio <= 0.82:
            continue

        template_score = x_template_score(white)
        line_score = x_hough_score(white)
        x_score = max(template_score, line_score * 0.75 + template_score * 0.25)
        if x_score < 0.22:
            continue

        x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
        marker_px = (w + h) * 0.5
        score = 0.45 * red_ratio + 0.35 * x_score + 0.20 * min(white_ratio / 0.25, 1.0)
        detections.append(
            Detection(
                bbox=(x, y, w, h),
                quad=quad,
                center=(float(x + w / 2), float(y + h / 2)),
                score=float(score),
                red_ratio=float(red_ratio),
                white_ratio=float(white_ratio),
                x_score=float(x_score),
                estimated_range_m=estimate_range(marker_px, focal_px),
            )
        )

    detections.sort(key=lambda det: det.score, reverse=True)
    return detections


def draw_detections(frame: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    output = frame.copy()
    for index, det in enumerate(detections, start=1):
        quad = det.quad.astype(np.int32)
        cv2.polylines(output, [quad], True, (0, 255, 0), 3)
        cx, cy = map(int, det.center)
        cv2.drawMarker(output, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        label = f"red-white-X #{index} score={det.score:.2f}"
        if det.estimated_range_m is not None:
            label += f" range={det.estimated_range_m:.1f}m"
        x, y, _, _ = det.bbox
        cv2.putText(
            output,
            label,
            (max(x, 4), max(y - 8, 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return output


def detection_to_dict(det: Detection) -> dict[str, object]:
    return {
        "bbox_xywh": [round(v, 2) for v in det.bbox],
        "center_px": [round(v, 2) for v in det.center],
        "score": round(det.score, 4),
        "red_ratio": round(det.red_ratio, 4),
        "white_ratio": round(det.white_ratio, 4),
        "x_score": round(det.x_score, 4),
        "estimated_range_m": (
            None if det.estimated_range_m is None else round(det.estimated_range_m, 3)
        ),
    }


def run_image(args: argparse.Namespace) -> int:
    frame = cv2.imread(args.source)
    if frame is None:
        raise SystemExit(f"Cannot read image: {args.source}")
    detections = detect_markers(frame, min_area=args.min_area, focal_px=args.focal_px)
    output = draw_detections(frame, detections)
    if args.output:
        cv2.imwrite(args.output, output)
    if args.print_json:
        print(json.dumps([detection_to_dict(det) for det in detections], ensure_ascii=False))
    if args.show:
        cv2.imshow("red white X detector", output)
        cv2.waitKey(0)
    return 0 if detections else 2


def run_video(args: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(parse_source(args.source))
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height or 720)
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    last_print = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections = detect_markers(
                frame, min_area=args.min_area, focal_px=args.focal_px
            )
            output = draw_detections(frame, detections)
            now = time.monotonic()
            if args.print_json and now - last_print >= args.json_interval:
                print(
                    json.dumps(
                        {
                            "timestamp": time.time(),
                            "detections": [
                                detection_to_dict(det) for det in detections
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                last_print = now
            if writer is not None:
                writer.write(output)
            if args.show:
                cv2.imshow("red white X detector", output)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenCV detector for a red 80cm x 80cm marker with a white X."
    )
    parser.add_argument("--source", default="0", help="camera index, video, or image path")
    parser.add_argument("--image", action="store_true", help="treat source as one image")
    parser.add_argument("--show", action="store_true", help="show OpenCV preview window")
    parser.add_argument("--output", help="write annotated image/video")
    parser.add_argument("--print-json", action="store_true", help="print detections as JSON")
    parser.add_argument("--json-interval", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--min-area", type=int, default=700)
    parser.add_argument(
        "--focal-px",
        type=float,
        default=None,
        help="optional calibrated focal length in pixels for rough range estimate",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.image or Path(args.source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        return run_image(args)
    return run_video(args)


if __name__ == "__main__":
    raise SystemExit(main())
