#!/usr/bin/env python3
"""D455 car approach bridge for the Unity UAV dashboard.

Mission:
  SEARCHING at high altitude
  -> lock one stable car while the aircraft remains under pilot control
  -> wait for the existing Unity CONFIRM button
  -> require the locked box near image center at CONFIRM
  -> hold the CONFIRM-time GPS column and descend to the final altitude.
"""

import argparse
import base64
import copy
import json
import math
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from std_msgs.msg import String

from unity_dashboard_bridge import (
    UnityDashboardBridge,
    load_config,
    wgs84_distance_m,
)


SEARCHING = "SEARCHING"
DESCENDING_TO_VERIFY = "DESCENDING_TO_VERIFY"
VERIFYING = "VERIFYING"
READY_FOR_UNITY_CONFIRM = "READY_FOR_UNITY_CONFIRM"
FINAL_APPROACH = "FINAL_APPROACH"
COMPLETE = "COMPLETE"
FAILED = "FAILED"
CANCELLED = "CANCELLED"
INDOOR_SAFE = "INDOOR_SAFE"
INDOOR_MAP_PREVIEW = "INDOOR_MAP_PREVIEW"


class D455CarApproachBridge(UnityDashboardBridge):
    """Mission-specific state machine layered on the proven dashboard bridge."""

    def __init__(self, config: Dict[str, Any]):
        merged = dict(config)
        merged["camera_source"] = "realsense"
        merged["use_realsense"] = True
        merged["target_color_mode"] = "any"
        merged["continuous_candidate_geolocation"] = True
        merged["candidate_id_prefix"] = str(
            merged.get("candidate_id_prefix", "car")
        )
        super().__init__(merged)

        self.approach_pub = self.create_publisher(
            String, "/unity/approach_state", 10
        )
        self.approach_state = SEARCHING
        self.approach_message = "Waiting for a stable high-altitude car detection"
        self.approach_command_id = ""
        self.coarse_target: Optional[Dict[str, Any]] = None
        self.verified_candidate: Optional[Dict[str, Any]] = None
        self.approach_started_monotonic = 0.0
        self.verification_seen_since = 0.0
        self.verification_sample_count = 0
        self._last_verification_geo_epoch = 0.0
        self._last_verification_valid_monotonic = 0.0
        self._last_approach_status_monotonic = 0.0
        self._unhealthy_since = 0.0
        self._final_target: Optional[Dict[str, Any]] = None
        self._descent_retarget_done = False
        self._latest_approach_candidates: List[Dict[str, Any]] = []
        self._latest_final_box_candidates: List[Dict[str, Any]] = []
        self._final_phase = ""
        self._final_paused_for_center = False
        self._final_last_seen_monotonic = 0.0
        self._final_last_retarget_monotonic = 0.0
        self._final_centered_since = 0.0
        self._final_completion_centered_since = 0.0
        self._final_visual_hold_sent = False
        self._final_locked_bbox: List[int] = []
        self._final_locked_track_id = -1
        self._final_locked_crop: Optional[np.ndarray] = None
        self._final_locked_crops: List[np.ndarray] = []
        self._final_last_box_seen_monotonic = 0.0
        self._final_last_velocity_command_monotonic = 0.0
        self._final_last_error_x = 0.0
        self._final_last_error_y = 0.0
        self._final_filtered_error_x = 0.0
        self._final_filtered_error_y = 0.0
        self._final_error_filter_initialized = False
        self._final_commanded_forward = 0.0
        self._final_commanded_right = 0.0
        self._staircase_steps: List[float] = []
        self._staircase_step_index = 0
        self._staircase_best_center_error = float("inf")
        self._staircase_diverging_since = 0.0
        if bool(self.config.get("enable_auto_verification", True)):
            if bool(self.config.get("direct_visual_servo_enabled", False)):
                if bool(self.config.get("staircase_visual_servo_enabled", False)):
                    self.get_logger().warning(
                        "D455 STAIRCASE VISUAL SERVO ENABLED: lock one car, "
                        "center at constant altitude, then descend one bounded "
                        "step at a time to 20 m"
                    )
                elif bool(
                    self.config.get(
                        "direct_fixed_position_descent_after_confirm", False
                    )
                ):
                    self.get_logger().warning(
                        "D455 FIXED-POSITION DESCENT ENABLED: center one car above "
                        f"{float(self.config.get('direct_visual_confirm_min_altitude_m', 80.0)):.0f} m, "
                        "press Unity CONFIRM, then hold that GPS column and descend"
                    )
                else:
                    self.get_logger().warning(
                        "D455 DIRECT VISUAL SERVO ENABLED: lock one car above "
                        f"{float(self.config.get('direct_visual_confirm_min_altitude_m', 80.0)):.0f} m, "
                        "wait for Unity CONFIRM, then center its image box while descending"
                    )
            else:
                self.get_logger().warning(
                    "D455 TWO-STAGE CAR APPROACH ENABLED: a stable car detection at high "
                    "altitude may automatically switch to GUIDED and descend to the "
                    f"{float(self.config['verification_altitude_m']):.1f} m verification altitude"
                )
        else:
            if bool(self.config.get("unity_map_preview", False)):
                self.approach_state = INDOOR_MAP_PREVIEW
                self.approach_message = (
                    "Fake GPS for Cesium map preview only; all flight commands are OFF"
                )
            else:
                self.approach_state = INDOOR_SAFE
                self.approach_message = (
                    "Indoor test only; automatic descent and all flight commands are OFF"
                )
            self.get_logger().warning(
                "D455 INDOOR SAFE MODE: automatic verification and flight control are OFF"
            )

    def _set_approach_state(self, state: str, message: str) -> None:
        changed = state != self.approach_state or message != self.approach_message
        self.approach_state = state
        self.approach_message = message
        if changed:
            self.get_logger().warning(f"Approach state {state}: {message}")

    def _candidate_record(
        self, candidate: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        track_id = int(candidate.get("track_id", -1))
        return self._candidate_cache.get(track_id) or self._candidate_archive.get(
            track_id
        )

    def _fresh_matching_candidate(
        self, candidates: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if self.coarse_target is None:
            return None, None
        now_epoch = time.time()
        max_age = float(self.config.get("verification_detection_max_age_s", 1.5))
        match_radius = float(self.config.get("verification_match_radius_m", 30.0))
        continuity_radius = float(
            self.config.get("verification_continuity_radius_m", 15.0)
        )
        max_error = float(self.config.get("verification_max_position_error_m", 12.0))
        minimum_edge_margin = float(
            self.config.get("verification_min_edge_margin_ratio", 0.05)
        )
        matches = []
        for candidate in candidates:
            record = self._candidate_record(candidate)
            if record is None:
                continue
            age = now_epoch - float(record.get("last_seen_epoch") or 0.0)
            if age > max_age:
                continue
            error = float(candidate.get("position_error_m") or 999.0)
            if error > max_error:
                continue
            if (
                float(candidate.get("bbox_edge_margin_ratio", 1.0))
                < minimum_edge_margin
            ):
                continue
            distance = wgs84_distance_m(
                float(self.coarse_target["latitude"]),
                float(self.coarse_target["longitude"]),
                float(candidate["latitude"]),
                float(candidate["longitude"]),
            )
            if distance <= match_radius:
                continuity_distance = distance
                if self.verified_candidate is not None:
                    continuity_distance = wgs84_distance_m(
                        float(self.verified_candidate["latitude"]),
                        float(self.verified_candidate["longitude"]),
                        float(candidate["latitude"]),
                        float(candidate["longitude"]),
                    )
                    if continuity_distance > continuity_radius:
                        continue
                matches.append(
                    (
                        continuity_distance,
                        distance,
                        error,
                        -float(candidate["confidence"]),
                        candidate,
                        record,
                    )
                )
        if not matches:
            return None, None
        matches.sort(key=lambda item: item[:4])
        return copy.deepcopy(matches[0][4]), matches[0][5]

    def _initial_candidate(
        self, candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        altitude = float(self._telemetry_state.get("relative_altitude") or 0.0)
        direct_visual = bool(self.config.get("direct_visual_servo_enabled", False))
        minimum = float(
            self.config.get("direct_visual_confirm_min_altitude_m", 80.0)
            if direct_visual
            else self.config.get("auto_verification_min_start_altitude_m", 70.0)
        )
        maximum = float(
            self.config.get("direct_visual_confirm_max_altitude_m", 130.0)
            if direct_visual
            else self.config.get("auto_verification_max_start_altitude_m", 130.0)
        )
        if altitude < minimum or altitude > maximum:
            return None
        min_confidence = float(
            self.config.get("auto_verification_min_confidence", 0.60)
        )
        max_error = float(
            self.config.get("auto_verification_max_initial_error_m", 25.0)
        )
        minimum_edge_margin = float(
            self.config.get("auto_verification_min_edge_margin_ratio", 0.18)
        )
        minimum_stable = float(
            self.config.get("auto_verification_min_stable_seconds", 3.0)
        )
        maximum_age = float(
            self.config.get("auto_verification_detection_max_age_s", 1.0)
        )
        eligible = [
            candidate
            for candidate in candidates
            if float(candidate.get("confidence") or 0.0) >= min_confidence
            and float(candidate.get("position_error_m") or 999.0) <= max_error
            and float(candidate.get("bbox_edge_margin_ratio", 1.0))
            >= minimum_edge_margin
            and float(candidate.get("stable_for_s") or 0.0) >= minimum_stable
            and float(candidate.get("age_s") or 999.0) <= maximum_age
        ]
        if not eligible:
            return None
        eligible.sort(
            key=lambda candidate: (
                -float(candidate.get("bbox_edge_margin_ratio") or 0.0),
                -float(candidate.get("confidence") or 0.0),
                float(candidate.get("position_error_m") or 999.0),
            )
        )
        return copy.deepcopy(eligible[0])

    def _start_auto_verification(self, candidate: Dict[str, Any]) -> None:
        if self.telemetry is None or self._active_command is not None:
            return
        command = {
            "candidate_id": candidate["candidate_id"],
            "track_id": int(candidate["track_id"]),
        }
        validated, distance, reason = self._validate_goto(
            command, int(candidate["track_id"])
        )
        if validated is None:
            self.approach_message = f"Auto verification blocked: {reason}"
            return
        mode_ok, mode_message = self.telemetry.ensure_guided_for_reposition()
        if not mode_ok:
            self._fail_approach(f"Cannot enter GUIDED for verification: {mode_message}", hold=False)
            return
        verification_altitude = float(self.config["verification_altitude_m"])
        sent, send_message = self.telemetry.send_reposition(
            float(validated["latitude"]),
            float(validated["longitude"]),
            verification_altitude,
        )
        if not sent:
            self._fail_approach(
                f"Verification reposition was not sent: {send_message}", hold=False
            )
            return
        self.approach_command_id = f"auto-verify-{uuid.uuid4().hex[:10]}"
        self.coarse_target = copy.deepcopy(validated)
        self.verified_candidate = None
        self.approach_started_monotonic = time.monotonic()
        self.verification_seen_since = 0.0
        self.verification_sample_count = 0
        self._last_verification_geo_epoch = 0.0
        self._last_verification_valid_monotonic = 0.0
        self._unhealthy_since = 0.0
        self._descent_retarget_done = False
        self._set_approach_state(
            DESCENDING_TO_VERIFY,
            f"Stable car found; flying above it and descending to {verification_altitude:.0f} m",
        )
        self._publish_status(
            self.approach_command_id,
            int(validated["track_id"]),
            "ACCEPTED",
            f"Automatic verification approach; {mode_message}; {send_message}",
            distance,
        )

    def _approach_health_ok(self) -> bool:
        blockers = self._control_blockers(self._telemetry_state)
        critical = {
            "camera_unavailable",
            "detector_not_ready",
            "mavlink_disconnected",
            "gps_invalid_or_stale",
            "aircraft_not_armed",
            "flight_mode_not_allowed",
        }
        unhealthy = [blocker for blocker in blockers if blocker in critical]
        if not unhealthy:
            self._unhealthy_since = 0.0
            return True
        if self._unhealthy_since <= 0.0:
            self._unhealthy_since = time.monotonic()
            return True
        grace = float(self.config.get("approach_health_grace_s", 3.0))
        if time.monotonic() - self._unhealthy_since <= grace:
            return True
        self._fail_approach("Safety input lost: " + ",".join(unhealthy), hold=True)
        return False

    def _fail_approach(self, message: str, hold: bool) -> None:
        was_final_approach = self.approach_state == FINAL_APPROACH
        if hold and self.telemetry is not None:
            held, hold_message = self.telemetry.send_hold_current_position()
            message += f"; {'hold sent' if held else 'hold failed'}: {hold_message}"
        track_id = int((self.verified_candidate or self.coarse_target or {}).get("track_id", -1))
        self._set_approach_state(FAILED, message)
        self._publish_status(
            self.approach_command_id, track_id, "FAILED", message, 0.0
        )
        if was_final_approach:
            self._active_command = None

    def _maybe_refine_descent_target(
        self, candidates: List[Dict[str, Any]]
    ) -> None:
        """Retarget once from a lower-altitude car fix while descending."""
        if (
            self._descent_retarget_done
            or not bool(self.config.get("verification_retarget_enabled", True))
            or self.telemetry is None
            or self.coarse_target is None
        ):
            return
        altitude = float(self._telemetry_state.get("relative_altitude") or 0.0)
        if altitude > float(
            self.config.get("verification_retarget_max_altitude_m", 70.0)
        ):
            return
        candidate, _record = self._fresh_matching_candidate(candidates)
        if candidate is None:
            return
        shift_m = wgs84_distance_m(
            float(self.coarse_target["latitude"]),
            float(self.coarse_target["longitude"]),
            float(candidate["latitude"]),
            float(candidate["longitude"]),
        )
        if shift_m < float(
            self.config.get("verification_retarget_min_shift_m", 2.0)
        ):
            self._descent_retarget_done = True
            return
        sent, message = self.telemetry.send_reposition(
            float(candidate["latitude"]),
            float(candidate["longitude"]),
            float(self.config["verification_altitude_m"]),
        )
        if not sent:
            self.approach_message = (
                f"Lower-altitude car fix found but retarget failed: {message}"
            )
            return
        self.coarse_target = copy.deepcopy(candidate)
        self._descent_retarget_done = True
        self.approach_message = (
            f"Car reacquired during descent; refined 60 m target by {shift_m:.1f} m"
        )
        self._publish_status(
            self.approach_command_id,
            int(candidate["track_id"]),
            "EXECUTING",
            self.approach_message,
            shift_m,
        )

    def _verification_tick(self, candidates: List[Dict[str, Any]]) -> None:
        if not self._approach_health_ok():
            return
        if self.coarse_target is None:
            self._fail_approach("Internal verification target is missing", hold=True)
            return
        elapsed = time.monotonic() - self.approach_started_monotonic
        if elapsed > float(self.config.get("verification_timeout_s", 120.0)):
            self._fail_approach("Verification approach timed out", hold=True)
            return

        state = self._telemetry_state
        if state.get("latitude") is None or state.get("longitude") is None:
            return
        distance = wgs84_distance_m(
            float(state["latitude"]),
            float(state["longitude"]),
            float(self.coarse_target["latitude"]),
            float(self.coarse_target["longitude"]),
        )
        verification_altitude = float(self.config["verification_altitude_m"])
        altitude_error = abs(
            float(state.get("relative_altitude") or 0.0) - verification_altitude
        )
        arrived = (
            distance <= float(self.config.get("verification_arrival_radius_m", 8.0))
            and altitude_error
            <= float(self.config.get("verification_altitude_tolerance_m", 4.0))
        )
        now = time.monotonic()
        if now - self._last_approach_status_monotonic >= 1.0:
            self._last_approach_status_monotonic = now
            self._publish_status(
                self.approach_command_id,
                int(self.coarse_target["track_id"]),
                "EXECUTING",
                (
                    "At verification altitude; confirming car"
                    if arrived
                    else "Flying to coarse car position and verification altitude"
                ),
                distance,
            )
        if not arrived:
            self._set_approach_state(
                DESCENDING_TO_VERIFY,
                f"Approaching verification point: {distance:.1f} m, altitude error {altitude_error:.1f} m",
            )
            self.verification_seen_since = 0.0
            self.verification_sample_count = 0
            self._last_verification_valid_monotonic = 0.0
            return

        self._set_approach_state(
            VERIFYING, "At verification altitude; waiting for stable low-altitude car fixes"
        )
        candidate, record = self._fresh_matching_candidate(candidates)
        needed_s = float(self.config.get("verification_confirm_seconds", 3.0))
        needed_samples = int(self.config.get("verification_min_samples", 5))
        if candidate is None or record is None:
            miss_grace = float(self.config.get("verification_miss_grace_s", 6.0))
            miss_age = (
                now - self._last_verification_valid_monotonic
                if self._last_verification_valid_monotonic > 0.0
                else miss_grace + 1.0
            )
            if self.verification_seen_since > 0.0 and miss_age <= miss_grace:
                self.approach_message = (
                    "Car briefly lost; keeping "
                    f"{self.verification_sample_count}/{needed_samples} fixes "
                    f"for {max(0.0, miss_grace - miss_age):.1f} s"
                )
                return
            self.verification_seen_since = 0.0
            self.verification_sample_count = 0
            self._last_verification_geo_epoch = 0.0
            self._last_verification_valid_monotonic = 0.0
            self.verified_candidate = None
            self.approach_message = (
                "At verification point; waiting for a nearby car with "
                f"position error <= {float(self.config.get('verification_max_position_error_m', 12.0)):.0f} m"
            )
            return

        geo_epoch = float(record.get("geolocated_at_epoch") or 0.0)
        if self.verification_seen_since <= 0.0:
            self.verification_seen_since = now
            self.verification_sample_count = 0
            self._last_verification_geo_epoch = 0.0
        if geo_epoch > self._last_verification_geo_epoch:
            self._last_verification_geo_epoch = geo_epoch
            self.verification_sample_count += 1
        self._last_verification_valid_monotonic = now
        self.verified_candidate = candidate
        stable_for = now - self.verification_seen_since
        self.approach_message = (
            f"Low-altitude car confirmation {stable_for:.1f}/{needed_s:.1f} s, "
            f"{self.verification_sample_count}/{needed_samples} fixes"
        )
        if stable_for >= needed_s and self.verification_sample_count >= needed_samples:
            self._set_approach_state(
                READY_FOR_UNITY_CONFIRM,
                "Car confirmed and position refined; press CONFIRM in Unity for 20 m",
            )
            self._publish_status(
                self.approach_command_id,
                int(candidate["track_id"]),
                "ARRIVED",
                "Verification complete; waiting for Unity CONFIRM",
                distance,
            )

    @staticmethod
    def _candidate_center_error(candidate: Dict[str, Any]) -> float:
        error_x, error_y = D455CarApproachBridge._candidate_center_offsets(candidate)
        return max(abs(error_x), abs(error_y))

    @staticmethod
    def _candidate_center_offsets(candidate: Dict[str, Any]) -> Tuple[float, float]:
        """Return signed image-center error normalized by each half-frame."""
        bbox = candidate.get("bbox") or []
        width = float(candidate.get("frame_width") or 0.0)
        height = float(candidate.get("frame_height") or 0.0)
        if len(bbox) != 4 or width <= 0.0 or height <= 0.0:
            return float("inf"), float("inf")
        center_x = 0.5 * (float(bbox[0]) + float(bbox[2]))
        center_y = 0.5 * (float(bbox[1]) + float(bbox[3]))
        error_x = (center_x - 0.5 * width) / max(1.0, 0.5 * width)
        error_y = (center_y - 0.5 * height) / max(1.0, 0.5 * height)
        return error_x, error_y

    @staticmethod
    def _candidate_crop(candidate: Dict[str, Any]) -> Optional[np.ndarray]:
        encoded = str(candidate.get("image_base64") or "")
        if not encoded:
            return None
        try:
            data = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except (ValueError, TypeError):
            return None
        return image if image is not None and image.size else None

    @staticmethod
    def _crop_appearance_score(reference: np.ndarray, current: np.ndarray) -> float:
        size = (64, 64)
        reference_resized = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
        current_resized = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
        reference_gray = cv2.equalizeHist(
            cv2.cvtColor(reference_resized, cv2.COLOR_BGR2GRAY)
        )
        current_gray = cv2.equalizeHist(
            cv2.cvtColor(current_resized, cv2.COLOR_BGR2GRAY)
        )
        structure = float(
            cv2.matchTemplate(
                reference_gray, current_gray, cv2.TM_CCOEFF_NORMED
            )[0, 0]
        )
        reference_hsv = cv2.cvtColor(reference_resized, cv2.COLOR_BGR2HSV)
        current_hsv = cv2.cvtColor(current_resized, cv2.COLOR_BGR2HSV)
        reference_hist = cv2.calcHist(
            [reference_hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]
        )
        current_hist = cv2.calcHist(
            [current_hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]
        )
        cv2.normalize(reference_hist, reference_hist)
        cv2.normalize(current_hist, current_hist)
        color = float(
            cv2.compareHist(reference_hist, current_hist, cv2.HISTCMP_CORREL)
        )
        return 0.65 * structure + 0.35 * color

    @staticmethod
    def _bbox_center_distance(
        first_bbox: List[int], second_bbox: List[int]
    ) -> float:
        if len(first_bbox) != 4 or len(second_bbox) != 4:
            return float("inf")
        first_x = 0.5 * (float(first_bbox[0]) + float(first_bbox[2]))
        first_y = 0.5 * (float(first_bbox[1]) + float(first_bbox[3]))
        second_x = 0.5 * (float(second_bbox[0]) + float(second_bbox[2]))
        second_y = 0.5 * (float(second_bbox[1]) + float(second_bbox[3]))
        return math.hypot(second_x - first_x, second_y - first_y)

    def _initialize_final_visual_lock(self, candidate: Dict[str, Any]) -> None:
        # Candidate previews are intentionally cached for Unity, so the image in
        # ``candidate`` may be several seconds old.  Prefer the live crop from
        # the CONFIRM frame when the same tracked box is still present.
        frame_shape = tuple(getattr(self, "_latest_target_frame_shape", ()) or ())
        live_candidate = None
        if len(frame_shape) >= 2:
            reference_bbox = [
                int(value) for value in (candidate.get("bbox") or [])
            ]
            live_matches = []
            for detection in getattr(
                self, "_latest_confirmed_target_detections", []
            ):
                detected_at = float(detection.get("detected_at") or 0.0)
                if (
                    time.time() - detected_at
                    > float(
                        self.config.get(
                            "final_tracking_detection_max_age_s", 0.7
                        )
                    )
                ):
                    continue
                current = self._candidate_from_detection(detection, frame_shape)
                same_track = int(current.get("track_id", -1)) == int(
                    candidate.get("track_id", -2)
                )
                bbox_distance = self._bbox_center_distance(
                    reference_bbox,
                    [int(value) for value in (current.get("bbox") or [])],
                )
                live_matches.append(
                    (
                        0 if same_track else 1,
                        bbox_distance,
                        -float(current.get("confidence") or 0.0),
                        current,
                    )
                )
            if live_matches:
                live_matches.sort(key=lambda item: item[:3])
                if live_matches[0][0] == 0 or live_matches[0][1] <= float(
                    self.config.get("final_tracking_confirm_bbox_gate_px", 100.0)
                ):
                    live_candidate = live_matches[0][3]
        if live_candidate is not None:
            candidate = live_candidate
        self._final_locked_bbox = [
            int(value) for value in (candidate.get("bbox") or [])
        ]
        self._final_locked_track_id = int(candidate.get("track_id", -1))
        self._final_locked_crop = self._candidate_crop(candidate)
        self._final_locked_crops = (
            [self._final_locked_crop.copy()]
            if self._final_locked_crop is not None
            else []
        )
        self._final_last_box_seen_monotonic = time.monotonic()

    def _update_final_visual_lock(self, candidate: Dict[str, Any]) -> None:
        bbox = [int(value) for value in (candidate.get("bbox") or [])]
        if len(bbox) == 4:
            self._final_locked_bbox = bbox
        self._final_locked_track_id = int(candidate.get("track_id", -1))
        current_crop = self._candidate_crop(candidate)
        if current_crop is not None:
            self._final_locked_crop = current_crop
            self._final_locked_crops.append(current_crop.copy())
            template_limit = max(
                2, int(self.config.get("final_tracking_template_bank_size", 8))
            )
            # Always retain the CONFIRM-frame template as an identity anchor,
            # plus the newest views that adapt to altitude, scale and rotation.
            if len(self._final_locked_crops) > template_limit:
                self._final_locked_crops = [
                    self._final_locked_crops[0],
                    *self._final_locked_crops[-(template_limit - 1) :],
                ]
        self._final_last_box_seen_monotonic = time.monotonic()

    def _fresh_final_candidate(
        self, candidates: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        reference = self._final_target or self.verified_candidate or self.coarse_target
        if reference is None:
            return None, None
        now_epoch = time.time()
        max_age = float(self.config.get("final_tracking_detection_max_age_s", 1.0))
        max_error = float(
            self.config.get("final_tracking_max_position_error_m", 20.0)
        )
        match_radius = float(self.config.get("final_tracking_match_radius_m", 50.0))
        lock_age = (
            time.monotonic() - self._final_last_box_seen_monotonic
            if self._final_last_box_seen_monotonic > 0.0
            else 0.0
        )
        staircase = bool(
            (self._active_command or {}).get("staircase_visual_servo")
        )
        bbox_gate = min(
            float(self.config.get("final_tracking_bbox_gate_max_px", 320.0)),
            float(self.config.get("final_tracking_bbox_gate_px", 160.0))
            + lock_age
            * float(self.config.get("final_tracking_bbox_gate_growth_px_s", 80.0)),
        )
        minimum_appearance = float(
            self.config.get("final_tracking_min_appearance_score", 0.50)
        )
        matches = []
        candidate_pool = list(candidates) + list(self._latest_final_box_candidates)
        for candidate in candidate_pool:
            is_box_candidate = bool(candidate.get("_final_box_candidate"))
            record = None if is_box_candidate else self._candidate_record(candidate)
            if is_box_candidate:
                age = now_epoch - float(
                    candidate.get("_final_detected_at_epoch") or 0.0
                )
            else:
                if record is None:
                    continue
                age = now_epoch - float(record.get("last_seen_epoch") or 0.0)
            if age > max_age:
                continue
            error = float(candidate.get("position_error_m") or 999.0)
            direct_visual = bool(
                self.config.get("direct_visual_servo_enabled", False)
            )
            if not direct_visual and error > max_error:
                continue
            center_error = self._candidate_center_error(candidate)
            if not math.isfinite(center_error):
                continue
            track_id = int(candidate.get("track_id", -1))
            same_track = track_id == self._final_locked_track_id
            bbox_distance = self._bbox_center_distance(
                self._final_locked_bbox,
                [int(value) for value in (candidate.get("bbox") or [])],
            )
            if staircase and not same_track:
                if lock_age > float(
                    self.config.get("staircase_reid_max_gap_s", 1.2)
                ):
                    continue
                if bbox_distance > float(
                    self.config.get("staircase_reid_bbox_gate_px", 60.0)
                ):
                    continue
            allowed_bbox_distance = bbox_gate * (1.5 if same_track else 1.0)
            if bbox_distance > allowed_bbox_distance:
                continue
            appearance_score = -1.0
            reference_crops = self._final_locked_crops or (
                [self._final_locked_crop]
                if self._final_locked_crop is not None
                else []
            )
            if staircase and not same_track and not reference_crops:
                # Without the CONFIRM appearance anchor, accepting a new
                # tracker ID could silently switch to another nearby car.
                continue
            if reference_crops:
                current_crop = self._candidate_crop(candidate)
                if current_crop is None:
                    if not same_track:
                        continue
                else:
                    appearance_score = max(
                        self._crop_appearance_score(reference_crop, current_crop)
                        for reference_crop in reference_crops
                    )
                    same_track_floor = float(
                        self.config.get(
                            "final_tracking_same_track_min_appearance_score",
                            0.15,
                        )
                    )
                    if staircase:
                        required_appearance = (
                            max(
                                same_track_floor,
                                float(
                                    self.config.get(
                                        "staircase_same_track_min_appearance_score",
                                        0.30,
                                    )
                                ),
                            )
                            if same_track
                            else float(
                                self.config.get(
                                    "staircase_reid_min_appearance_score", 0.78
                                )
                            )
                        )
                    else:
                        required_appearance = (
                            same_track_floor if same_track else minimum_appearance
                        )
                    if appearance_score < required_appearance:
                        continue
            distance = 0.0
            if not direct_visual:
                distance = wgs84_distance_m(
                    float(reference["latitude"]),
                    float(reference["longitude"]),
                    float(candidate["latitude"]),
                    float(candidate["longitude"]),
                )
                if distance > match_radius:
                    continue
            matches.append(
                (
                    0 if same_track else 1,
                    0 if is_box_candidate else 1,
                    bbox_distance,
                    -appearance_score,
                    distance,
                    error,
                    center_error,
                    -float(candidate.get("confidence") or 0.0),
                    candidate,
                    record,
                )
            )
        if not matches:
            return None, None
        matches.sort(key=lambda item: item[:8])
        return copy.deepcopy(matches[0][8]), matches[0][9]

    def _select_visual_detections(
        self, visual_detections: List[Dict[str, Any]], draw_all: bool
    ) -> List[Dict[str, Any]]:
        if self.approach_state not in (FINAL_APPROACH, COMPLETE):
            return super()._select_visual_detections(
                visual_detections, draw_all
            )
        locked_track_id = int(self._final_locked_track_id)
        # The detector still runs on every car so the locked vehicle can be
        # re-identified, but Unity only sees the one physical navigation target.
        locked_detections = [
            item
            for item in visual_detections
            if int(item.get("track_id", -1)) == locked_track_id
        ]
        return super()._select_visual_detections(
            locked_detections, draw_all
        )[:1]

    def _refresh_final_box_candidates(self) -> None:
        self._latest_final_box_candidates = []
        if self.approach_state != FINAL_APPROACH:
            return
        frame_shape = tuple(getattr(self, "_latest_target_frame_shape", ()) or ())
        if len(frame_shape) < 2:
            return
        now_epoch = time.time()
        max_age = float(self.config.get("final_tracking_detection_max_age_s", 0.7))
        minimum_confidence = float(
            self.config.get("final_tracking_min_box_confidence", 0.45)
        )
        for detection in getattr(
            self, "_latest_confirmed_target_detections", []
        ):
            detected_at = float(detection.get("detected_at") or 0.0)
            if now_epoch - detected_at > max_age:
                continue
            if float(detection.get("confidence") or 0.0) < minimum_confidence:
                continue
            candidate = self._candidate_from_detection(detection, frame_shape)
            candidate["_final_box_candidate"] = True
            candidate["_final_detected_at_epoch"] = detected_at
            self._latest_final_box_candidates.append(candidate)

    def _set_final_setpoint(
        self,
        candidate: Dict[str, Any],
        altitude_m: float,
        reason: str,
        force: bool = False,
    ) -> bool:
        if self.telemetry is None or self._active_command is None:
            return False
        now = time.monotonic()
        interval = float(self.config.get("final_tracking_retarget_interval_s", 0.8))
        if not force and now - self._final_last_retarget_monotonic < interval:
            return True
        shift_m = wgs84_distance_m(
            float(self._final_target["latitude"]),
            float(self._final_target["longitude"]),
            float(candidate["latitude"]),
            float(candidate["longitude"]),
        )
        altitude_change = abs(
            float(self._active_command.get("target_relative_altitude") or altitude_m)
            - altitude_m
        )
        if (
            not force
            and shift_m
            < float(self.config.get("final_tracking_min_retarget_shift_m", 0.8))
            and altitude_change < 0.5
        ):
            return True
        max_shift = float(self.config.get("final_tracking_max_retarget_shift_m", 35.0))
        if shift_m > max_shift:
            self.approach_message = (
                f"Rejected {shift_m:.1f} m visual jump; keeping current final target"
            )
            return False
        sent, message = self.telemetry.send_reposition(
            float(candidate["latitude"]),
            float(candidate["longitude"]),
            float(altitude_m),
        )
        if not sent:
            self.approach_message = f"Visual retarget failed: {message}"
            return False
        self._final_target = copy.deepcopy(candidate)
        self.verified_candidate = copy.deepcopy(candidate)
        self._active_command["track_id"] = int(candidate["track_id"])
        self._active_command["candidate_id"] = str(candidate["candidate_id"])
        self._active_command["target_latitude"] = float(candidate["latitude"])
        self._active_command["target_longitude"] = float(candidate["longitude"])
        self._active_command["target_relative_altitude"] = float(altitude_m)
        self._active_command["arrival_hits"] = 0
        self._final_last_retarget_monotonic = now
        self._final_visual_hold_sent = False
        self.approach_message = reason
        return True

    def _start_direct_visual_ready(self, candidate: Dict[str, Any]) -> None:
        """Lock one high-altitude car without issuing any flight command."""
        self.approach_command_id = f"visual-ready-{uuid.uuid4().hex[:10]}"
        self.coarse_target = copy.deepcopy(candidate)
        self.verified_candidate = copy.deepcopy(candidate)
        self._final_target = copy.deepcopy(candidate)
        self._last_verification_valid_monotonic = time.monotonic()
        self._initialize_final_visual_lock(candidate)
        altitude = float(self._telemetry_state.get("relative_altitude") or 0.0)
        staircase = bool(
            self.config.get("staircase_visual_servo_enabled", False)
        )
        fixed_descent = bool(
            self.config.get("direct_fixed_position_descent_after_confirm", False)
        )
        self._set_approach_state(
            READY_FOR_UNITY_CONFIRM,
            (
                f"One car locked at {altitude:.0f} m; press Unity CONFIRM for "
                "step-by-step centering descent"
                if staircase
                else f"One car locked at {altitude:.0f} m; center it and press Unity "
                "CONFIRM for fixed vertical descent"
                if fixed_descent
                else f"One car locked at {altitude:.0f} m; press Unity CONFIRM "
                "to start visual descent"
            ),
        )

    def _start_direct_visual_approach(
        self, command: Dict[str, Any], verified: Dict[str, Any]
    ) -> None:
        command_id = str(command.get("command_id") or "")
        track_id = int(verified.get("track_id", -1))
        self._publish_status(
            command_id,
            track_id,
            "RECEIVED",
            "Orin NX received staircase visual-servo command"
            if bool(self.config.get("staircase_visual_servo_enabled", False))
            else "Orin NX received fixed vertical-descent command"
            if bool(
                self.config.get(
                    "direct_fixed_position_descent_after_confirm", False
                )
            )
            else "Orin NX received direct visual-servo command",
            0.0,
        )
        if self.mode == "mock" or bool(self.config["simulate_flight_commands"]):
            self._publish_status(
                command_id,
                track_id,
                "REJECTED",
                "Direct visual servo requires live flight-control mode",
                0.0,
            )
            return
        staircase = bool(
            self.config.get("staircase_visual_servo_enabled", False)
        )
        fixed_descent = bool(
            self.config.get("direct_fixed_position_descent_after_confirm", False)
        ) and not staircase
        candidate, _distance, reason = self._validate_goto(
            command,
            track_id,
            require_target_position=not (fixed_descent or staircase),
        )
        if candidate is None:
            self._publish_status(command_id, track_id, "REJECTED", reason, 0.0)
            return
        assert self.telemetry is not None
        self._telemetry_state = self.telemetry.get()
        altitude = float(self._telemetry_state.get("relative_altitude") or 0.0)
        minimum = float(
            self.config.get("direct_visual_confirm_min_altitude_m", 80.0)
        )
        maximum = float(
            self.config.get("direct_visual_confirm_max_altitude_m", 130.0)
        )
        if altitude < minimum or altitude > maximum:
            self._publish_status(
                command_id,
                track_id,
                "REJECTED",
                f"CONFIRM altitude must be {minimum:.0f}-{maximum:.0f} m; current {altitude:.1f} m",
                0.0,
            )
            return
        if fixed_descent or staircase:
            candidate_record = self._candidate_record(candidate)
            last_seen_epoch = float(
                (candidate_record or {}).get("last_seen_epoch") or 0.0
            )
            detection_age = time.time() - last_seen_epoch
            maximum_age = float(
                self.config.get(
                    "direct_fixed_position_confirm_max_detection_age_s", 1.0
                )
            )
            if (
                detection_age < 0.0
                or detection_age > maximum_age
            ):
                self._publish_status(
                    command_id,
                    track_id,
                    "REJECTED",
                    "Locked car is not fresh in the live camera; reacquire before CONFIRM",
                    0.0,
                )
                return
            center_error = self._candidate_center_error(candidate)
            if not math.isfinite(center_error):
                self._publish_status(
                    command_id,
                    track_id,
                    "REJECTED",
                    "Locked car box is unavailable in the live camera",
                    0.0,
                )
                return
            if fixed_descent:
                center_limit = float(
                    self.config.get(
                        "direct_fixed_position_confirm_center_ratio", 0.10
                    )
                )
            else:
                center_limit = 1.0
                minimum_edge_margin = float(
                    self.config.get(
                        "staircase_confirm_min_edge_margin_ratio", 0.08
                    )
                )
                if (
                    float(candidate.get("bbox_edge_margin_ratio") or 0.0)
                    < minimum_edge_margin
                ):
                    self._publish_status(
                        command_id,
                        track_id,
                        "REJECTED",
                        "Locked car is too close to the image edge for safe staircase correction",
                        0.0,
                    )
                    return
            if center_error > center_limit:
                self._publish_status(
                    command_id,
                    track_id,
                    "REJECTED",
                    (
                        "Move aircraft until the locked car is near the camera "
                        f"center before CONFIRM ({center_error:.2f}>{center_limit:.2f})"
                    ),
                    0.0,
                )
                return
        mode_ok, mode_message = self.telemetry.ensure_guided_for_reposition()
        if not mode_ok:
            self._publish_status(
                command_id, track_id, "REJECTED", mode_message, 0.0
            )
            return
        self._telemetry_state = self.telemetry.get()
        final_altitude = float(self.config.get("goto_relative_altitude_m", 20.0))
        confirm_latitude = float(self._telemetry_state["latitude"])
        confirm_longitude = float(self._telemetry_state["longitude"])
        hold_yaw_rad = float(self._telemetry_state.get("yaw") or 0.0)
        if fixed_descent:
            sent, send_message = self.telemetry.send_reposition(
                confirm_latitude,
                confirm_longitude,
                final_altitude,
            )
        elif staircase and bool(
            self.config.get("staircase_hold_heading_enabled", True)
        ):
            sent, send_message = self.telemetry.send_body_velocity(
                0.0,
                0.0,
                0.0,
                yaw_rad=hold_yaw_rad,
            )
        else:
            sent, send_message = self.telemetry.send_body_velocity(
                0.0, 0.0, 0.0
            )
        if not sent:
            self._publish_status(
                command_id, track_id, "REJECTED", send_message, 0.0
            )
            return
        now = time.monotonic()
        self.approach_command_id = command_id
        self._active_command = {
            "command_id": command_id,
            "track_id": track_id,
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "target_latitude": confirm_latitude,
            "target_longitude": confirm_longitude,
            "target_relative_altitude": final_altitude,
            "distance": max(0.0, altitude - final_altitude),
            "started_monotonic": now,
            "last_status_time": now,
            "arrival_hits": 0,
            "simulated": False,
            "direct_visual_servo": True,
            "fixed_position_descent": fixed_descent,
            "staircase_visual_servo": staircase,
            "hold_yaw_rad": hold_yaw_rad,
        }
        self._final_target = copy.deepcopy(verified)
        if staircase:
            fixed_handoff = bool(
                self.config.get(
                    "staircase_fixed_descent_handoff_enabled", False
                )
            )
            handoff_altitude = float(
                self.config.get("staircase_handoff_altitude_m", 60.0)
            )
            configured_steps = [
                float(value)
                for value in self.config.get(
                    "staircase_altitude_steps_m",
                    [100.0, 80.0, 70.0, 60.0],
                )
            ]
            self._staircase_steps = [
                value
                for value in configured_steps
                if (
                    (value >= handoff_altitude if fixed_handoff else value >= final_altitude)
                    and value < altitude - 1.0
                )
            ]
            final_staircase_altitude = (
                handoff_altitude if fixed_handoff else final_altitude
            )
            if (
                not self._staircase_steps
                or abs(
                    self._staircase_steps[-1] - final_staircase_altitude
                )
                > 0.1
            ):
                self._staircase_steps.append(final_staircase_altitude)
            self._staircase_steps = sorted(
                set(self._staircase_steps), reverse=True
            )
            self._staircase_step_index = 0
            self._staircase_best_center_error = self._candidate_center_error(
                candidate
            )
            self._staircase_diverging_since = 0.0
            self._final_phase = "STAIRCASE_CENTER"
        else:
            self._staircase_steps = []
            self._staircase_step_index = 0
            self._staircase_best_center_error = float("inf")
            self._staircase_diverging_since = 0.0
            self._final_phase = (
                "FIXED_POSITION_DESCENT" if fixed_descent else "VISUAL_SERVO"
            )
        self._final_paused_for_center = False
        self._final_last_seen_monotonic = now
        self._final_last_velocity_command_monotonic = 0.0
        self._final_filtered_error_x = 0.0
        self._final_filtered_error_y = 0.0
        self._final_error_filter_initialized = False
        self._final_commanded_forward = 0.0
        self._final_commanded_right = 0.0
        self._final_centered_since = 0.0
        self._final_completion_centered_since = 0.0
        self._final_visual_hold_sent = False
        self._initialize_final_visual_lock(verified)
        self._set_approach_state(
            FINAL_APPROACH,
            (
                "Unity confirmed; centering one locked car before each "
                f"altitude step to {final_altitude:.0f} m"
                if staircase
                else f"Unity confirmed; holding CONFIRM position and descending "
                f"vertically to {final_altitude:.0f} m; detector changes ignored"
                if fixed_descent
                else f"Unity confirmed; centering locked car while descending "
                f"to {final_altitude:.0f} m"
            ),
        )
        self._publish_status(
            command_id,
            track_id,
            "ACCEPTED",
            (
                f"Staircase visual servo started with {len(self._staircase_steps)} "
                f"altitude steps; corrected camera Y-axis mapping; {mode_message}; "
                f"{send_message}"
                if staircase
                else f"Fixed vertical descent started; one target frozen; "
                f"{mode_message}; {send_message}"
                if fixed_descent
                else f"Direct visual servo started; {mode_message}; {send_message}"
            ),
            max(0.0, altitude - final_altitude),
        )

    def _fixed_position_descent_tick(self, now: float) -> None:
        """Monitor a descent to the aircraft position captured at CONFIRM.

        No detector output is consumed here.  The flight controller holds the
        captured latitude/longitude while descending, so new cars and tracker
        ID changes cannot redirect the aircraft.
        """
        if self.telemetry is None or self._active_command is None:
            return
        state = self._telemetry_state
        if state.get("latitude") is None or state.get("longitude") is None:
            return
        command = self._active_command
        altitude = float(state.get("relative_altitude") or 0.0)
        final_altitude = float(command["target_relative_altitude"])
        altitude_error = altitude - final_altitude
        horizontal_error = wgs84_distance_m(
            float(state["latitude"]),
            float(state["longitude"]),
            float(command["target_latitude"]),
            float(command["target_longitude"]),
        )
        maximum_drift = float(
            self.config.get("direct_fixed_position_max_drift_m", 10.0)
        )
        if horizontal_error > maximum_drift:
            self._fail_approach(
                f"Fixed descent drift exceeded {maximum_drift:.1f} m "
                f"({horizontal_error:.1f} m)",
                hold=True,
            )
            return
        self.approach_message = (
            f"Fixed vertical descent {altitude:.1f}->{final_altitude:.0f} m; "
            f"CONFIRM-position offset {horizontal_error:.1f} m; "
            "other detections ignored"
        )
        self._active_command["distance"] = abs(altitude_error)
        altitude_tolerance = float(
            self.config.get("direct_visual_altitude_tolerance_m", 0.6)
        )
        arrival_radius = float(
            self.config.get("direct_fixed_position_arrival_radius_m", 3.0)
        )
        if (
            abs(altitude_error) <= altitude_tolerance
            and horizontal_error <= arrival_radius
        ):
            if self._final_completion_centered_since <= 0.0:
                self._final_completion_centered_since = now
        else:
            self._final_completion_centered_since = 0.0
        if (
            self._final_completion_centered_since > 0.0
            and now - self._final_completion_centered_since
            >= float(self.config.get("direct_visual_completion_hold_s", 2.0))
        ):
            command_id = str(command.get("command_id") or "")
            track_id = int(command.get("track_id", -1))
            self.telemetry.send_hold_current_position()
            self._active_command = None
            self._set_approach_state(
                COMPLETE,
                f"Reached fixed CONFIRM position at {altitude:.1f} m; mission complete",
            )
            self._publish_status(
                command_id,
                track_id,
                "ARRIVED",
                "Aircraft reached 20 m above the position captured at CONFIRM",
                horizontal_error,
            )

    def _start_fixed_final_descent_from_staircase(
        self, altitude: float, center_error: float
    ) -> None:
        """Freeze the centered 60 m position and finish without detections."""
        if self.telemetry is None or self._active_command is None:
            return
        state = self._telemetry_state
        if state.get("latitude") is None or state.get("longitude") is None:
            self._fail_approach(
                "Cannot start final descent without current GPS",
                hold=True,
            )
            return
        final_altitude = float(
            self.config.get("goto_relative_altitude_m", 20.0)
        )
        latitude = float(state["latitude"])
        longitude = float(state["longitude"])
        sent, message = self.telemetry.send_reposition(
            latitude,
            longitude,
            final_altitude,
        )
        if not sent:
            self._fail_approach(
                f"Fixed final descent command failed: {message}",
                hold=True,
            )
            return
        command = self._active_command
        command["target_latitude"] = latitude
        command["target_longitude"] = longitude
        command["target_relative_altitude"] = final_altitude
        command["fixed_position_descent"] = True
        command["staircase_visual_servo"] = False
        command["distance"] = max(0.0, altitude - final_altitude)
        self._final_phase = "FIXED_POSITION_DESCENT"
        self._final_completion_centered_since = 0.0
        self._final_last_velocity_command_monotonic = 0.0
        self.approach_message = (
            f"Car centered at {altitude:.1f} m (error={center_error:.2f}); "
            f"position frozen; descending directly to {final_altitude:.0f} m"
        )
        self._publish_status(
            str(command.get("command_id") or ""),
            int(command.get("track_id", -1)),
            "EXECUTING",
            f"60 m visual centering complete; detector-independent final descent: {message}",
            command["distance"],
        )

    def _send_staircase_body_velocity(
        self,
        forward_mps: float,
        right_mps: float,
        down_mps: float,
    ) -> Tuple[bool, str]:
        """Send a gentle staircase command while preserving CONFIRM heading."""
        assert self.telemetry is not None
        yaw_rad = None
        if (
            bool(self.config.get("staircase_hold_heading_enabled", True))
            and self._active_command is not None
        ):
            stored_yaw = self._active_command.get("hold_yaw_rad")
            yaw_rad = float(
                stored_yaw
                if stored_yaw is not None
                else self._telemetry_state.get("yaw") or 0.0
            )
        return self.telemetry.send_body_velocity(
            forward_mps,
            right_mps,
            down_mps,
            yaw_rad=yaw_rad,
        )

    def _limit_staircase_horizontal_slew(
        self,
        forward_mps: float,
        right_mps: float,
        now: float,
    ) -> Tuple[float, float]:
        """Limit horizontal acceleration so image shake cannot jerk the UAV."""
        interval = float(
            self.config.get("direct_visual_command_interval_s", 0.20)
        )
        elapsed = (
            now - self._final_last_velocity_command_monotonic
            if self._final_last_velocity_command_monotonic > 0.0
            else interval
        )
        elapsed = max(interval, min(elapsed, 1.0))
        maximum_delta = float(
            self.config.get(
                "staircase_horizontal_accel_limit_mps2", 0.12
            )
        ) * elapsed

        def clamp_delta(target: float, previous: float) -> float:
            return max(
                previous - maximum_delta,
                min(previous + maximum_delta, target),
            )

        limited_forward = clamp_delta(
            forward_mps, self._final_commanded_forward
        )
        limited_right = clamp_delta(
            right_mps, self._final_commanded_right
        )
        self._final_commanded_forward = limited_forward
        self._final_commanded_right = limited_right
        return limited_forward, limited_right

    def _staircase_visual_servo_tick(
        self, candidate: Dict[str, Any], now: float
    ) -> None:
        """Center one locked car, then descend through bounded altitude steps."""
        if self.telemetry is None or self._active_command is None:
            return
        interval = float(
            self.config.get("direct_visual_command_interval_s", 0.20)
        )
        if now - self._final_last_velocity_command_monotonic < interval:
            return
        raw_error_x, raw_error_y = self._candidate_center_offsets(candidate)
        if not math.isfinite(raw_error_x) or not math.isfinite(raw_error_y):
            return
        filter_alpha = max(
            0.05,
            min(
                1.0,
                float(
                    self.config.get(
                        "staircase_error_filter_alpha", 0.25
                    )
                ),
            ),
        )
        if not self._final_error_filter_initialized:
            self._final_filtered_error_x = raw_error_x
            self._final_filtered_error_y = raw_error_y
            self._final_error_filter_initialized = True
        else:
            self._final_filtered_error_x = (
                filter_alpha * raw_error_x
                + (1.0 - filter_alpha) * self._final_filtered_error_x
            )
            self._final_filtered_error_y = (
                filter_alpha * raw_error_y
                + (1.0 - filter_alpha) * self._final_filtered_error_y
            )
        error_x = self._final_filtered_error_x
        error_y = self._final_filtered_error_y
        center_error = max(abs(error_x), abs(error_y))
        self._final_last_error_x = error_x
        self._final_last_error_y = error_y
        altitude = float(self._telemetry_state.get("relative_altitude") or 0.0)
        command = self._active_command
        correction_distance = wgs84_distance_m(
            float(self._telemetry_state["latitude"]),
            float(self._telemetry_state["longitude"]),
            float(command["target_latitude"]),
            float(command["target_longitude"]),
        )
        maximum_correction = float(
            self.config.get("staircase_max_total_horizontal_correction_m", 15.0)
        )
        if correction_distance > maximum_correction:
            self._fail_approach(
                f"Staircase horizontal correction exceeded "
                f"{maximum_correction:.1f} m ({correction_distance:.1f} m)",
                hold=True,
            )
            return

        if bool(
            self.config.get(
                "staircase_fixed_descent_handoff_enabled", False
            )
        ):
            handoff_altitude = float(
                self.config.get("staircase_handoff_altitude_m", 60.0)
            )
            handoff_altitude_tolerance = float(
                self.config.get(
                    "staircase_handoff_altitude_tolerance_m", 1.0
                )
            )
            handoff_ratio = float(
                self.config.get("staircase_handoff_center_ratio", 0.20)
            )
            if (
                altitude <= handoff_altitude + handoff_altitude_tolerance
                and center_error <= handoff_ratio
            ):
                self._send_staircase_body_velocity(0.0, 0.0, 0.0)
                self._final_commanded_forward = 0.0
                self._final_commanded_right = 0.0
                self._start_fixed_final_descent_from_staircase(
                    altitude,
                    center_error,
                )
                return

        improvement_margin = float(
            self.config.get("staircase_divergence_improvement_margin", 0.02)
        )
        divergence_margin = float(
            self.config.get("staircase_divergence_error_margin", 0.20)
        )
        if center_error < self._staircase_best_center_error - improvement_margin:
            self._staircase_best_center_error = center_error
            self._staircase_diverging_since = 0.0
        elif center_error > self._staircase_best_center_error + divergence_margin:
            if self._staircase_diverging_since <= 0.0:
                self._staircase_diverging_since = now
            elif now - self._staircase_diverging_since >= float(
                self.config.get("staircase_divergence_hold_s", 2.5)
            ):
                self._send_staircase_body_velocity(0.0, 0.0, 0.0)
                self._fail_approach(
                    "Image error increased during staircase correction; "
                    "stopped before runaway",
                    hold=True,
                )
                return
        else:
            self._staircase_diverging_since = 0.0

        deadband = float(
            self.config.get("staircase_center_deadband_ratio", 0.03)
        )
        gain = float(self.config.get("staircase_horizontal_gain_mps", 1.2))
        # The failed bag showed error_y becoming more negative while the old
        # -gain*error_y command was active.  The D455 image Y axis is therefore
        # reversed relative to that old assumption.
        forward_sign = float(
            self.config.get("staircase_forward_from_image_y_sign", 1.0)
        )
        right_sign = float(
            self.config.get("staircase_right_from_image_x_sign", 1.0)
        )
        horizontal_forward = (
            forward_sign * gain * self._apply_deadband(error_y, deadband)
        )
        horizontal_right = (
            right_sign * gain * self._apply_deadband(error_x, deadband)
        )
        if altitude <= 30.0:
            speed_limit = float(
                self.config.get("staircase_max_horizontal_speed_below_30_mps", 0.20)
            )
        elif altitude <= 50.0:
            speed_limit = float(
                self.config.get("staircase_max_horizontal_speed_below_50_mps", 0.35)
            )
        else:
            speed_limit = float(
                self.config.get("staircase_max_horizontal_speed_mps", 0.50)
            )
        speed = math.hypot(horizontal_forward, horizontal_right)
        if speed > speed_limit > 0.0:
            scale = speed_limit / speed
            horizontal_forward *= scale
            horizontal_right *= scale

        center_ratio = float(
            self.config.get("staircase_center_ratio", 0.08)
        )
        pause_ratio = float(
            self.config.get("staircase_descent_pause_ratio", 0.14)
        )
        centered = center_error <= center_ratio
        if centered:
            if self._final_centered_since <= 0.0:
                self._final_centered_since = now
        else:
            self._final_centered_since = 0.0
        centered_for = (
            now - self._final_centered_since
            if self._final_centered_since > 0.0
            else 0.0
        )

        if self._staircase_step_index >= len(self._staircase_steps):
            if bool(
                self.config.get(
                    "staircase_fixed_descent_handoff_enabled", False
                )
            ):
                handoff_ratio = float(
                    self.config.get(
                        "staircase_handoff_center_ratio", 0.12
                    )
                )
                handoff_centered = center_error <= handoff_ratio
                if handoff_centered:
                    if self._final_centered_since <= 0.0:
                        self._final_centered_since = now
                else:
                    self._final_centered_since = 0.0
                handoff_centered_for = (
                    now - self._final_centered_since
                    if self._final_centered_since > 0.0
                    else 0.0
                )
                horizontal_forward, horizontal_right = (
                    self._limit_staircase_horizontal_slew(
                        horizontal_forward,
                        horizontal_right,
                        now,
                    )
                )
                sent, message = self._send_staircase_body_velocity(
                    horizontal_forward,
                    horizontal_right,
                    0.0,
                )
                self._final_last_velocity_command_monotonic = now
                if not sent:
                    self._fail_approach(
                        f"60 m handoff centering failed: {message}",
                        hold=True,
                    )
                    return
                self.approach_message = (
                    f"60 m handoff centering "
                    f"(x={error_x:+.2f}, y={error_y:+.2f}); "
                    "final descent starts when stable"
                )
                if handoff_centered_for >= float(
                    self.config.get(
                        "staircase_handoff_center_hold_s", 0.6
                    )
                ):
                    self._send_staircase_body_velocity(0.0, 0.0, 0.0)
                    self._start_fixed_final_descent_from_staircase(
                        altitude,
                        center_error,
                    )
                return
            final_altitude = float(
                self.config.get("goto_relative_altitude_m", 20.0)
            )
            tolerance = float(
                self.config.get("direct_visual_altitude_tolerance_m", 0.6)
            )
            vertical_down = 0.0
            if altitude < final_altitude - tolerance:
                vertical_down = -min(0.20, 0.3 * (final_altitude - altitude))
            horizontal_forward, horizontal_right = (
                self._limit_staircase_horizontal_slew(
                    horizontal_forward,
                    horizontal_right,
                    now,
                )
            )
            sent, message = self._send_staircase_body_velocity(
                horizontal_forward,
                horizontal_right,
                vertical_down,
            )
            self._final_last_velocity_command_monotonic = now
            if not sent:
                self._fail_approach(
                    f"Staircase final-centering command failed: {message}",
                    hold=True,
                )
                return
            self.approach_message = (
                f"Final centering at {altitude:.1f} m "
                f"(x={error_x:+.2f}, y={error_y:+.2f})"
            )
            if (
                centered
                and abs(altitude - final_altitude) <= tolerance
                and centered_for
                >= float(self.config.get("staircase_final_center_hold_s", 2.0))
            ):
                command_id = str(command.get("command_id") or "")
                track_id = int(command.get("track_id", -1))
                self._send_staircase_body_velocity(0.0, 0.0, 0.0)
                self.telemetry.send_hold_current_position()
                self._active_command = None
                self._set_approach_state(
                    COMPLETE,
                    f"Locked car centered at {altitude:.1f} m; mission complete",
                )
                self._publish_status(
                    command_id,
                    track_id,
                    "ARRIVED",
                    "Staircase visual servo reached 20 m centered above one locked car",
                    center_error,
                )
            return

        target_altitude = float(
            self._staircase_steps[self._staircase_step_index]
        )
        vertical_down = 0.0
        center_hold = float(
            self.config.get("staircase_center_hold_s", 1.0)
        )
        if self._final_phase == "STAIRCASE_CENTER":
            if centered_for >= center_hold:
                self._final_phase = "STAIRCASE_DESCEND"
                self._final_centered_since = 0.0
                self._staircase_best_center_error = center_error
                self._staircase_diverging_since = 0.0
            self.approach_message = (
                f"Step {self._staircase_step_index + 1}/"
                f"{len(self._staircase_steps)}: centering at {altitude:.1f} m "
                f"(x={error_x:+.2f}, y={error_y:+.2f})"
            )
        if self._final_phase == "STAIRCASE_DESCEND":
            if center_error > pause_ratio:
                self._final_phase = "STAIRCASE_CENTER"
                self._final_centered_since = 0.0
                self._staircase_best_center_error = center_error
                self._staircase_diverging_since = 0.0
                self.approach_message = (
                    f"Step descent paused at {altitude:.1f} m; recentering "
                    f"(x={error_x:+.2f}, y={error_y:+.2f})"
                )
            else:
                if altitude > 50.0:
                    descent_speed = float(
                        self.config.get("staircase_descent_speed_high_mps", 0.45)
                    )
                elif altitude > 30.0:
                    descent_speed = float(
                        self.config.get("staircase_descent_speed_mid_mps", 0.30)
                    )
                else:
                    descent_speed = float(
                        self.config.get("staircase_descent_speed_low_mps", 0.18)
                    )
                altitude_error = altitude - target_altitude
                tolerance = float(
                    self.config.get("staircase_altitude_tolerance_m", 0.7)
                )
                if altitude_error > tolerance:
                    vertical_down = min(descent_speed, 0.30 * altitude_error)
                    horizontal_forward *= float(
                        self.config.get(
                            "staircase_descent_horizontal_scale", 0.60
                        )
                    )
                    horizontal_right *= float(
                        self.config.get(
                            "staircase_descent_horizontal_scale", 0.60
                        )
                    )
                    self.approach_message = (
                        f"Step {self._staircase_step_index + 1}/"
                        f"{len(self._staircase_steps)}: descending "
                        f"{altitude:.1f}->{target_altitude:.0f} m "
                        f"(x={error_x:+.2f}, y={error_y:+.2f})"
                    )
                else:
                    horizontal_forward = 0.0
                    horizontal_right = 0.0
                    self._staircase_step_index += 1
                    self._final_phase = "STAIRCASE_CENTER"
                    self._final_centered_since = 0.0
                    self._staircase_best_center_error = center_error
                    self._staircase_diverging_since = 0.0
                    self.approach_message = (
                        f"Altitude step reached at {altitude:.1f} m; "
                        "holding to recenter the same car"
                    )

        horizontal_forward, horizontal_right = (
            self._limit_staircase_horizontal_slew(
                horizontal_forward,
                horizontal_right,
                now,
            )
        )
        sent, message = self._send_staircase_body_velocity(
            horizontal_forward,
            horizontal_right,
            vertical_down,
        )
        self._final_last_velocity_command_monotonic = now
        if not sent:
            self._fail_approach(
                f"Staircase velocity command failed: {message}",
                hold=True,
            )
            return
        command["distance"] = abs(
            altitude
            - float(self.config.get("goto_relative_altitude_m", 20.0))
        )

    @staticmethod
    def _apply_deadband(value: float, deadband: float) -> float:
        if abs(value) <= deadband:
            return 0.0
        return math.copysign(abs(value) - deadband, value)

    def _direct_visual_servo_tick(
        self, candidate: Dict[str, Any], now: float
    ) -> None:
        if self.telemetry is None or self._active_command is None:
            return
        interval = float(
            self.config.get("direct_visual_command_interval_s", 0.20)
        )
        if now - self._final_last_velocity_command_monotonic < interval:
            return
        error_x, error_y = self._candidate_center_offsets(candidate)
        if not math.isfinite(error_x) or not math.isfinite(error_y):
            return
        self._final_last_error_x = error_x
        self._final_last_error_y = error_y
        altitude = float(self._telemetry_state.get("relative_altitude") or 0.0)
        final_altitude = float(self.config.get("goto_relative_altitude_m", 20.0))
        deadband = float(
            self.config.get("direct_visual_center_deadband_ratio", 0.025)
        )
        gain = float(
            self.config.get("direct_visual_horizontal_gain_mps", 3.0)
        )
        horizontal_forward = -gain * self._apply_deadband(error_y, deadband)
        horizontal_right = gain * self._apply_deadband(error_x, deadband)
        if altitude <= 25.0:
            speed_limit = float(
                self.config.get(
                    "direct_visual_max_horizontal_speed_below_25_mps", 0.6
                )
            )
        elif altitude <= 40.0:
            speed_limit = float(
                self.config.get(
                    "direct_visual_max_horizontal_speed_below_40_mps", 1.0
                )
            )
        else:
            speed_limit = float(
                self.config.get("direct_visual_max_horizontal_speed_mps", 2.0)
            )
        speed = math.hypot(horizontal_forward, horizontal_right)
        if speed > speed_limit > 0.0:
            scale = speed_limit / speed
            horizontal_forward *= scale
            horizontal_right *= scale

        center_error = max(abs(error_x), abs(error_y))
        descent_gate = float(
            self.config.get("direct_visual_descent_gate_ratio", 0.20)
        )
        altitude_error = altitude - final_altitude
        vertical_down = 0.0
        if center_error <= descent_gate:
            if altitude > 40.0:
                vertical_limit = float(
                    self.config.get("direct_visual_descent_speed_high_mps", 0.8)
                )
            elif altitude > 25.0:
                vertical_limit = float(
                    self.config.get("direct_visual_descent_speed_mid_mps", 0.5)
                )
            else:
                vertical_limit = float(
                    self.config.get("direct_visual_descent_speed_low_mps", 0.25)
                )
            altitude_tolerance = float(
                self.config.get("direct_visual_altitude_tolerance_m", 0.6)
            )
            if altitude_error > altitude_tolerance:
                vertical_down = min(vertical_limit, 0.35 * altitude_error)
            elif altitude_error < -altitude_tolerance:
                vertical_down = max(-0.30, 0.35 * altitude_error)

        sent, message = self.telemetry.send_body_velocity(
            horizontal_forward, horizontal_right, vertical_down
        )
        self._final_last_velocity_command_monotonic = now
        if not sent:
            self._fail_approach(f"Visual velocity command failed: {message}", hold=True)
            return
        self._active_command["track_id"] = int(candidate.get("track_id", -1))
        self._active_command["candidate_id"] = str(
            candidate.get("candidate_id") or ""
        )
        self._active_command["distance"] = abs(altitude_error)
        if center_error > descent_gate:
            self._final_paused_for_center = True
            self.approach_message = (
                f"Re-centering locked car at {altitude:.1f} m "
                f"(x={error_x:+.2f}, y={error_y:+.2f}); descent paused"
            )
        else:
            self._final_paused_for_center = False
            self.approach_message = (
                f"Locked car centered (x={error_x:+.2f}, y={error_y:+.2f}); "
                f"descending {altitude:.1f}->{final_altitude:.0f} m"
            )

        completion_ratio = float(
            self.config.get("direct_visual_completion_center_ratio", 0.06)
        )
        completion_altitude_tolerance = float(
            self.config.get("direct_visual_altitude_tolerance_m", 0.6)
        )
        if (
            center_error <= completion_ratio
            and abs(altitude_error) <= completion_altitude_tolerance
        ):
            if self._final_completion_centered_since <= 0.0:
                self._final_completion_centered_since = now
        else:
            self._final_completion_centered_since = 0.0
        if (
            self._final_completion_centered_since > 0.0
            and now - self._final_completion_centered_since
            >= float(self.config.get("direct_visual_completion_hold_s", 2.0))
        ):
            command_id = str(self._active_command.get("command_id") or "")
            track_id = int(candidate.get("track_id", -1))
            self.telemetry.send_body_velocity(0.0, 0.0, 0.0)
            self.telemetry.send_hold_current_position()
            self._active_command = None
            self._set_approach_state(
                COMPLETE,
                f"Locked car centered at {altitude:.1f} m; mission complete",
            )
            self._publish_status(
                command_id,
                track_id,
                "ARRIVED",
                "Direct visual servo reached 20 m centered above the locked car",
                0.0,
            )

    def _hold_for_lost_final_target(self, lost_age: float) -> None:
        if self.telemetry is None or self._final_visual_hold_sent:
            return
        if bool(self.config.get("direct_visual_servo_enabled", False)):
            self.telemetry.send_body_velocity(0.0, 0.0, 0.0)
        held, hold_message = self.telemetry.send_hold_current_position()
        state = self._telemetry_state
        if (
            held
            and self._active_command is not None
            and not bool(
                self._active_command.get("staircase_visual_servo")
            )
        ):
            self._active_command["target_latitude"] = float(state["latitude"])
            self._active_command["target_longitude"] = float(state["longitude"])
            self._active_command["target_relative_altitude"] = float(
                state.get("relative_altitude") or 0.0
            )
            self._active_command["arrival_hits"] = 0
        self._final_visual_hold_sent = True
        self._final_paused_for_center = True
        self._final_centered_since = 0.0
        self.approach_message = (
            f"Car lost for {lost_age:.1f} s; "
            f"{'holding position' if held else 'hold failed'}: {hold_message}"
        )

    def _final_approach_tick(self, candidates: List[Dict[str, Any]]) -> None:
        if self._active_command is None or self._final_target is None:
            return
        if not self._approach_health_ok():
            return
        state = self._telemetry_state
        if state.get("latitude") is None or state.get("longitude") is None:
            return
        now = time.monotonic()
        if (
            bool(self._active_command.get("fixed_position_descent"))
            and self._final_phase == "FIXED_POSITION_DESCENT"
        ):
            # Deliberately bypass _fresh_final_candidate(): after CONFIRM,
            # neither a new car nor a tracker-ID change may steer the aircraft.
            self._fixed_position_descent_tick(now)
            return
        candidate, _record = self._fresh_final_candidate(candidates)
        if candidate is None:
            lost_age = (
                now - self._final_last_seen_monotonic
                if self._final_last_seen_monotonic > 0.0
                else float("inf")
            )
            if (
                bool(
                    self._active_command.get(
                        "staircase_visual_servo", False
                    )
                )
                and bool(
                    self.config.get(
                        "staircase_fixed_descent_handoff_enabled", False
                    )
                )
            ):
                altitude = float(state.get("relative_altitude") or 0.0)
                handoff_altitude = float(
                    self.config.get(
                        "staircase_handoff_altitude_m", 60.0
                    )
                )
                altitude_tolerance = float(
                    self.config.get(
                        "staircase_handoff_altitude_tolerance_m", 1.0
                    )
                )
                cached_error = max(
                    abs(self._final_last_error_x),
                    abs(self._final_last_error_y),
                )
                if (
                    altitude <= handoff_altitude + altitude_tolerance
                    and cached_error
                    <= float(
                        self.config.get(
                            "staircase_handoff_center_ratio", 0.20
                        )
                    )
                    and lost_age
                    <= float(
                        self.config.get(
                            "staircase_handoff_detection_grace_s", 3.0
                        )
                    )
                ):
                    self._send_staircase_body_velocity(0.0, 0.0, 0.0)
                    self._start_fixed_final_descent_from_staircase(
                        altitude,
                        cached_error,
                    )
                    return
            if lost_age >= float(
                self.config.get("final_tracking_lost_hold_s", 1.0)
            ):
                self._hold_for_lost_final_target(lost_age)
            return

        self._final_last_seen_monotonic = now
        self._final_visual_hold_sent = False
        self._update_final_visual_lock(candidate)
        if bool(self._active_command.get("staircase_visual_servo")):
            self._staircase_visual_servo_tick(candidate, now)
            return
        if (
            bool(self.config.get("direct_visual_servo_enabled", False))
            and self._final_phase == "VISUAL_SERVO"
        ):
            self._direct_visual_servo_tick(candidate, now)
            return
        center_error = self._candidate_center_error(candidate)
        center_tolerance = float(
            self.config.get("final_tracking_center_tolerance_ratio", 0.18)
        )
        pause_offset = float(
            self.config.get("final_tracking_pause_offset_ratio", 0.45)
        )
        centered = center_error <= center_tolerance
        if centered:
            if self._final_centered_since <= 0.0:
                self._final_centered_since = now
        else:
            self._final_centered_since = 0.0
        centered_long_enough = (
            self._final_centered_since > 0.0
            and now - self._final_centered_since
            >= float(self.config.get("final_tracking_center_hold_s", 0.8))
        )

        altitude = float(state.get("relative_altitude") or 0.0)
        verification_altitude = float(self.config["verification_altitude_m"])
        intermediate_altitude = float(
            self.config.get("final_tracking_intermediate_altitude_m", 30.0)
        )
        final_altitude = float(self.config.get("goto_relative_altitude_m", 20.0))
        altitude_tolerance = float(
            self.config.get("final_tracking_altitude_tolerance_m", 2.5)
        )

        if self._final_phase == "CENTER_60":
            self._set_final_setpoint(
                candidate,
                verification_altitude,
                f"Car box visible at {verification_altitude:.0f} m; preparing tracked descent",
            )
            if (
                centered_long_enough
                and abs(altitude - verification_altitude) <= altitude_tolerance
            ):
                self._final_phase = "DESCEND_30"
                self._final_paused_for_center = False
                self._final_centered_since = 0.0
                self._set_final_setpoint(
                    candidate,
                    intermediate_altitude,
                    f"Car box visible; tracking while descending to {intermediate_altitude:.0f} m",
                    force=True,
                )
            return

        if self._final_phase == "DESCEND_30":
            was_paused = self._final_paused_for_center
            if center_error > pause_offset:
                self._final_paused_for_center = True
            if self._final_paused_for_center:
                self._set_final_setpoint(
                    candidate,
                    altitude,
                    f"Car moved off-center; pausing descent at {altitude:.0f} m to recenter",
                    force=not was_paused,
                )
                if centered_long_enough:
                    self._final_paused_for_center = False
                    self._final_centered_since = 0.0
                    self._set_final_setpoint(
                        candidate,
                        intermediate_altitude,
                        f"Car recentered; resuming descent to {intermediate_altitude:.0f} m",
                        force=True,
                    )
                return
            self._set_final_setpoint(
                candidate,
                intermediate_altitude,
                f"Tracking car while descending to {intermediate_altitude:.0f} m",
            )
            if altitude <= intermediate_altitude + altitude_tolerance:
                self._final_phase = "CENTER_30"
                self._final_centered_since = 0.0
            return

        if self._final_phase == "CENTER_30":
            self._set_final_setpoint(
                candidate,
                intermediate_altitude,
                f"Car box visible at {intermediate_altitude:.0f} m; preparing final descent",
            )
            if (
                centered_long_enough
                and abs(altitude - intermediate_altitude) <= altitude_tolerance
            ):
                self._final_phase = "DESCEND_20"
                self._final_paused_for_center = False
                self._final_centered_since = 0.0
                self._set_final_setpoint(
                    candidate,
                    final_altitude,
                    f"Car box visible; tracking while descending to {final_altitude:.0f} m",
                    force=True,
                )
            return

        if self._final_phase == "DESCEND_20":
            was_paused = self._final_paused_for_center
            if center_error > pause_offset:
                self._final_paused_for_center = True
            if self._final_paused_for_center:
                self._set_final_setpoint(
                    candidate,
                    altitude,
                    f"Car moved off-center; pausing descent at {altitude:.0f} m to recenter",
                    force=not was_paused,
                )
                if centered_long_enough:
                    self._final_paused_for_center = False
                    self._final_centered_since = 0.0
                    self._set_final_setpoint(
                        candidate,
                        final_altitude,
                        f"Car recentered; resuming descent to {final_altitude:.0f} m",
                        force=True,
                    )
                return
            self._set_final_setpoint(
                candidate,
                final_altitude,
                f"Tracking car while descending to {final_altitude:.0f} m",
            )
            if altitude <= final_altitude + altitude_tolerance:
                self._final_phase = "CENTER_20"
                self._final_centered_since = 0.0
            return

        if self._final_phase == "CENTER_20":
            self._set_final_setpoint(
                candidate,
                final_altitude,
                f"Following visible car box at {final_altitude:.0f} m",
            )
            distance = wgs84_distance_m(
                float(state["latitude"]),
                float(state["longitude"]),
                float(candidate["latitude"]),
                float(candidate["longitude"]),
            )
            completion_tolerance = float(
                self.config.get("final_tracking_completion_center_ratio", 0.12)
            )
            if center_error <= completion_tolerance:
                if self._final_completion_centered_since <= 0.0:
                    self._final_completion_centered_since = now
            else:
                self._final_completion_centered_since = 0.0
            completion_centered_long_enough = (
                self._final_completion_centered_since > 0.0
                and now - self._final_completion_centered_since
                >= float(
                    self.config.get(
                        "final_tracking_completion_center_hold_s", 1.0
                    )
                )
            )
            if (
                completion_centered_long_enough
                and distance <= float(self.config.get("arrival_radius_m", 4.0))
                and abs(altitude - final_altitude) <= altitude_tolerance
            ):
                command_id = str(self._active_command.get("command_id") or "")
                track_id = int(candidate["track_id"])
                self._active_command = None
                self._set_approach_state(
                    COMPLETE,
                    f"Car box visible at {final_altitude:.0f} m; mission complete",
                )
                self._publish_status(
                    command_id,
                    track_id,
                    "ARRIVED",
                    "Aircraft reached 20 m while the car remained in the camera",
                    distance,
                )

    def _command_status_tick(self) -> None:
        if self.approach_state != FINAL_APPROACH:
            super()._command_status_tick()
            return
        if self._active_command is None:
            return
        command = self._active_command
        now = time.monotonic()
        if now - float(command["last_status_time"]) < 1.0:
            return
        command["last_status_time"] = now
        state = self.telemetry.get() if self.telemetry is not None else {}
        if not state.get("connected") or state.get("latitude") is None:
            self._fail_approach("Final visual approach lost telemetry", hold=True)
            self._active_command = None
            return
        timeout_s = float(self.config.get("final_tracking_timeout_s", 180.0))
        if now - float(command["started_monotonic"]) > timeout_s:
            self._fail_approach("Final visual tracking timed out", hold=True)
            self._active_command = None
            return
        if bool(command.get("direct_visual_servo")):
            distance = abs(
                float(state.get("relative_altitude") or 0.0)
                - float(command["target_relative_altitude"])
            )
        else:
            distance = wgs84_distance_m(
                float(state["latitude"]),
                float(state["longitude"]),
                float(command["target_latitude"]),
                float(command["target_longitude"]),
            )
        command["distance"] = distance
        self._publish_status(
            str(command["command_id"]),
            int(command["track_id"]),
            "EXECUTING",
            self.approach_message,
            distance,
        )

    def _after_candidates_updated(self, candidates: List[Dict[str, Any]]) -> None:
        self._latest_approach_candidates = copy.deepcopy(candidates)
        self._refresh_final_box_candidates()
        if bool(self.config.get("enable_auto_verification", True)):
            if self.approach_state == SEARCHING:
                candidate = self._initial_candidate(candidates)
                if candidate is not None:
                    if bool(
                        self.config.get("direct_visual_servo_enabled", False)
                    ):
                        self._start_direct_visual_ready(candidate)
                    else:
                        self._start_auto_verification(candidate)
            elif self.approach_state in (DESCENDING_TO_VERIFY, VERIFYING):
                if self.approach_state == DESCENDING_TO_VERIFY:
                    self._maybe_refine_descent_target(candidates)
                self._verification_tick(candidates)
            elif self.approach_state == READY_FOR_UNITY_CONFIRM:
                if bool(self.config.get("direct_visual_servo_enabled", False)):
                    candidate, _record = self._fresh_final_candidate(candidates)
                else:
                    candidate, _record = self._fresh_matching_candidate(candidates)
                if candidate is not None:
                    self.verified_candidate = candidate
                    self._last_verification_valid_monotonic = time.monotonic()
                    if bool(
                        self.config.get("direct_visual_servo_enabled", False)
                    ):
                        self._update_final_visual_lock(candidate)
                        altitude = float(
                            self._telemetry_state.get("relative_altitude") or 0.0
                        )
                        self.approach_message = (
                            f"One car locked at {altitude:.0f} m; press Unity CONFIRM"
                        )
                else:
                    ready_hold = float(
                        self.config.get("verification_ready_hold_s", 5.0)
                    )
                    ready_age = (
                        time.monotonic() - self._last_verification_valid_monotonic
                        if self._last_verification_valid_monotonic > 0.0
                        else ready_hold + 1.0
                    )
                    if ready_age <= ready_hold:
                        self.approach_message = (
                            "Car verified; press CONFIRM within "
                            f"{max(0.0, ready_hold - ready_age):.1f} s"
                        )
                    else:
                        if bool(
                            self.config.get("direct_visual_servo_enabled", False)
                        ):
                            self._reset_search(
                                "Locked car became stale before CONFIRM; searching again"
                            )
                        else:
                            self.verified_candidate = None
                            self.verification_seen_since = 0.0
                            self.verification_sample_count = 0
                            self._last_verification_geo_epoch = 0.0
                            self._last_verification_valid_monotonic = 0.0
                            self._set_approach_state(
                                VERIFYING,
                                "Verified car fix became stale; reacquiring before Unity CONFIRM",
                            )
            elif self.approach_state == FINAL_APPROACH:
                self._final_approach_tick(candidates)

        selected_track = int((self.verified_candidate or self.coarse_target or {}).get("track_id", -1))
        for candidate in candidates:
            candidate["approach_selected"] = int(candidate.get("track_id", -2)) == selected_track
            candidate["low_altitude_verified"] = (
                self.approach_state in (READY_FOR_UNITY_CONFIRM, FINAL_APPROACH, COMPLETE)
                and candidate["approach_selected"]
            )
        single_target_states = {FINAL_APPROACH, COMPLETE}
        if bool(self.config.get("direct_visual_servo_enabled", False)):
            single_target_states.add(READY_FOR_UNITY_CONFIRM)
        if self.approach_state in single_target_states:
            locked_track_id = int(self._final_locked_track_id)
            candidates[:] = [
                candidate
                for candidate in candidates
                if int(candidate.get("track_id", -1)) == locked_track_id
            ][:1]
        self._publish_approach_state()

    def _candidate_confirm_blockers(
        self, candidates: List[Dict[str, Any]]
    ) -> List[str]:
        if self.approach_state != READY_FOR_UNITY_CONFIRM:
            return ["low_altitude_car_verification_not_ready"]
        if self.verified_candidate is None:
            return ["verified_car_fix_unavailable"]
        return []

    def _approach_payload(self) -> Dict[str, Any]:
        target = self.verified_candidate or self.coarse_target or {}
        return {
            "timestamp": time.time(),
            "state": self.approach_state,
            "message": self.approach_message,
            "ready_for_unity_confirm": (
                self.approach_state == READY_FOR_UNITY_CONFIRM
                and self.verified_candidate is not None
            ),
            "selected_candidate_id": str(target.get("candidate_id") or ""),
            "selected_track_id": int(target.get("track_id", -1)),
            "target_latitude": target.get("latitude"),
            "target_longitude": target.get("longitude"),
            "target_position_error_m": target.get("position_error_m"),
            "verification_altitude_m": float(self.config["verification_altitude_m"]),
            "final_altitude_m": float(
                self.config.get("goto_relative_altitude_m", 20.0)
            ),
            "verification_sample_count": int(self.verification_sample_count),
        }

    def _detection_payload_extras(
        self, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {"approach": self._approach_payload()}

    def _telemetry_payload_extras(self) -> Dict[str, Any]:
        return {"approach": self._approach_payload()}

    def _publish_approach_state(self) -> None:
        if not hasattr(self, "approach_pub"):
            return
        msg = String()
        msg.data = json.dumps(
            self._approach_payload(), separators=(",", ":"), allow_nan=False
        )
        self._safe_publish(self.approach_pub, msg, "approach state")

    def _reset_search(self, message: str) -> None:
        self.coarse_target = None
        self.verified_candidate = None
        self._final_target = None
        self.approach_command_id = ""
        self.approach_started_monotonic = 0.0
        self.verification_seen_since = 0.0
        self.verification_sample_count = 0
        self._last_verification_geo_epoch = 0.0
        self._last_verification_valid_monotonic = 0.0
        self._descent_retarget_done = False
        self._latest_approach_candidates = []
        self._latest_final_box_candidates = []
        self._final_phase = ""
        self._final_paused_for_center = False
        self._final_last_seen_monotonic = 0.0
        self._final_last_retarget_monotonic = 0.0
        self._final_centered_since = 0.0
        self._final_completion_centered_since = 0.0
        self._final_visual_hold_sent = False
        self._final_locked_bbox = []
        self._final_locked_track_id = -1
        self._final_locked_crop = None
        self._final_locked_crops = []
        self._final_last_box_seen_monotonic = 0.0
        self._final_last_velocity_command_monotonic = 0.0
        self._final_last_error_x = 0.0
        self._final_last_error_y = 0.0
        self._set_approach_state(SEARCHING, message)

    def _command_callback(self, msg: String) -> None:
        try:
            command = json.loads(msg.data)
            if not isinstance(command, dict):
                raise ValueError("command must be an object")
        except (json.JSONDecodeError, ValueError) as exc:
            self.get_logger().error(f"Invalid target command: {exc}")
            return
        action = str(command.get("action", "")).upper()
        command_id = str(command.get("command_id") or "")

        if action in ("RESET_SEARCH", "RETRY_SEARCH"):
            if self._active_command is not None:
                self._publish_status(
                    command_id, -1, "REJECTED", "Final approach is still active", 0.0
                )
                return
            self._reset_search("Search reset by Unity")
            self._publish_status(command_id, -1, "ACCEPTED", "Car search reset", 0.0)
            return

        if action == "CANCEL" and self.approach_state in (
            DESCENDING_TO_VERIFY,
            VERIFYING,
            READY_FOR_UNITY_CONFIRM,
        ):
            held = False
            hold_message = "telemetry unavailable"
            if self.telemetry is not None:
                held, hold_message = self.telemetry.send_hold_current_position()
            self._set_approach_state(
                CANCELLED,
                f"Cancelled by Unity; {'hold sent' if held else 'hold failed'}: {hold_message}",
            )
            track_id = int((self.verified_candidate or self.coarse_target or {}).get("track_id", -1))
            self._publish_status(
                command_id or self.approach_command_id,
                track_id,
                "CANCELLED",
                self.approach_message,
                0.0,
            )
            return

        if action in ("GOTO_TARGET", "CONFIRM"):
            if (
                self.approach_state != READY_FOR_UNITY_CONFIRM
                or self.verified_candidate is None
            ):
                try:
                    track_id = int(command.get("track_id", -1))
                except (TypeError, ValueError):
                    track_id = -1
                self._publish_status(
                    command_id,
                    track_id,
                    "REJECTED",
                    "Wait for low-altitude car verification before CONFIRM",
                    0.0,
                )
                return
            verified = copy.deepcopy(self.verified_candidate)
            command["candidate_id"] = verified["candidate_id"]
            command["track_id"] = int(verified["track_id"])
            if bool(self.config.get("direct_visual_servo_enabled", False)):
                self._start_direct_visual_approach(command, verified)
                return
            command["target_latitude"] = float(verified["latitude"])
            command["target_longitude"] = float(verified["longitude"])
            command["target_relative_altitude"] = float(
                self.config["verification_altitude_m"]
            )
            forwarded = String()
            forwarded.data = json.dumps(command, separators=(",", ":"))
            super()._command_callback(forwarded)
            if (
                self._active_command is not None
                and str(self._active_command.get("command_id") or "") == command_id
            ):
                self._final_target = verified
                self._final_phase = "CENTER_60"
                self._final_paused_for_center = False
                self._final_last_seen_monotonic = time.monotonic()
                self._final_last_retarget_monotonic = 0.0
                self._final_centered_since = 0.0
                self._final_completion_centered_since = 0.0
                self._final_visual_hold_sent = False
                self._initialize_final_visual_lock(verified)
                self._set_approach_state(
                    FINAL_APPROACH,
                    f"Unity confirmed; waiting for any visible car box at "
                    f"{float(self.config['verification_altitude_m']):.0f} m",
                )
            return

        previous_state = self.approach_state
        super()._command_callback(msg)
        if action == "CANCEL" and previous_state == FINAL_APPROACH:
            self._set_approach_state(CANCELLED, "Final approach cancelled by Unity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config_d455_car_approach.yaml", help="YAML config path"
    )
    parser.add_argument("--mode", choices=("live", "mock"), help="override config mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if os.environ.get("D455_APPROACH_INDOOR_SAFE", "").strip() == "1":
        config["enable_flight_control"] = False
        config["simulate_flight_commands"] = True
        config["enable_auto_verification"] = False
    if os.environ.get("D455_APPROACH_MAP_PREVIEW", "").strip() == "1":
        config["enable_flight_control"] = False
        config["simulate_flight_commands"] = True
        config["enable_auto_verification"] = False
        config["unity_map_preview"] = True
    if args.mode:
        config["mode"] = args.mode
    rclpy.init()
    node: Optional[D455CarApproachBridge] = None
    try:
        node = D455CarApproachBridge(config)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
