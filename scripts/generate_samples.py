"""Generates realistic synthetic test images and video clips for road safety evaluation."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def draw_road_scene(
    w: int = 1280,
    h: int = 720,
    lane_offset_x: int = 0,
    pedestrian_x: int = 0,
    pedestrian_y: int = 0,
    show_stop_sign: bool = False,
    lead_vehicle_y: int = 0,
) -> np.ndarray:
    """Generate a single road scene frame with asphalt, horizon, sky, and lane markings."""
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Sky (gradient blue to light cyan)
    horizon_y = int(h * 0.52)
    for y in range(horizon_y):
        ratio = y / float(horizon_y)
        b = int(180 + ratio * 40)
        g = int(120 + ratio * 60)
        r = int(70 + ratio * 80)
        img[y, :] = (b, g, r)

    # 2. Road Asphalt (dark gray)
    img[horizon_y:, :] = (45, 45, 48)

    # 3. Shoulder / Grass borders
    shoulder_left = np.array([[0, horizon_y], [int(w * 0.22), horizon_y], [0, h]], dtype=np.int32)
    shoulder_right = np.array([[w, horizon_y], [int(w * 0.78), horizon_y], [w, h]], dtype=np.int32)
    cv2.fillPoly(img, [shoulder_left], (35, 75, 40))
    cv2.fillPoly(img, [shoulder_right], (35, 75, 40))

    # 4. Painted Lane Markings
    y_top = int(h * 0.58)
    y_bot = int(h * 0.96)

    left_x_bot = int(w * 0.22) + lane_offset_x
    left_x_top = int(w * 0.44) + lane_offset_x
    right_x_bot = int(w * 0.78) + lane_offset_x
    right_x_top = int(w * 0.56) + lane_offset_x

    # Solid Left White Line
    cv2.line(img, (left_x_bot, y_bot), (left_x_top, y_top), (245, 245, 245), 14, cv2.LINE_AA)
    # Solid Right White Line
    cv2.line(img, (right_x_bot, y_bot), (right_x_top, y_top), (245, 245, 245), 14, cv2.LINE_AA)

    # Dashed Center Yellow/White Line
    for step in range(6):
        t1 = step / 6.0
        t2 = (step + 0.5) / 6.0
        c_y1 = int(y_bot - t1 * (y_bot - y_top))
        c_y2 = int(y_bot - t2 * (y_bot - y_top))
        c_x1 = int((left_x_bot + right_x_bot) / 2 - t1 * ((left_x_bot + right_x_bot) / 2 - (left_x_top + right_x_top) / 2))
        c_x2 = int((left_x_bot + right_x_bot) / 2 - t2 * ((left_x_bot + right_x_bot) / 2 - (left_x_top + right_x_top) / 2))
        cv2.line(img, (c_x1, c_y1), (c_x2, c_y2), (0, 215, 255), 8, cv2.LINE_AA)

    # 5. Optional Stop Sign
    if show_stop_sign:
        sx, sy = int(w * 0.82), int(h * 0.38)
        # Pole
        cv2.line(img, (sx, sy + 30), (sx, sy + 180), (160, 160, 160), 6)
        # Octagon
        pts = np.array([
            [sx - 20, sy - 50], [sx + 20, sy - 50],
            [sx + 50, sy - 20], [sx + 50, sy + 20],
            [sx + 20, sy + 50], [sx - 20, sy + 50],
            [sx - 50, sy + 20], [sx - 50, sy - 20],
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], (20, 20, 220))
        cv2.polylines(img, [pts], True, (255, 255, 255), 3)
        cv2.putText(img, "STOP", (sx - 36, sy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 6. Optional Pedestrian
    if pedestrian_x > 0 and pedestrian_y > 0:
        px, py = pedestrian_x, pedestrian_y
        scale = max(0.4, (py - horizon_y) / float(h - horizon_y))
        body_h = int(120 * scale)
        body_w = int(40 * scale)
        head_r = int(14 * scale)

        # Head
        cv2.circle(img, (px, py - body_h - head_r), head_r, (180, 150, 130), -1)
        # Torso
        cv2.rectangle(img, (px - body_w // 2, py - body_h), (px + body_w // 2, py - body_h // 3), (180, 60, 40), -1)
        # Legs
        cv2.line(img, (px - body_w // 4, py - body_h // 3), (px - body_w // 4, py), (30, 30, 80), max(2, int(6 * scale)))
        cv2.line(img, (px + body_w // 4, py - body_h // 3), (px + body_w // 4, py), (30, 30, 80), max(2, int(6 * scale)))

    # 7. Optional Lead Vehicle
    if lead_vehicle_y > 0:
        vy = lead_vehicle_y
        vx = int((left_x_top + right_x_top) / 2 + (vy - y_top) / float(y_bot - y_top) * ((left_x_bot + right_x_bot) / 2 - (left_x_top + right_x_top) / 2))
        v_scale = max(0.3, (vy - horizon_y) / float(h - horizon_y))
        vw = int(180 * v_scale)
        vh = int(110 * v_scale)

        # Vehicle Body
        cv2.rectangle(img, (vx - vw // 2, vy - vh), (vx + vw // 2, vy), (40, 40, 180), -1)
        # Roof / Window
        cv2.rectangle(img, (vx - vw // 3, vy - vh - int(vh * 0.4)), (vx + vw // 3, vy - int(vh * 0.8)), (120, 150, 160), -1)
        # Tail lights
        cv2.circle(img, (vx - vw // 2 + 10, vy - 20), max(3, int(8 * v_scale)), (0, 0, 255), -1)
        cv2.circle(img, (vx + vw // 2 - 10, vy - 20), max(3, int(8 * v_scale)), (0, 0, 255), -1)

    return img


def main():
    root = Path(__file__).resolve().parent.parent
    img_dir = root / "assets" / "sample_images"
    vid_dir = root / "assets" / "sample_videos"
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    print("Generating sample road scene images...")

    # Sample 1: Highway with standard lanes
    s1 = draw_road_scene(lane_offset_x=0)
    cv2.imwrite(str(img_dir / "highway_lanes.jpg"), s1)

    # Sample 2: Urban pedestrian in forward corridor
    s2 = draw_road_scene(lane_offset_x=0, pedestrian_x=640, pedestrian_y=600)
    cv2.imwrite(str(img_dir / "urban_pedestrian.jpg"), s2)

    # Sample 3: Hazard stop sign with pedestrian near roadway
    s3 = draw_road_scene(lane_offset_x=0, pedestrian_x=420, pedestrian_y=480, show_stop_sign=True, lead_vehicle_y=500)
    cv2.imwrite(str(img_dir / "hazard_stop.jpg"), s3)

    print("Generating sample driving video clip (60 frames)...")
    vid_path = vid_dir / "sample_road_driving.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid_path), fourcc, 30.0, (1280, 720))

    for frame_i in range(60):
        # Animate gentle lane oscillation and forward pedestrian
        offset = int(25 * math.sin(frame_i * 0.1))
        ped_y = int(450 + frame_i * 2.5)
        ped_x = int(620 + 20 * math.cos(frame_i * 0.05))
        f = draw_road_scene(lane_offset_x=offset, pedestrian_x=ped_x, pedestrian_y=ped_y)
        writer.write(f)

    writer.release()
    print(f"Sample assets successfully created in:\n - {img_dir}\n - {vid_dir}")


if __name__ == "__main__":
    main()
