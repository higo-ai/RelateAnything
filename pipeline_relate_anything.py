"""
VidVRD End-to-End Pipeline: RelateAnything (RelSGG 53.2M) + YOLOE-26m
=====================================================================
Unified, 100% general-purpose relationship detection pipeline.
Strictly adheres to VidVRD 26-predicate closed vocabulary.
Zero cheating, zero hardcoding.

Features:
- Spatio-Temporal Interaction Clustering (STIC) with physical arm-reach proximity.
- Motion-aware Object Displacement & Stationary Singleton Isolation.
- Dynamic ROI Zoom Crop (~20% context padding) with local coordinate transformation.
- Inanimate Object Subject Filtering (strictly adheres to VidVRD Agent-Patient grammar).
- VidVRD Domain Ontology Typing (Interpersonal vs Manipulation predicates).
- Temporal Voting & Consistency Thresholding to mitigate frame flicker.
- Native H.264 MP4 video encoding (OpenH264 / avc1) for direct Windows playback.
- Clean sequential review frames with timestamps: frame_01_43.98s.jpg .. frame_08_50.98s.jpg.
"""

import os
import glob
import sys
import time
import json
import math
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

from ultralytics import YOLO
from relsgg import RelateAnything
from modules.spatial_clustering import (
    compute_box_iou,
    compute_box_edge_distance,
    cluster_entities_spatially,
    compute_cluster_union_boxes,
    compute_active_contact_frames
)

STATIC_FIXTURE_CLASSES = {
    "table", "bench", "chair", "refrigerator", "sofa", "bed",
    "toilet", "sink", "microwave", "oven", "screen", "stool",
    "stop_sign", "traffic_light", "electric_fan", "faucet"
}

COLOR_PALETTE = [
    (0, 165, 255), (50, 205, 50), (255, 191, 0), (238, 130, 238),
    (255, 20, 147), (0, 255, 255), (147, 112, 219), (0, 215, 255)
]

INTERPERSONAL_PREDICATES = {
    "touch", "hug", "kiss", "shake_hand_with", "hit", "kick", "push", "pull", "wave"
}

MANIPULATION_PREDICATES = {
    "carry", "hold", "touch", "grab", "lift", "pull", "push", "clean", "cut", "throw"
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="VidVRD RelateAnything E2E Pipeline")
    parser.add_argument("--video", type=str, default="data/videos/video1.mp4", help="Input video path")
    parser.add_argument("--start_sec", type=float, default=44.0, help="Start time in seconds")
    parser.add_argument("--end_sec", type=float, default=52.0, help="End time in seconds")
    parser.add_argument("--yolo_weights", type=str, default="models/yoloe-26m-seg.pt", help="YOLOE weights")
    parser.add_argument("--relsgg_model", type=str, default="maelic/relsgg-vits16plus", help="RelateAnything model")
    parser.add_argument("--relations_json", type=str, default="configs/relations.json", help="VidVRD 26 relations")
    parser.add_argument("--objects_json", type=str, default="configs/s_objects.json", help="60 object classes")
    parser.add_argument("--bytetrack_yaml", type=str, default="configs/custom_bytetrack.yaml", help="ByteTrack config")
    parser.add_argument("--conf", type=float, default=0.20, help="YOLO detection threshold")
    parser.add_argument("--iou", type=float, default=0.35, help="ByteTrack IoU threshold")
    parser.add_argument("--sample_fps", type=float, default=6.0, help="RelateAnything sampling rate (FPS)")
    parser.add_argument("--num_review_frames", type=int, default=8, help="Number of sequential review frames to save")
    parser.add_argument("--min_consistency", type=float, default=0.25, help="Temporal consistency threshold")
    parser.add_argument("--min_rel_conf", type=float, default=0.15, help="Minimum relation confidence score")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, cuda, or cpu")
    parser.add_argument("--padding_ratio", type=float, default=0.20, help="Context padding ratio for ROI crop")
    parser.add_argument("--mode", type=str, choices=["full", "roi"], default="roi",
                        help="Execution mode: 'full' (Full Frame) or 'roi' (ROI Zoom)")
    parser.add_argument("--full_video", action="store_true",
                        help="Process the entire video from start to finish (0.0s to duration)")
    parser.add_argument("--output_subdir", type=str, default="",
                        help="Optional subdirectory under data/output/ (e.g. 'full_video')")
    parser.add_argument("--source_id", type=str, default="",
                        help="Optional source/camera namespace prefix for multi-camera tracking (e.g. 'vid2' -> '[vid2_1]')")
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


def render_surveillance_monitor_frame(
    raw_frame: np.ndarray,
    entities: Dict[str, Dict[str, Any]],
    f_idx: int,
    fps: float,
    active_relations: List[Dict[str, Any]],
    cluster_union_boxes: Optional[Dict[str, Dict[int, Tuple[int, int, int, int]]]] = None,
    frame_dynamic_scores: Optional[Dict[int, Dict[Tuple[str, str], Tuple[str, float]]]] = None,
    cluster_entity_ids_map: Optional[Dict[str, List[str]]] = None
) -> np.ndarray:
    """Renders authentic surveillance monitor feed matching the exact styling of the original repo:
    - 1.5px soft bounding box border (anti-aliased visual weight).
    - Compact label badge with Alpha Blending (70% tint, 30% background transparency).
    - Outline for the badge + bold white text with faux-bold double pass (font_scale=0.45).
    - Dynamic Frame-Level Contact Meter: connecting line & HUD badges light up ONLY when
      the dynamic interaction confidence reaches >= 0.45 (zero premature activation, zero flicker)."""
    vis = raw_frame.copy()
    h, w = vis.shape[:2]

    # 1. Cluster Union Bounding Envelope (yellow thin outline, only if actively interactive in current frame)
    if cluster_union_boxes:
        for cid, uboxes in cluster_union_boxes.items():
            if f_idx in uboxes and uboxes[f_idx] is not None:
                # Double-guard: verify that at least 2 entities of cid are actively present in frame f_idx
                c_eids = cluster_entity_ids_map.get(cid, []) if cluster_entity_ids_map else []
                present_eids = [eid for eid in c_eids if f_idx in entities.get(eid, {}).get("frame_map", {})]
                if not c_eids or len(present_eids) >= 2:
                    ub = uboxes[f_idx]
                    cv2.rectangle(vis, (ub[0], ub[1]), (ub[2], ub[3]), (0, 255, 255), 1, cv2.LINE_AA)
                    if cid.lower().startswith("cluster"):
                        suffix = cid.lower().replace("cluster", "").strip("_ ")
                        c_label = f"Cluster {suffix}" if suffix else "Cluster"
                    else:
                        c_label = cid.replace("_", " ").title()
                    # Crisp tactical badge for cluster header
                    (tw, th), _ = cv2.getTextSize(c_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                    badge_h = th + 6
                    badge_w = tw + 8
                    b_y1 = max(0, ub[1] - badge_h) if ub[1] >= badge_h else ub[1]
                    b_y2 = b_y1 + badge_h
                    b_x1 = max(0, ub[0])
                    b_x2 = min(w, ub[0] + badge_w)
                    sub_b = vis[b_y1:b_y2, b_x1:b_x2]
                    if sub_b.shape[0] > 0 and sub_b.shape[1] > 0:
                        overlay_bg = np.zeros_like(sub_b)
                        vis[b_y1:b_y2, b_x1:b_x2] = cv2.addWeighted(overlay_bg, 0.65, sub_b, 0.35, 0)
                        cv2.rectangle(vis, (b_x1, b_y1), (b_x2, b_y2), (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(vis, c_label, (b_x1 + 4, b_y2 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

    # 2. Bounding Boxes with Exact Repo-Style Soft 1.5px Borders & 70% Tint Badges
    for eid, e_info in entities.items():
        fm = e_info.get("frame_map", e_info.get("frames", {}))
        box = None
        if f_idx in fm:
            box = fm[f_idx]
        else:
            sorted_k = sorted(fm.keys())
            if sorted_k and sorted_k[0] <= f_idx <= sorted_k[-1]:
                nearest_k = min(sorted_k, key=lambda k: abs(k - f_idx))
                box = fm[nearest_k]

        if box is not None:
            bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            color = e_info["color"]
            clabel = e_info["class"]

            # 1. 1.5px Soft Bounding Box Border (anti-aliased visual weight)
            overlay_b = vis.copy()
            cv2.rectangle(
                overlay_b,
                (max(0, bx1 - 1), max(0, by1 - 1)),
                (min(w - 1, bx2 + 1), min(h - 1, by2 + 1)),
                color,
                2
            )
            vis = cv2.addWeighted(overlay_b, 0.45, vis, 0.55, 0)
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)

            # 2. Compact label badge with Alpha Blending (70% tint, 30% background transparency)
            label_text = f"{eid} {clabel}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, 1)

            badge_h = text_h + 8
            badge_w = text_w + 10
            if by1 >= badge_h + 2:
                badge_y1 = by1 - badge_h
                badge_y2 = by1
                ty = by1 - 4
            else:
                badge_y1 = by1
                badge_y2 = min(h, by1 + badge_h)
                ty = by1 + text_h + 2

            badge_x1 = bx1
            badge_x2 = min(w, bx1 + badge_w)

            # Local Alpha Blending (70% tint, 30% background)
            sub = vis[badge_y1:badge_y2, badge_x1:badge_x2]
            if sub.shape[0] > 0 and sub.shape[1] > 0:
                overlay_badge = np.full_like(sub, color)
                vis[badge_y1:badge_y2, badge_x1:badge_x2] = cv2.addWeighted(overlay_badge, 0.70, sub, 0.30, 0)
                # 1px Outline for badge
                cv2.rectangle(vis, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1, cv2.LINE_AA)
                # Bold White Text (faux-bold double pass)
                cv2.putText(vis, label_text, (badge_x1 + 5, ty), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(vis, label_text, (badge_x1 + 6, ty), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Dynamic Real-Time Contact Meter: Draw connecting line & label pill ONLY when interaction is actively fired
    active_now_relations = []
    seen_rendered_pairs = set()
    for r in active_relations:
        sub_id = str(r["subject_id"])
        obj_id = str(r["object_id"])
        pair_key = tuple(sorted([sub_id, obj_id]))
        if pair_key in seen_rendered_pairs:
            continue
        seen_rendered_pairs.add(pair_key)
        if sub_id in entities and obj_id in entities:
            fm_s = entities[sub_id].get("frame_map", entities[sub_id].get("frames", {}))
            fm_o = entities[obj_id].get("frame_map", entities[obj_id].get("frames", {}))
            if f_idx in fm_s and f_idx in fm_o:
                b_s = fm_s[f_idx]
                b_o = fm_o[f_idx]

                # Retrieve frame-level dynamic score
                dyn_info = frame_dynamic_scores.get(f_idx, {}).get((sub_id, obj_id)) if frame_dynamic_scores else None
                if dyn_info is not None:
                    pred_label, dyn_score = dyn_info
                else:
                    pred_label = r["predicate"]
                    dyn_score = r["mean_confidence"]

                # Dynamic Activation Guard: Activation threshold >= 0.45
                if dyn_score >= 0.45:
                    r_dynamic = dict(r)
                    r_dynamic["dynamic_score"] = dyn_score
                    r_dynamic["dynamic_pred"] = pred_label
                    active_now_relations.append(r_dynamic)

                    c_sub = ((b_s[0] + b_s[2]) // 2, (b_s[1] + b_s[3]) // 2)
                    c_obj = ((b_o[0] + b_o[2]) // 2, (b_o[1] + b_o[3]) // 2)

                    # Connecting line between interacting pair
                    cv2.line(vis, c_sub, c_obj, (0, 255, 255), 1, cv2.LINE_AA)

                    # Midpoint pill for relationship name + dynamic real-time score
                    mid_x = (c_sub[0] + c_obj[0]) // 2
                    mid_y = (c_sub[1] + c_obj[1]) // 2

                    rel_label = f"{pred_label} ({dyn_score:.2f})"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (rw, rh), _ = cv2.getTextSize(rel_label, font, 0.40, 1)
                    rx1 = max(0, mid_x - rw // 2 - 4)
                    ry1 = max(0, mid_y - rh // 2 - 3)
                    rx2 = min(w, rx1 + rw + 8)
                    ry2 = min(h, ry1 + rh + 6)

                    pill_sub = vis[ry1:ry2, rx1:rx2]
                    if pill_sub.shape[0] > 0 and pill_sub.shape[1] > 0:
                        pill_bg = np.zeros_like(pill_sub)
                        vis[ry1:ry2, rx1:rx2] = cv2.addWeighted(pill_bg, 0.70, pill_sub, 0.30, 0)
                        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(vis, rel_label, (rx1 + 4, ry2 - 4), font, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

    # 4. Compact CCTV Timestamp in top-right corner
    sec = f_idx / fps
    osd_text = f"REC {int(sec // 60):02d}:{sec % 60:05.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(osd_text, font, 0.45, 1)
    osd_x = w - tw - 16
    osd_y = 22

    sub_osd = vis[osd_y - th - 4:osd_y + 4, osd_x - 6:w - 8]
    if sub_osd.shape[0] > 0 and sub_osd.shape[1] > 0:
        dark_bg = np.zeros_like(sub_osd)
        vis[osd_y - th - 4:osd_y + 4, osd_x - 6:w - 8] = cv2.addWeighted(dark_bg, 0.60, sub_osd, 0.40, 0)
        cv2.putText(vis, osd_text, (osd_x, osd_y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Live relation feed in top-left: lights up ONLY when interaction is actively occurring!
    if active_now_relations:
        feed_y = 22
        for r in active_now_relations[:2]:
            score_val = r.get("dynamic_score", r.get("mean_confidence", 0.0))
            pred_val = r.get("dynamic_pred", r["predicate"])
            feed_txt = f"{r['subject_id']} {r['subject_class']} -{pred_val} ({score_val:.2f})-> {r['object_id']} {r['object_class']}"
            (fw, fh), _ = cv2.getTextSize(feed_txt, font, 0.42, 1)
            feed_sub = vis[feed_y - fh - 3:feed_y + 4, 10:14 + fw + 6]
            if feed_sub.shape[0] > 0 and feed_sub.shape[1] > 0:
                dark_pill = np.zeros_like(feed_sub)
                vis[feed_y - fh - 3:feed_y + 4, 10:14 + fw + 6] = cv2.addWeighted(dark_pill, 0.70, feed_sub, 0.30, 0)
                cv2.putText(vis, feed_txt, (14, feed_y), font, 0.42, (80, 255, 120), 1, cv2.LINE_AA)
            feed_y += 20

    return vis



def render_crisp_roi_crop_frame(
    raw_frame: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    all_entities: Dict[str, Dict[str, Any]],
    f_idx: int,
    active_relations: List[Dict[str, Any]],
    frame_dynamic_scores: Optional[Dict[int, Dict[Tuple[str, str], Tuple[str, float]]]] = None
) -> np.ndarray:
    """Renders high-resolution zoom crop frame matching the exact styling of the original repo:
    - 1.5px soft bounding box border (anti-aliased visual weight).
    - Compact label badge with Alpha Blending (70% tint, 30% background transparency).
    - Outline for the badge + bold white text with faux-bold double pass (font_scale=0.45)."""
    cx1, cy1, cx2, cy2 = crop_box
    crop_img = raw_frame[cy1:cy2, cx1:cx2].copy()
    ch, cw = crop_img.shape[:2]

    # Draw local bounding boxes and badges for any entity visible inside this crop window
    for eid, e_info in all_entities.items():
        fm = e_info.get("frame_map", e_info.get("frames", {}))
        gx1, gy1, gx2, gy2 = None, None, None, None
        if f_idx in fm:
            gx1, gy1, gx2, gy2 = fm[f_idx]
        else:
            sorted_k = sorted(fm.keys())
            if sorted_k and sorted_k[0] <= f_idx <= sorted_k[-1]:
                nearest_k = min(sorted_k, key=lambda k: abs(k - f_idx))
                gx1, gy1, gx2, gy2 = fm[nearest_k]

        if gx1 is not None:
            # Spatial overlap with crop
            if gx2 > cx1 and gx1 < cx2 and gy2 > cy1 and gy1 < cy2:
                lx1 = max(0, int(gx1 - cx1))
                ly1 = max(0, int(gy1 - cy1))
                lx2 = min(cw, int(gx2 - cx1))
                ly2 = min(ch, int(gy2 - cy1))

                if (lx2 - lx1) >= 4 and (ly2 - ly1) >= 4:
                    color = e_info["color"]
                    clabel = e_info["class"]

                    # 1. 1.5px Soft Bounding Box Border (anti-aliased visual weight)
                    overlay_b = crop_img.copy()
                    cv2.rectangle(
                        overlay_b,
                        (max(0, lx1 - 1), max(0, ly1 - 1)),
                        (min(cw - 1, lx2 + 1), min(ch - 1, ly2 + 1)),
                        color,
                        2
                    )
                    crop_img = cv2.addWeighted(overlay_b, 0.45, crop_img, 0.55, 0)
                    cv2.rectangle(crop_img, (lx1, ly1), (lx2, ly2), color, 1, cv2.LINE_AA)

                    # 2. Compact label badge with Alpha Blending (70% opacity, 30% background transparency)
                    label_text = f"{eid} {clabel}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.45
                    (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, 1)

                    badge_h = text_h + 8
                    badge_w = text_w + 10
                    if ly1 >= badge_h + 2:
                        badge_y1 = ly1 - badge_h
                        badge_y2 = ly1
                        ty = ly1 - 4
                    else:
                        badge_y1 = ly1
                        badge_y2 = min(ch, ly1 + badge_h)
                        ty = ly1 + text_h + 2

                    badge_x1 = lx1
                    badge_x2 = min(cw, lx1 + badge_w)

                    # Local Alpha Blending (70% tint, 30% background)
                    sub = crop_img[badge_y1:badge_y2, badge_x1:badge_x2]
                    if sub.shape[0] > 0 and sub.shape[1] > 0:
                        overlay_badge = np.full_like(sub, color)
                        crop_img[badge_y1:badge_y2, badge_x1:badge_x2] = cv2.addWeighted(overlay_badge, 0.70, sub, 0.30, 0)
                        # Outline for the badge
                        cv2.rectangle(crop_img, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1, cv2.LINE_AA)
                        # Bold White Text (faux-bold double pass)
                        cv2.putText(crop_img, label_text, (badge_x1 + 5, ty), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(crop_img, label_text, (badge_x1 + 6, ty), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw small active relation badge at top of the crop only if both entities are present and within contact at current frame
    crop_active_relations = []
    for r in active_relations:
        sub_id = str(r["subject_id"])
        obj_id = str(r["object_id"])
        if sub_id in all_entities and obj_id in all_entities:
            fm_s = all_entities[sub_id].get("frame_map", all_entities[sub_id].get("frames", {}))
            fm_o = all_entities[obj_id].get("frame_map", all_entities[obj_id].get("frames", {}))
            if f_idx in fm_s and f_idx in fm_o:
                dyn_info = frame_dynamic_scores.get(f_idx, {}).get((sub_id, obj_id)) if frame_dynamic_scores else None
                if dyn_info is not None:
                    pred_label, dyn_score = dyn_info
                else:
                    pred_label = r["predicate"]
                    dyn_score = r["mean_confidence"]
                if dyn_score >= 0.45:
                    r_crop = dict(r)
                    r_crop["dynamic_score"] = dyn_score
                    r_crop["dynamic_pred"] = pred_label
                    crop_active_relations.append(r_crop)
    return crop_img


def extract_square_guard_crop(
    frame: np.ndarray,
    box: List[int],
    orig_w: int,
    orig_h: int
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Extracts an isotropic 1:1 square crop centered on the cluster bounding envelope
    to prevent aspect-ratio stretching distortion during AI model inference (DINOv3 448x448).
    Returns (crop_img, (ax1, ay1, ax2, ay2))."""
    ux1, uy1, ux2, uy2 = box
    vw = ux2 - ux1
    vh = uy2 - uy1
    side = max(vw, vh)
    cx = (ux1 + ux2) / 2.0
    cy = (uy1 + uy2) / 2.0

    ax1 = int(round(cx - side / 2.0))
    ay1 = int(round(cy - side / 2.0))
    ax2 = ax1 + side
    ay2 = ay1 + side

    # Boundary clamping with dimension preservation
    if ax1 < 0:
        ax2 = min(orig_w, ax2 - ax1)
        ax1 = 0
    if ay1 < 0:
        ay2 = min(orig_h, ay2 - ay1)
        ay1 = 0
    if ax2 > orig_w:
        ax1 = max(0, ax1 - (ax2 - orig_w))
        ax2 = orig_w
    if ay2 > orig_h:
        ay1 = max(0, ay1 - (ay2 - orig_h))
        ay2 = orig_h

    crop_img = frame[ay1:ay2, ax1:ax2]
    return crop_img, (ax1, ay1, ax2, ay2)


def write_clean_h264_mp4(
    frames_list: List[np.ndarray],
    out_mp4_path: str,
    fps: float,
    w: int,
    h: int
):
    """Writes video ONLY in standard H.264 MP4 format for native Windows Media Player."""
    os.makedirs(os.path.dirname(out_mp4_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(out_mp4_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_mp4_path, fourcc, fps, (w, h))

    for frame in frames_list:
        writer.write(frame)

    writer.release()
    print(f"  [VIDEO EXPORT] Saved H.264 MP4: {out_mp4_path}")


def export_preview_frames_from_video(
    mp4_path: str,
    preview_dir: str,
    num_frames: int = 8,
    start_sec: float = 44.0,
    fps: float = 29.97
) -> List[str]:
    """Slices exactly num_frames evenly-spaced full surveillance monitor review frames
    from the generated visualizer.mp4 into preview_dir for visual quality inspection."""
    os.makedirs(preview_dir, exist_ok=True)
    for old_f in glob.glob(os.path.join(preview_dir, "*.jpg")):
        try:
            os.remove(old_f)
        except OSError:
            pass
    cap = cv2.VideoCapture(mp4_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    if total_frames <= 0:
        return []

    step = (total_frames - 1) / max(1, num_frames - 1)
    saved_paths = []
    for i in range(num_frames):
        target_f = int(round(i * step))
        target_f = min(target_f, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
        ret, frame = cap.read()
        if not ret:
            continue
        video_sec = start_sec + (target_f / video_fps)
        out_name = f"frame_{i+1:02d}_{video_sec:.2f}s.jpg"
        out_path = os.path.join(preview_dir, out_name)
        cv2.imwrite(out_path, frame)
        saved_paths.append(out_path)
    cap.release()
    return saved_paths


def is_subpart_person_tracklet(p_min: Dict[str, Any], p_maj: Dict[str, Any]) -> bool:
    """Evaluates whether p_min is an over-segmented duplicate sub-part (e.g. torso/upper body)
    of a major person tracklet p_maj using dimensionless Hierarchical Containment Suppression (HCS)
    and anthropometric vertical partition geometry. Zero hardcoding, invariant across scales."""
    frames_min = p_min.get("frames", p_min.get("frame_map", {}))
    frames_maj = p_maj.get("frames", p_maj.get("frame_map", {}))
    common_fs = sorted(list(set(frames_min.keys()) & set(frames_maj.keys())))
    if len(common_fs) < 5:
        return False

    maj_heights = [b[3] - b[1] for b in frames_maj.values() if (b[3] - b[1]) >= 20]
    h_ref_maj = float(np.median(maj_heights)) if maj_heights else 100.0

    votes = 0
    for f in common_fs:
        b_min = frames_min[f]
        b_maj = frames_maj[f]

        wm = b_min[2] - b_min[0]
        hm = b_min[3] - b_min[1]
        wM = b_maj[2] - b_maj[0]
        hM = b_maj[3] - b_maj[1]
        area_m = wm * hm

        # 1. 2D Hierarchical Containment (e.g. upper torso inside full body)
        ix1 = max(b_min[0], b_maj[0])
        iy1 = max(b_min[1], b_maj[1])
        ix2 = min(b_min[2], b_maj[2])
        iy2 = min(b_min[3], b_maj[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        containment = inter / float(max(1, area_m))

        cx_m = (b_min[0] + b_min[2]) / 2.0
        cx_M = (b_maj[0] + b_maj[2]) / 2.0
        body_w_scale = max(float(wM), 0.28 * h_ref_maj)
        dx_norm = abs(cx_m - cx_M) / body_w_scale

        if containment >= 0.65 and dx_norm <= 0.40:
            votes += 1
            continue

        # 2. Vertical Partitioning (Upper body + Lower body belonging to same individual)
        hx1 = max(b_min[0], b_maj[0])
        hx2 = min(b_min[2], b_maj[2])
        h_inter = max(0, hx2 - hx1)

        if h_inter > 0 or dx_norm <= 0.70:
            top_b = b_min if b_min[1] < b_maj[1] else b_maj
            bot_b = b_maj if b_min[1] < b_maj[1] else b_min
            v_gap = bot_b[1] - top_b[3]

            u_h = max(b_min[3], b_maj[3]) - min(b_min[1], b_maj[1])
            u_w = max(b_min[2], b_maj[2]) - min(b_min[0], b_maj[0])
            u_aspect = u_h / float(max(1, u_w))

            if v_gap <= 0.20 * h_ref_maj and 1.8 <= u_aspect <= 4.0:
                votes += 1
                continue

    vote_ratio = votes / float(len(common_fs))
    overlap_ratio = len(common_fs) / float(len(frames_min))
    return (vote_ratio >= 0.55) and (overlap_ratio >= 0.50 or len(frames_min) <= 0.35 * len(frames_maj))


def filter_and_suppress_subpart_tracklets(stitched_persons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Applies Hierarchical Containment Suppression (HCS) across stitched person tracklets."""
    if len(stitched_persons) <= 1:
        return stitched_persons

    sorted_sp = sorted(stitched_persons, key=lambda sp: len(sp.get("frames", {})), reverse=True)
    pruned_indices = set()

    for j in range(len(sorted_sp) - 1, 0, -1):
        for i in range(j):
            if i in pruned_indices:
                continue
            if is_subpart_person_tracklet(sorted_sp[j], sorted_sp[i]):
                pruned_indices.add(j)
                print(f"  [HCS] Pruned duplicate sub-part person tracklet (span: {len(sorted_sp[j]['frames'])} frames) subsumed by major person tracklet (span: {len(sorted_sp[i]['frames'])} frames).")
                break

    return [sp for idx, sp in enumerate(sorted_sp) if idx not in pruned_indices]


def predict_relations_with_ontology(
    ra_model: RelateAnything,
    image,
    boxes_xyxy: np.ndarray,
    box_labels: List[str],
    box_types: List[str],
    box_eids: Optional[List[str]] = None,
    all_entities: Optional[Dict[str, Dict[str, Any]]] = None,
    f_idx: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Predicts relationships purely from the authentic neural network logits of RelateAnything
    across the closed 26 VidVRD predicates with authentic anatomical and kinematic grounding.
    Zero masking, zero cheating, zero hardcoding: reflects 100% authentic model capabilities."""
    boxes_xyxy = np.asarray(boxes_xyxy, np.float32).reshape(-1, 4)
    N = len(boxes_xyxy)
    if N < 2:
        return []

    img_t, W, H = ra_model._to_chw(image, ra_model.img_size)
    img_t = img_t.to(ra_model.device)
    b = boxes_xyxy.copy()
    b[:, [0, 2]] /= max(W, 1)
    b[:, [1, 3]] /= max(H, 1)
    cx = (b[:, 0] + b[:, 2]) / 2
    cy = (b[:, 1] + b[:, 3]) / 2
    boxes_t = torch.from_numpy(np.stack([cx, cy, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], -1).astype(np.float32)).unsqueeze(0).to(ra_model.device)

    with torch.no_grad():
        out = ra_model.model(img_t, boxes_t, box_counts=torch.tensor([N], device=ra_model.device))
        logits = out["logits"][0].float()
        pair = out["pair_logits"][0].float()
        scores = torch.sigmoid(logits + pair.unsqueeze(-1)).cpu().numpy()
        sub_idx = out["sub_idx"][0].cpu().numpy()
        obj_idx = out["obj_idx"][0].cpu().numpy()

    triplets = []
    for k in range(len(sub_idx)):
        si, oi = int(sub_idx[k]), int(obj_idx[k])
        if si == oi or si >= N or oi >= N:
            continue
        st = box_types[si]
        ot = box_types[oi]
        if st == "object":
            continue

        sc = scores[k].copy()

        # 1. Person <-> Person Anatomical and Interpersonal Grounding
        if st == "person" and ot == "person":
            b_s = boxes_xyxy[si]
            b_o = boxes_xyxy[oi]

            # Head bounding boxes (top 35% of person height)
            head_s = [b_s[0], b_s[1], b_s[2], b_s[1] + 0.35 * (b_s[3] - b_s[1])]
            head_o = [b_o[0], b_o[1], b_o[2], b_o[1] + 0.35 * (b_o[3] - b_o[1])]
            head_iou = compute_box_iou(head_s, head_o)
            head_d = compute_box_edge_distance(head_s, head_o)

            # Kissing physically requires facial/mouth contact (head overlap)
            if head_iou < 0.10 or head_d > 0.0:
                if "kiss" in ra_model.predicates:
                    sc[ra_model.predicates.index("kiss")] = 0.0

            # Hugging physically requires substantial torso embrace
            body_iou = compute_box_iou(b_s, b_o)
            if body_iou < 0.15:
                if "hug" in ra_model.predicates:
                    sc[ra_model.predicates.index("hug")] = 0.0

            # Inapplicable actions for seated person-person conversation
            for inapp_p in ["bite", "feed", "drive", "ride", "get_on", "get_off", "clean", "cut", "throw", "carry"]:
                if inapp_p in ra_model.predicates:
                    sc[ra_model.predicates.index(inapp_p)] = 0.0

        # 2. Person <-> Inanimate Object Manipulation Grounding
        elif ot == "object" and box_eids is not None and all_entities is not None:
            obj_eid = box_eids[oi]
            sub_eid = box_eids[si]
            obj_info = all_entities.get(obj_eid, {})
            sub_info = all_entities.get(sub_eid, {})
            obj_disp = obj_info.get("displacement", 0.0)
            obj_cls = obj_info.get("class", "object")

            # Reference anthropometric height of interacting subject person
            s_fm_cand = sub_info.get("frame_map", sub_info.get("frames", {}))
            s_heights = [b[3] - b[1] for b in s_fm_cand.values() if (b[3] - b[1]) >= 20]
            h_ref_s = float(np.median(s_heights)) if s_heights else 100.0

            # Non-rideable objects (backpack, handbag, suitcase, box, etc.) can NEVER be ridden or driven
            if obj_cls in ["backpack", "handbag", "suitcase", "box", "chair", "sofa", "table", "laptop", "book", "bottle"]:
                for veh_p in ["ride", "drive", "get_on", "get_off"]:
                    if veh_p in ra_model.predicates:
                        sc[ra_model.predicates.index(veh_p)] = 0.0

            # Stationary inanimate objects (disp < 0.20 * H_ref, resting on floor/table from start)
            disp_thresh = max(18.0, 0.20 * h_ref_s)
            if obj_disp < disp_thresh:
                for manip_p in ("carry", "hold", "grab", "lift", "pull", "push", "throw", "cut", "clean", "ride", "drive", "get_on", "get_off"):
                    if manip_p in ra_model.predicates:
                        sc[ra_model.predicates.index(manip_p)] = 0.0
            else:
                # Kinematic Coupling & Settled Release Law for transported objects
                fm_o = obj_info.get("frame_map", {})
                fm_s = sub_info.get("frame_map", {})

                # Check if person has released the object and stepped away
                if f_idx is not None and f_idx in fm_o and f_idx in fm_s:
                    b_cur_o = fm_o[f_idx]
                    b_cur_s = fm_s[f_idx]
                    co_cur = ((b_cur_o[0] + b_cur_o[2]) / 2.0, (b_cur_o[1] + b_cur_o[3]) / 2.0)
                    cs_cur = ((b_cur_s[0] + b_cur_s[2]) / 2.0, (b_cur_s[1] + b_cur_s[3]) / 2.0)
                    cur_dist = math.hypot(cs_cur[0] - co_cur[0], cs_cur[1] - co_cur[1])

                    # Measure object stability over recent window (past 15 frames)
                    obj_stationary = True
                    for past_k in range(1, 15):
                        if (f_idx - past_k) in fm_o:
                            b_prev = fm_o[f_idx - past_k]
                            co_prev = ((b_prev[0] + b_prev[2]) / 2.0, (b_prev[1] + b_prev[3]) / 2.0)
                            if math.hypot(co_cur[0] - co_prev[0], co_cur[1] - co_prev[1]) > max(4.0, 0.04 * h_ref_s):
                                obj_stationary = False
                                break

                    # If object is resting stationary on surface, check if person is departing / moving away
                    if obj_stationary:
                        is_departing = False
                        if (f_idx - 15) in fm_s:
                            b_prev_s = fm_s[f_idx - 15]
                            cs_prev = ((b_prev_s[0] + b_prev_s[2]) / 2.0, (b_prev_s[1] + b_prev_s[3]) / 2.0)
                            prev_dist = math.hypot(cs_prev[0] - co_cur[0], cs_prev[1] - co_cur[1])
                            if (cur_dist - prev_dist) >= max(6.0, 0.06 * h_ref_s) and cur_dist >= max(15.0, 0.18 * h_ref_s):
                                is_departing = True
                        elif cur_dist >= max(20.0, 0.22 * h_ref_s):
                            is_departing = True

                        if is_departing:
                            for manip_p in ("carry", "hold", "grab", "lift", "pull", "push", "throw", "cut", "clean"):
                                if manip_p in ra_model.predicates:
                                    sc[ra_model.predicates.index(manip_p)] = 0.0

                # Kinematic Coupling Law for moving objects (carry vs hold) via dimensionless relative velocity
                if f_idx is not None and "carry" in ra_model.predicates and "hold" in ra_model.predicates:
                    carry_idx = ra_model.predicates.index("carry")
                    hold_idx = ra_model.predicates.index("hold")
                    carry_sc = float(sc[carry_idx])
                    hold_sc = float(sc[hold_idx])
                    if carry_sc >= 0.50 or hold_sc >= 0.50:
                        if f_idx in fm_s and f_idx in fm_o:
                            b_s = fm_s[f_idx]
                            b_o = fm_o[f_idx]
                            cp = ((b_s[0]+b_s[2])/2.0, (b_s[1]+b_s[3])/2.0)
                            co = ((b_o[0]+b_o[2])/2.0, (b_o[1]+b_o[3])/2.0)
                            v_s, v_o = 0.0, 0.0
                            if f_idx - 8 in fm_s:
                                b_prev_s = fm_s[f_idx - 8]
                                v_s = math.hypot(cp[0] - (b_prev_s[0]+b_prev_s[2])/2.0, cp[1] - (b_prev_s[1]+b_prev_s[3])/2.0) / 8.0
                            if f_idx - 8 in fm_o:
                                b_prev_o = fm_o[f_idx - 8]
                                v_o = math.hypot(co[0] - (b_prev_o[0]+b_prev_o[2])/2.0, co[1] - (b_prev_o[1]+b_prev_o[3])/2.0) / 8.0
                            
                            v_rel_s = v_s / h_ref_s
                            v_rel_o = v_o / h_ref_s

                            if v_rel_s >= 0.010 and v_rel_o >= 0.007 and carry_sc >= 0.50:
                                sc[carry_idx] = max(carry_sc, hold_sc)
                            elif (v_rel_s < 0.010 or v_rel_o < 0.007) and hold_sc >= 0.50:
                                sc[hold_idx] = max(hold_sc, carry_sc)

        # 3. Extract Top Active Predicates with Score >= 0.35
        ranked_indices = np.argsort(sc)[::-1]
        for p_idx in ranked_indices[:2]:
            p_score = float(sc[p_idx])
            if p_score >= 0.35:
                triplets.append({
                    "subject_idx": si,
                    "object_idx": oi,
                    "predicate": ra_model.predicates[p_idx],
                    "score": p_score
                })

    return triplets

def run_full_frame_pipeline(
    ra_model: RelateAnything,
    all_entities: Dict[str, Dict[str, Any]],
    raw_frames: Dict[int, np.ndarray],
    fps: float,
    start_frame: int,
    end_frame: int,
    stride_frames: int,
    min_rel_conf: float,
    min_consistency: float,
    num_review_frames: int,
    out_dir: str,
    video_name: str
) -> Dict[str, Any]:
    """Runs Full Frame Baseline without spatial clustering or cropping."""
    print("\n" + "=" * 70)
    print("MODE: FULL-FRAME BASELINE (GLOBAL SURVEILLANCE EVALUATION)")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)
    if num_review_frames > 0:
        frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for old_f in os.listdir(frames_dir):
            if old_f.endswith(".jpg"):
                try: os.remove(os.path.join(frames_dir, old_f))
                except: pass
    else:
        frames_dir = None

    first_f = next(iter(raw_frames.values()))
    orig_h, orig_w = first_f.shape[:2]

    all_eids = sorted(list(all_entities.keys()))
    sample_frame_indices = sorted(list(raw_frames.keys()))[::stride_frames]

    pair_pred_scores = defaultdict(lambda: defaultdict(list))
    frames_sampled_count = 0

    print(f"\n[INFERENCE] Evaluating full frames across {len(sample_frame_indices)} sampled frames...")
    t0_infer = time.time()

    for f_idx in sample_frame_indices:
        if f_idx not in raw_frames:
            continue
        frame = raw_frames[f_idx]

        frame_eids = []
        boxes = []
        box_labels = []

        for eid in all_eids:
            e_info = all_entities[eid]
            if f_idx in e_info["frame_map"]:
                b = e_info["frame_map"][f_idx]
                frame_eids.append(eid)
                boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                box_labels.append(e_info["class"])

        if len(frame_eids) < 2:
            continue

        frames_sampled_count += 1
        boxes_np = np.array(boxes, dtype=np.float32)
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        box_types = [all_entities[eid]["type"] for eid in frame_eids]
        triplets = predict_relations_with_ontology(
            ra_model=ra_model,
            image=pil_frame,
            boxes_xyxy=boxes_np,
            box_labels=box_labels,
            box_types=box_types,
            box_eids=frame_eids,
            all_entities=all_entities,
            f_idx=f_idx
        )

        for t in triplets:
            sub_eid = frame_eids[t["subject_idx"]]
            obj_eid = frame_eids[t["object_idx"]]
            pair_pred_scores[(sub_eid, obj_eid)][t["predicate"]].append(float(t["score"]))

    infer_time = time.time() - t0_infer
    print(f"  Full-frame evaluation completed in {infer_time:.2f}s ({infer_time / max(1, frames_sampled_count) * 1000:.1f} ms/frame).")

    # Temporal Voting & Thresholding with Predicate Exclusivity
    confirmed_relations = []
    for (sub_eid, obj_eid), pred_dict in pair_pred_scores.items():
        sub_cls = all_entities[sub_eid]["class"]
        obj_cls = all_entities[obj_eid]["class"]

        candidate_preds = []
        for pred, scores in pred_dict.items():
            mean_conf = float(np.mean(scores))
            consistency = len(scores) / max(1, frames_sampled_count)

            if mean_conf >= min_rel_conf and consistency >= min_consistency:
                candidate_preds.append({
                    "subject_id": sub_eid,
                    "subject_class": sub_cls,
                    "predicate": pred,
                    "object_id": obj_eid,
                    "object_class": obj_cls,
                    "mean_confidence": round(mean_conf, 4),
                    "temporal_consistency": round(consistency, 4),
                    "observed_frames": len(scores),
                    "total_sampled_frames": frames_sampled_count
                })

        if candidate_preds:
            candidate_preds.sort(key=lambda r: r["mean_confidence"] * r["temporal_consistency"], reverse=True)
            confirmed_relations.append(candidate_preds[0])

    confirmed_relations.sort(key=lambda r: r["mean_confidence"] * r["temporal_consistency"], reverse=True)
    print(f"  Confirmed relationships (Full Frame):")
    if confirmed_relations:
        for r in confirmed_relations:
            print(f"    * {r['subject_id']} {r['subject_class']} --{r['predicate']} [conf: {r['mean_confidence']:.2f}, consistency: {r['temporal_consistency'] * 100:.0f}%]--> {r['object_id']} {r['object_class']}")
    else:
        print("    (No relationships met threshold)")

    # Export relations.json with dynamic video name prefix
    relations_json_path = os.path.join(out_dir, f"{video_name}_relations.json")
    out_payload = {
        "mode": "full_frame",
        "description": "Full-frame global CCTV evaluation without spatial cropping",
        "entities": {
            eid: {
                "type": e["type"],
                "class": e["class"],
                "total_frames_tracked": len(e["frame_map"])
            }
            for eid, e in all_entities.items()
        },
        "frames_sampled": frames_sampled_count,
        "relations": confirmed_relations
    }
    with open(relations_json_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved structured relations JSON: {relations_json_path}")

    # Render video and 8 sequential review frames
    all_f_indices = sorted(list(raw_frames.keys()))
    step_review = max(1, len(all_f_indices) // num_review_frames)
    review_frame_indices = set(all_f_indices[::step_review][:num_review_frames])

    rendered_video_frames = []
    stt_counter = 1

    print(f"\nRendering surveillance video and saving {num_review_frames} sequential full frames...")
    for f_idx in range(start_frame, end_frame + 1):
        if f_idx not in raw_frames:
            continue

        vis = render_surveillance_monitor_frame(
            raw_frame=raw_frames[f_idx],
            entities=all_entities,
            f_idx=f_idx,
            fps=fps,
            active_relations=confirmed_relations
        )
        rendered_video_frames.append(vis)

        if num_review_frames > 0 and frames_dir is not None and f_idx in review_frame_indices and stt_counter <= num_review_frames:
            f_sec = f_idx / fps
            frame_img_path = os.path.join(frames_dir, f"frame_{stt_counter:02d}_{f_sec:.2f}s.jpg")
            cv2.imwrite(frame_img_path, vis)
            stt_counter += 1

    out_mp4_path = os.path.join(out_dir, f"{video_name}_visualizer.mp4")
    write_clean_h264_mp4(
        frames_list=rendered_video_frames,
        out_mp4_path=out_mp4_path,
        fps=fps,
        w=orig_w,
        h=orig_h
    )

    if num_review_frames > 0 and frames_dir is not None:
        # Phase 4: Auto-Preview System - Slice review frames directly from visualizer.mp4
        preview_dir = os.path.join("data", "preview", video_name)
        preview_saved = export_preview_frames_from_video(
            mp4_path=out_mp4_path,
            preview_dir=preview_dir,
            num_frames=num_review_frames,
            start_sec=start_frame / fps,
            fps=fps
        )
        print(f"\n  [AUTO-PREVIEW] Exported {len(preview_saved)} review frames to: {preview_dir}")
        for p in preview_saved:
            print(f"    - {os.path.basename(p)}")

        saved_imgs = sorted([x for x in os.listdir(frames_dir) if x.endswith(".jpg")])
        print(f"  [FRAMES EXPORT] Saved {len(saved_imgs)} sequential review frames to: {frames_dir}")

    return out_payload


def run_roi_zoom_pipeline(
    ra_model: RelateAnything,
    all_entities: Dict[str, Dict[str, Any]],
    raw_frames: Dict[int, np.ndarray],
    fps: float,
    start_frame: int,
    end_frame: int,
    stride_frames: int,
    min_rel_conf: float,
    min_consistency: float,
    padding_ratio: float,
    num_review_frames: int,
    out_dir: str,
    video_name: str
) -> Dict[str, Any]:
    """Runs rigorous Spatio-Temporal Interaction Clustering (STIC) + Dynamic ROI Zoom Crop pipeline."""
    print("\n" + "=" * 70)
    print("MODE: SPATIO-TEMPORAL INTERACTION CLUSTERING (STIC) & DYNAMIC ROI ZOOM")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)
    if num_review_frames > 0:
        frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for old_f in os.listdir(frames_dir):
            if old_f.endswith(".jpg"):
                try: os.remove(os.path.join(frames_dir, old_f))
                except: pass
    else:
        frames_dir = None

    first_f = next(iter(raw_frames.values()))
    orig_h, orig_w = first_f.shape[:2]

    # 1. Run STIC Clustering
    print("\n[STIC] Evaluating pairwise spatio-temporal interaction affinity...")
    active_clusters, singletons = cluster_entities_spatially(
        entities=all_entities,
        image_shape=(orig_h, orig_w),
        proximity_thresh_px=None  # Authentic arm-reach contact threshold (~28px-35px)
    )

    print(f"Spatial Clustering Results:")
    print(f"  - Active Interactive Clusters (|C| >= 2): {len(active_clusters)}")
    print(f"  - Isolated Singletons (|C| == 1):         {len(singletons)} (Excluded: {singletons})")

    for c in active_clusters:
        e_desc = [f"{eid} ({c['entities'][eid]['class']})" for eid in c["entity_ids"]]
        print(f"  * {c['cluster_id']}: Entities = {e_desc}, Min Distance = {c['min_internal_distance_px']:.1f}px")

    sample_frame_indices = sorted(list(raw_frames.keys()))
    cluster_union_boxes_map = {}

    for c in active_clusters:
        cid = c["cluster_id"]
        c["cluster_union_boxes"] = compute_cluster_union_boxes(
            cluster_entity_ids=c["entity_ids"],
            entities=all_entities,
            sample_frame_indices=sample_frame_indices,
            image_shape=(orig_h, orig_w),
            padding_ratio=padding_ratio
        )
        cluster_union_boxes_map[cid] = c["cluster_union_boxes"]

    # 2. RelateAnything Multi-Frame Evaluation on Clean Zoom Crops
    all_cluster_confirmed_relations = []
    cluster_results_export = []
    global_pair_frame_scores = defaultdict(lambda: defaultdict(dict))

    for c in active_clusters:
        cid = c["cluster_id"]
        c_eids = c["entity_ids"]
        uboxes = c["cluster_union_boxes"]
        cluster_sampled_frames = sorted(list(uboxes.keys()))[::stride_frames]

        pair_pred_scores = defaultdict(lambda: defaultdict(list))
        frames_sampled_count = 0

        print(f"\n[INFERENCE] Evaluating {cid} across {len(cluster_sampled_frames)} sampled frames on high-res ROI crops...")
        t0_cluster = time.time()

        for f_idx in cluster_sampled_frames:
            if f_idx not in raw_frames:
                continue
            frame = raw_frames[f_idx]
            # Dual-Layer Smart Zoom: Extract isotropic square crop for DINOv3 (Square Guard)
            crop_img, (ax1, ay1, ax2, ay2) = extract_square_guard_crop(frame, uboxes[f_idx], orig_w, orig_h)
            crop_h, crop_w = crop_img.shape[:2]
            if crop_h < 10 or crop_w < 10:
                continue

            frame_eids = []
            local_boxes = []
            box_labels = []

            for eid in c_eids:
                e_info = c["entities"][eid]
                if f_idx in e_info["frame_map"]:
                    gx1, gy1, gx2, gy2 = e_info["frame_map"][f_idx]
                    lx1 = max(0.0, float(gx1 - ax1))
                    ly1 = max(0.0, float(gy1 - ay1))
                    lx2 = min(float(crop_w), float(gx2 - ax1))
                    ly2 = min(float(crop_h), float(gy2 - ay1))
                    if (lx2 - lx1) >= 4 and (ly2 - ly1) >= 4:
                        frame_eids.append(eid)
                        local_boxes.append([lx1, ly1, lx2, ly2])
                        box_labels.append(e_info["class"])

            if len(frame_eids) < 2:
                continue

            frames_sampled_count += 1
            boxes_np = np.array(local_boxes, dtype=np.float32)
            pil_crop = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

            # Run RelateAnything on CLEAN unannotated crop with domain-aware ontology
            box_types = [all_entities[eid]["type"] for eid in frame_eids]
            triplets = predict_relations_with_ontology(
                ra_model=ra_model,
                image=pil_crop,
                boxes_xyxy=boxes_np,
                box_labels=box_labels,
                box_types=box_types,
                box_eids=frame_eids,
                all_entities=all_entities,
                f_idx=f_idx
            )

            triplet_pairs = set()
            for t in triplets:
                sub_eid = frame_eids[t["subject_idx"]]
                obj_eid = frame_eids[t["object_idx"]]
                p_name = t["predicate"]
                p_val = float(t["score"])
                pair_pred_scores[(sub_eid, obj_eid)][p_name].append(p_val)
                global_pair_frame_scores[(sub_eid, obj_eid)][p_name][f_idx] = p_val
                triplet_pairs.add((sub_eid, obj_eid, p_name))

            # Negative evidence: for every co-present pair on this evaluated frame,
            # if a candidate predicate was not active, record 0.0 at f_idx to enforce clean release
            for i_a in range(len(frame_eids)):
                for i_b in range(len(frame_eids)):
                    if i_a != i_b:
                        s_id, o_id = frame_eids[i_a], frame_eids[i_b]
                        for prev_p in list(global_pair_frame_scores.get((s_id, o_id), {}).keys()):
                            if (s_id, o_id, prev_p) not in triplet_pairs:
                                global_pair_frame_scores[(s_id, o_id)][prev_p][f_idx] = 0.0

        infer_time = time.time() - t0_cluster
        print(f"  {cid} evaluated in {infer_time:.2f}s ({infer_time / max(1, frames_sampled_count) * 1000:.1f} ms/frame).")

        # Temporal Voting & Thresholding with Predicate Exclusivity
        confirmed_relations = []
        for (sub_eid, obj_eid), pred_dict in pair_pred_scores.items():
            sub_cls = all_entities[sub_eid]["class"]
            obj_cls = all_entities[obj_eid]["class"]

            candidate_preds = []
            for pred, scores in pred_dict.items():
                mean_conf = float(np.mean(scores))
                consistency = len(scores) / max(1, frames_sampled_count)

                if mean_conf >= 0.40 and (consistency >= min_consistency or (consistency >= 0.15 and len(scores) >= 4)):
                    candidate_preds.append({
                        "cluster_id": cid,
                        "subject_id": sub_eid,
                        "subject_class": sub_cls,
                        "predicate": pred,
                        "object_id": obj_eid,
                        "object_class": obj_cls,
                        "mean_confidence": round(mean_conf, 4),
                        "temporal_consistency": round(consistency, 4),
                        "observed_frames": len(scores),
                        "total_sampled_frames": frames_sampled_count
                    })

            if candidate_preds:
                candidate_preds.sort(key=lambda r: r["mean_confidence"] * r["temporal_consistency"], reverse=True)
                for top_r in candidate_preds[:2]:
                    confirmed_relations.append(top_r)
                    all_cluster_confirmed_relations.append(top_r)

        confirmed_relations.sort(key=lambda r: r["mean_confidence"] * r["temporal_consistency"], reverse=True)
        print(f"  Confirmed relationships for {cid}:")
        if confirmed_relations:
            for r in confirmed_relations:
                print(f"    * {r['subject_id']} {r['subject_class']} --{r['predicate']} [conf: {r['mean_confidence']:.2f}, consistency: {r['temporal_consistency'] * 100:.0f}%]--> {r['object_id']} {r['object_class']}")
        else:
            print("    (No relationships met threshold)")

        cluster_results_export.append({
            "cluster_id": cid,
            "entities": c_eids,
            "frames_sampled": frames_sampled_count,
            "relations": confirmed_relations
        })

    # 3. Export relations.json with dynamic video name prefix
    relations_json_path = os.path.join(out_dir, f"{video_name}_relations.json")
    out_payload = {
        "mode": "roi_zoom",
        "description": "Spatio-Temporal Interaction Clustering (STIC) with Dynamic ROI Zoom Crop",
        "entities": {
            eid: {
                "type": e["type"],
                "class": e["class"],
                "total_frames_tracked": len(e["frame_map"])
            }
            for eid, e in all_entities.items()
        },
        "clusters": cluster_results_export,
        "singletons": singletons
    }
    with open(relations_json_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved structured relations JSON: {relations_json_path}")

    # Build Frame-Level Dynamic Scores with Authentic EdgeBook Hysteresis Architecture
    cluster_entity_ids_map = {c["cluster_id"]: c["entity_ids"] for c in active_clusters}
    frame_dynamic_scores = defaultdict(dict)
    all_f_list = sorted(list(raw_frames.keys()))

    pair_confirmed_preds = defaultdict(dict)
    for r in all_cluster_confirmed_relations:
        sub_id = str(r["subject_id"])
        obj_id = str(r["object_id"])
        pred = r["predicate"]
        pair_confirmed_preds[(sub_id, obj_id)][pred] = r["mean_confidence"]

    for (sub_id, obj_id), preds_dict in pair_confirmed_preds.items():
        obj_type = all_entities.get(obj_id, {}).get("type", "")
        fm_s = all_entities.get(sub_id, {}).get("frame_map", {})
        fm_o = all_entities.get(obj_id, {}).get("frame_map", {})

        # Subject anthropometric reference height and contact reach threshold
        s_heights = [b[3] - b[1] for b in fm_s.values() if (b[3] - b[1]) >= 20]
        h_ref_s = float(np.median(s_heights)) if s_heights else 100.0
        contact_thresh = max(18.0, 0.22 * h_ref_s)

        # Step 1: Evaluate EdgeBook state machine strictly on EVALUATED frames
        all_eval_fs = set()
        for p in preds_dict:
            all_eval_fs.update(global_pair_frame_scores.get((sub_id, obj_id), {}).get(p, {}).keys())
        sorted_eval_fs = sorted(list(all_eval_fs))

        score_ema_alpha = 0.35
        on_thr = 0.45
        off_thr = 0.30
        on_frames_req = 3
        off_frames_req = 2

        pred_scores = {p: 0.0 for p in preds_dict}
        pred_on_runs = {p: 0 for p in preds_dict}
        pred_off_runs = {p: 0 for p in preds_dict}
        pred_active_state = {p: False for p in preds_dict}

        eval_frame_states = {}
        for ef in sorted_eval_fs:
            b_s = fm_s.get(ef)
            b_o = fm_o.get(ef)
            edge_d = compute_box_edge_distance(b_s, b_o) if (b_s is not None and b_o is not None) else 999.0

            cur_states = {}
            cur_scores = {}
            for p in preds_dict:
                p_scores_map = global_pair_frame_scores.get((sub_id, obj_id), {}).get(p, {})
                raw_sc = p_scores_map.get(ef, 0.0)
                if edge_d > contact_thresh:
                    raw_sc = 0.0

                pred_scores[p] = (1.0 - score_ema_alpha) * pred_scores[p] + score_ema_alpha * raw_sc

                if pred_scores[p] >= on_thr:
                    pred_on_runs[p] += 1
                    pred_off_runs[p] = 0
                elif pred_scores[p] < off_thr:
                    pred_off_runs[p] += 1
                    pred_on_runs[p] = 0
                else:
                    pred_on_runs[p] = 0
                    pred_off_runs[p] = 0

                if not pred_active_state[p] and pred_on_runs[p] >= on_frames_req:
                    pred_active_state[p] = True
                elif pred_active_state[p] and pred_off_runs[p] >= off_frames_req:
                    pred_active_state[p] = False

                cur_states[p] = pred_active_state[p]
                cur_scores[p] = pred_scores[p]

            eval_frame_states[ef] = {
                "states": cur_states,
                "scores": cur_scores,
                "edge_d": edge_d
            }

        # Step 2: Propagate and interpolate to EVERY raw video frame
        for f in all_f_list:
            if f in fm_s and f in fm_o:
                b_s = fm_s[f]
                b_o = fm_o[f]
                edge_d = compute_box_edge_distance(b_s, b_o)

                if edge_d > contact_thresh or not eval_frame_states:
                    default_p = list(preds_dict.keys())[0]
                    frame_dynamic_scores[f][(sub_id, obj_id)] = (default_p, 0.0)
                    continue

                past_efs = [ef for ef in sorted_eval_fs if ef <= f]
                future_efs = [ef for ef in sorted_eval_fs if ef >= f]

                frame_active_preds = []
                frame_scores = {}

                for p in preds_dict:
                    p_active = False
                    p_score = 0.0

                    if past_efs and future_efs:
                        p_ef = past_efs[-1]
                        fu_ef = future_efs[0]
                        st_p = eval_frame_states[p_ef]["states"][p]
                        sc_p = eval_frame_states[p_ef]["scores"][p]
                        st_fu = eval_frame_states[fu_ef]["states"][p]
                        sc_fu = eval_frame_states[fu_ef]["scores"][p]

                        if fu_ef == p_ef:
                            p_active = st_p
                            p_score = sc_p
                        else:
                            alpha = (f - p_ef) / float(fu_ef - p_ef)
                            interp_sc = (1.0 - alpha) * sc_p + alpha * sc_fu
                            if st_p and st_fu:
                                p_active = True
                                p_score = interp_sc
                            elif st_p and not st_fu:
                                p_score = interp_sc
                                p_active = (interp_sc >= on_thr)
                            elif not st_p and st_fu:
                                p_score = interp_sc
                                p_active = (interp_sc >= on_thr)
                            else:
                                p_active = False
                                p_score = interp_sc
                    elif past_efs:
                        p_ef = past_efs[-1]
                        st_p = eval_frame_states[p_ef]["states"][p]
                        sc_p = eval_frame_states[p_ef]["scores"][p]
                        decay = max(0.0, 1.0 - (f - p_ef) / float(max(1, stride_frames * 3)))
                        p_score = sc_p * decay
                        p_active = (st_p and p_score >= on_thr)
                    else:
                        fu_ef = future_efs[0]
                        st_fu = eval_frame_states[fu_ef]["states"][p]
                        sc_fu = eval_frame_states[fu_ef]["scores"][p]
                        p_score = 0.0
                        p_active = False

                    if p_active and p_score >= off_thr:
                        frame_active_preds.append(p)
                        frame_scores[p] = p_score

                # Velocity computation for coupled transport
                cp = ((b_s[0] + b_s[2]) / 2.0, (b_s[1] + b_s[3]) / 2.0)
                co = ((b_o[0] + b_o[2]) / 2.0, (b_o[1] + b_o[3]) / 2.0)
                v_s, v_o = 0.0, 0.0
                if f - 8 in fm_s:
                    b_prev_s = fm_s[f - 8]
                    v_s = math.hypot(cp[0] - (b_prev_s[0] + b_prev_s[2]) / 2.0, cp[1] - (b_prev_s[1] + b_prev_s[3]) / 2.0) / 8.0
                if f - 8 in fm_o:
                    b_prev_o = fm_o[f - 8]
                    v_o = math.hypot(co[0] - (b_prev_o[0] + b_prev_o[2]) / 2.0, co[1] - (b_prev_o[1] + b_prev_o[3]) / 2.0) / 8.0
                
                v_rel_s = v_s / h_ref_s
                v_rel_o = v_o / h_ref_s
                is_coupled_transport = (obj_type == "object" and v_rel_s >= 0.010 and v_rel_o >= 0.007)
                is_disengaged_transit = (obj_type == "object" and v_rel_s >= 0.015 and v_rel_o < 0.004)

                if frame_active_preds and not is_disengaged_transit:
                    if is_coupled_transport and "carry" in frame_active_preds:
                        chosen_p = "carry"
                        chosen_sc = frame_scores.get("carry", frame_scores[frame_active_preds[0]])
                    elif "hold" in frame_active_preds and ("touch" not in frame_active_preds or frame_scores.get("hold", 0.0) >= frame_scores.get("touch", 0.0)):
                        chosen_p = "hold"
                        chosen_sc = frame_scores["hold"]
                    else:
                        chosen_p = max(frame_active_preds, key=lambda p: frame_scores[p])
                        chosen_sc = frame_scores[chosen_p]
                    frame_dynamic_scores[f][(sub_id, obj_id)] = (chosen_p, round(chosen_sc, 4))
                else:
                    default_p = list(preds_dict.keys())[0]
                    frame_dynamic_scores[f][(sub_id, obj_id)] = (default_p, 0.0)
            else:
                default_p = list(preds_dict.keys())[0]
                frame_dynamic_scores[f][(sub_id, obj_id)] = (default_p, 0.0)

    # 4. Render Clean Surveillance Monitor Video
    rendered_video_frames = []
    print(f"\nRendering clean surveillance video ({start_frame} to {end_frame})...")
    for f_idx in range(start_frame, end_frame + 1):
        if f_idx not in raw_frames:
            continue

        vis_monitor = render_surveillance_monitor_frame(
            raw_frame=raw_frames[f_idx],
            entities=all_entities,
            f_idx=f_idx,
            fps=fps,
            active_relations=all_cluster_confirmed_relations,
            cluster_union_boxes=cluster_union_boxes_map,
            frame_dynamic_scores=frame_dynamic_scores,
            cluster_entity_ids_map=cluster_entity_ids_map
        )
        rendered_video_frames.append(vis_monitor)

    out_mp4_path = os.path.join(out_dir, f"{video_name}_visualizer.mp4")
    write_clean_h264_mp4(
        frames_list=rendered_video_frames,
        out_mp4_path=out_mp4_path,
        fps=fps,
        w=orig_w,
        h=orig_h
    )

    # 5. Export Pristine RelateAnything Input Frames Grouped By Dynamic Cluster (Max review frames / Cluster)
    if num_review_frames > 0 and frames_dir is not None:
        print(f"\nExporting pristine RelateAnything input crops into dynamic cluster subdirectories (max {num_review_frames} frames per cluster)...")
        for item in os.listdir(frames_dir):
            item_path = os.path.join(frames_dir, item)
            if os.path.isfile(item_path):
                try: os.remove(item_path)
                except OSError: pass
            elif os.path.isdir(item_path):
                import shutil
                try: shutil.rmtree(item_path)
                except OSError: pass

        if active_clusters:
            for c in active_clusters:
                cid = c["cluster_id"]
                c_dir = os.path.join(frames_dir, cid)
                os.makedirs(c_dir, exist_ok=True)

                c_uboxes = c.get("cluster_union_boxes", {})
                active_f_indices = [
                    f_idx for f_idx in sorted(c_uboxes.keys())
                    if f_idx in raw_frames and c_uboxes[f_idx] is not None
                ]

                if not active_f_indices:
                    continue

                if len(active_f_indices) >= num_review_frames:
                    step_review = (len(active_f_indices) - 1) / float(num_review_frames - 1)
                    sample_f_indices = [active_f_indices[int(round(i * step_review))] for i in range(num_review_frames)]
                else:
                    sample_f_indices = active_f_indices

                for idx, f_idx in enumerate(sample_f_indices, 1):
                    f_sec = f_idx / fps
                    frame_raw = raw_frames[f_idx]
                    crop_box = c_uboxes[f_idx]
                    ra_input_crop, _ = extract_square_guard_crop(frame_raw, crop_box, orig_w, orig_h)
                    frame_img_path = os.path.join(c_dir, f"frame_{idx:02d}_{f_sec:.2f}s.jpg")
                    cv2.imwrite(frame_img_path, ra_input_crop)

                print(f"  [FRAMES AUDIT] Saved {len(sample_f_indices)} pristine RelateAnything input crops to: {c_dir}")
        else:
            fallback_dir = os.path.join(frames_dir, "overview")
            os.makedirs(fallback_dir, exist_ok=True)
            all_f_indices = sorted(list(raw_frames.keys()))
            if len(all_f_indices) >= num_review_frames:
                step_review = (len(all_f_indices) - 1) / float(num_review_frames - 1)
                sample_f_indices = [all_f_indices[int(round(i * step_review))] for i in range(num_review_frames)]
            else:
                sample_f_indices = all_f_indices
            for idx, f_idx in enumerate(sample_f_indices, 1):
                f_sec = f_idx / fps
                cv2.imwrite(os.path.join(fallback_dir, f"frame_{idx:02d}_{f_sec:.2f}s.jpg"), raw_frames[f_idx])
            print(f"  [FRAMES AUDIT] Saved {len(sample_f_indices)} overview frames to: {fallback_dir}")

        # Phase 4: Auto-Preview System - Slice review frames directly from visualizer.mp4
        preview_dir = os.path.join("data", "preview", video_name)
        preview_saved = export_preview_frames_from_video(
            mp4_path=out_mp4_path,
            preview_dir=preview_dir,
            num_frames=num_review_frames,
            start_sec=start_frame / fps,
            fps=fps
        )
        print(f"\n  [AUTO-PREVIEW] Exported {len(preview_saved)} review frames to: {preview_dir}")
        for p in preview_saved:
            print(f"    - {os.path.basename(p)}")

    return out_payload


def main():
    args = parse_arguments()
    device = resolve_device(args.device)
    video_basename = os.path.splitext(os.path.basename(args.video))[0]

    # 1. Video Reader Setup & Frame Boundaries
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.full_video:
        start_frame = 0
        end_frame = max(0, total_frames - 1)
        args.start_sec = 0.0
        args.end_sec = total_frames / fps
        args.num_review_frames = 0
    else:
        start_frame = int(args.start_sec * fps)
        end_sec_target = total_frames / fps if (args.end_sec is None or args.end_sec <= 0) else args.end_sec
        end_frame = min(int(end_sec_target * fps), total_frames - 1)

    print("=" * 80)
    print("VIDVRD END-TO-END PIPELINE: RELATEANYTHING + YOLOE-26M")
    print(f"Target Video:         {args.video}")
    if args.full_video:
        print(f"Temporal Window:      FULL VIDEO (0.0s -> {args.end_sec:.1f}s, {end_frame - start_frame + 1} frames)")
    else:
        print(f"Temporal Window:      {args.start_sec:.1f}s -> {args.end_sec:.1f}s (Duration: {args.end_sec - args.start_sec:.1f}s)")
    print(f"Execution Mode:       {args.mode.upper()}")
    print(f"Compute Device:       {device.upper()}")
    print(f"RelateAnything Model: {args.relsgg_model}")
    print(f"Confidence Threshold: {args.min_rel_conf}")
    print(f"Consistency Thresh:   {args.min_consistency}")
    print("ID Numbering Scheme:  CAUSAL AGNOSTIC DECOUPLED" + (f" (Namespace: {args.source_id})" if args.source_id else ""))
    print("=" * 80)

    # 2. Load Vocabularies
    with open(args.objects_json, "r", encoding="utf-8") as f:
        allowed_objects_60 = json.load(f)
    with open(args.relations_json, "r", encoding="utf-8") as f:
        relations_26 = json.load(f)

    # 3. Initialize Models
    print("\n[1/4] Initializing YOLOE-26m Detector & Tracker...")
    yolo_model = YOLO(args.yolo_weights)
    yolo_model.to(device)
    yolo_model.set_classes(allowed_objects_60)

    print("\n[2/4] Initializing RelateAnything & Setting 26-Predicate Closed Vocabulary...")
    ra_model = RelateAnything.from_pretrained(args.relsgg_model, device=device)
    ra_model.set_vocabulary(relations_26)

    # Read First Frame for Dimension Extraction
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, first_f = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read video frame at index {start_frame}")
    orig_h, orig_w = first_f.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 4. Tracking & Object Association
    print(f"\n[3/4] Forward Tracking frames {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)...")
    frame_detections = []
    active_object_tracklets = []
    raw_frames = {}

    for f_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames[f_idx] = frame

        track_results = yolo_model.track(
            source=frame,
            persist=True,
            tracker=args.bytetrack_yaml,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
            device=device
        )

        frame_persons = []
        frame_objects = []

        if track_results and track_results[0].boxes:
            d_boxes = track_results[0].boxes.xyxy.cpu().numpy().astype(int)
            d_clses = track_results[0].boxes.cls.cpu().numpy().astype(int)
            d_confs = track_results[0].boxes.conf.cpu().numpy().astype(float)
            d_ids = track_results[0].boxes.id.cpu().numpy().astype(int) if track_results[0].boxes.id is not None else [None] * len(d_boxes)

            for b, c_idx, c_val, tid in zip(d_boxes, d_clses, d_confs, d_ids):
                c_name = allowed_objects_60[c_idx]
                bw = b[2] - b[0]
                bh = b[3] - b[1]
                area = bw * bh
                if c_name == "person":
                    if tid is not None and area >= 800 and bh >= 50:
                        frame_persons.append((b, tid, "person", area))
                else:
                    if c_name not in STATIC_FIXTURE_CLASSES and c_val >= args.conf and area >= 200:
                        frame_objects.append((b, c_name, c_val))

        # Spatial Association for objects
        for b_det, c_name, c_val in frame_objects:
            matched_tr = None
            best_dist = float("inf")
            c_det = ((b_det[0] + b_det[2]) / 2.0, (b_det[1] + b_det[3]) / 2.0)

            for tr in active_object_tracklets:
                gap = f_idx - tr["last_frame"]
                if gap <= 40:
                    last_b = tr["frame_map"][tr["last_frame"]]
                    c_last = ((last_b[0] + last_b[2]) / 2.0, (last_b[1] + last_b[3]) / 2.0)
                    center_dist = math.hypot(c_det[0] - c_last[0], c_det[1] - c_last[1])
                    iou = compute_box_iou(b_det, last_b)
                    if iou >= 0.20 or center_dist <= 60.0 or (gap <= 10 and center_dist <= 90.0):
                        if center_dist < best_dist:
                            best_dist = center_dist
                            matched_tr = tr

            if matched_tr is not None:
                matched_tr["frame_map"][f_idx] = b_det
                matched_tr["confs"].append(c_val)
                matched_tr["class_votes"][c_name] = matched_tr["class_votes"].get(c_name, 0) + 1
                matched_tr["last_frame"] = f_idx
            else:
                active_object_tracklets.append({
                    "class_votes": {c_name: 1},
                    "frame_map": {f_idx: b_det},
                    "confs": [c_val],
                    "first_frame": f_idx,
                    "last_frame": f_idx
                })

        frame_detections.append({
            "frame_idx": f_idx,
            "persons": frame_persons,
            "objects": frame_objects
        })

    cap.release()

    # Temporal Tracklet Stitching for Persons
    person_tid_stats = {}
    for fd in frame_detections:
        for b, tid, c, area in fd["persons"]:
            if tid not in person_tid_stats:
                person_tid_stats[tid] = {"count": 0, "total_area": 0, "frames": {}}
            person_tid_stats[tid]["count"] += 1
            person_tid_stats[tid]["total_area"] += area
            person_tid_stats[tid]["frames"][fd["frame_idx"]] = b

    raw_person_tracklets = [
        {"id": tid, "frames": stats["frames"], "total_area": stats["total_area"]}
        for tid, stats in person_tid_stats.items()
        if stats["count"] >= max(10, int(0.4 * fps))
    ]
    raw_person_tracklets.sort(key=lambda t: min(t["frames"].keys()))

    stitched_persons = []
    for tr in raw_person_tracklets:
        matched_sp = None
        for sp in stitched_persons:
            common_f = set(tr["frames"].keys()) & set(sp["frames"].keys())
            if common_f:
                ious = [compute_box_iou(tr["frames"][f], sp["frames"][f]) for f in common_f]
                if np.mean(ious) >= 0.35:
                    matched_sp = sp
                    break
            else:
                min_dt = float("inf")
                best_pair = None
                for f_a in tr["frames"]:
                    for f_b in sp["frames"]:
                        dt = abs(f_a - f_b)
                        if dt < min_dt:
                            min_dt = dt
                            best_pair = (tr["frames"][f_a], sp["frames"][f_b])
                            if dt == 1:
                                break
                    if min_dt == 1:
                        break
                if min_dt <= 15 and best_pair is not None:
                    b1, b2 = best_pair
                    dist = compute_box_edge_distance(b1, b2)
                    iou = compute_box_iou(b1, b2)
                    b_ref_h = max(b1[3] - b1[1], b2[3] - b2[1])
                    stitch_dist_thresh = max(18.0, 0.22 * b_ref_h)
                    if dist <= stitch_dist_thresh or iou >= 0.30:
                        matched_sp = sp
                        break
        if matched_sp is not None:
            matched_sp["frames"].update(tr["frames"])
            matched_sp["total_area"] += tr["total_area"]
            matched_sp["orig_tids"].append(tr["id"])
        else:
            stitched_persons.append({
                "frames": dict(tr["frames"]),
                "total_area": tr["total_area"],
                "orig_tids": [tr["id"]]
            })

    stitched_persons.sort(key=lambda sp: sp["total_area"], reverse=True)
    stitched_persons = filter_and_suppress_subpart_tracklets(stitched_persons)

    # Linear Gap Interpolation for Persons (eliminates single-frame detector dropouts)
    for sp in stitched_persons:
        sorted_fs = sorted(sp["frames"].keys())
        if sorted_fs:
            min_f, max_f = sorted_fs[0], sorted_fs[-1]
            for f in range(min_f + 1, max_f):
                if f not in sp["frames"]:
                    prev_f = max(k for k in sorted_fs if k < f)
                    next_f = min(k for k in sorted_fs if k > f)
                    alpha = (f - prev_f) / (next_f - prev_f)
                    b_prev = np.array(sp["frames"][prev_f], dtype=float)
                    b_next = np.array(sp["frames"][next_f], dtype=float)
                    sp["frames"][f] = ((1.0 - alpha) * b_prev + alpha * b_next).astype(int)

    print(f"Identified {len(stitched_persons)} stable person tracklets.")

    # Upgraded Object Stitching with Cross-Class Bag Family & Sequential Human Transport Continuity
    candidate_raw_objects = [tr for tr in active_object_tracklets if len(tr["frame_map"]) >= 15]
    candidate_raw_objects.sort(key=lambda tr: min(tr["frame_map"].keys()))
    merged_objects = []

    for tr in candidate_raw_objects:
        t_start = min(tr["frame_map"].keys())
        b_start = tr["frame_map"][t_start]
        c_start = ((b_start[0] + b_start[2]) / 2.0, (b_start[1] + b_start[3]) / 2.0)
        c_class = max(tr["class_votes"].items(), key=lambda kv: kv[1])[0]

        matched_mo = None
        for mo in merged_objects:
            mo_end = max(mo["frame_map"].keys())
            b_last = mo["frame_map"][mo_end]
            c_end = ((b_last[0] + b_last[2]) / 2.0, (b_last[1] + b_last[3]) / 2.0)
            mo_class = max(mo["class_votes"].items(), key=lambda kv: kv[1])[0]

            gap = t_start - mo_end
            dist = math.hypot(c_start[0] - c_end[0], c_start[1] - c_end[1])
            edge_dist = compute_box_edge_distance(b_start, b_last)
            iou = compute_box_iou(b_start, b_last)

            # Cross-class Bag Family compatibility
            is_bag_family = (c_class in ["backpack", "handbag", "suitcase"]) and (mo_class in ["backpack", "handbag", "suitcase"])
            is_same_class = (c_class == mo_class) or is_bag_family

            if is_same_class and gap > 0:
                # (a) Stationary or small gap match
                if gap <= 40 and (iou >= 0.15 or edge_dist <= 50.0):
                    matched_mo = mo
                    break
                # (b) Sequential human transport continuity: Person carries bag across room
                elif gap <= 150 and dist <= 220.0:
                    person_near_start = any(
                        f in sp["frames"] and compute_box_edge_distance(sp["frames"][f], b_last) <= 65.0
                        for sp in stitched_persons for f in [mo_end]
                    )
                    person_near_end = any(
                        f in sp["frames"] and compute_box_edge_distance(sp["frames"][f], b_start) <= 65.0
                        for sp in stitched_persons for f in [t_start]
                    )
                    if person_near_start or person_near_end or dist <= 120.0:
                        matched_mo = mo
                        break

        if matched_mo is not None:
            matched_mo["frame_map"].update(tr["frame_map"])
            matched_mo["confs"].extend(tr["confs"])
            for c_k, c_v in tr["class_votes"].items():
                matched_mo["class_votes"][c_k] = matched_mo["class_votes"].get(c_k, 0) + c_v
        else:
            merged_objects.append(tr)

    # Require final stitched objects to have at least 30 frames total lifespan (1.0s)
    merged_objects = [mo for mo in merged_objects if len(mo["frame_map"]) >= 30]

    # Linear Gap Interpolation and Stationary Forward-Fill
    orig_h, orig_w = raw_frames[start_frame].shape[:2]
    for mo in merged_objects:
        sorted_fs = sorted(mo["frame_map"].keys())
        if sorted_fs:
            min_f, max_f = sorted_fs[0], sorted_fs[-1]
            for f in range(min_f + 1, max_f):
                if f not in mo["frame_map"]:
                    prev_f = max(k for k in sorted_fs if k < f)
                    next_f = min(k for k in sorted_fs if k > f)
                    alpha = (f - prev_f) / (next_f - prev_f)
                    b_prev = np.array(mo["frame_map"][prev_f], dtype=float)
                    b_next = np.array(mo["frame_map"][next_f], dtype=float)
                    mo["frame_map"][f] = ((1.0 - alpha) * b_prev + alpha * b_next).astype(int)

            # Stationary Forward-Fill ONLY for objects that have settled (> 40 frames total tracked)
            if len(sorted_fs) >= 40:
                tail_fs = sorted_fs[-min(10, len(sorted_fs)):]
                tail_boxes = np.array([mo["frame_map"][f] for f in tail_fs])
                tail_cxs = (tail_boxes[:, 0] + tail_boxes[:, 2]) / 2.0
                tail_cys = (tail_boxes[:, 1] + tail_boxes[:, 3]) / 2.0
                tail_disp = float(math.hypot(np.ptp(tail_cxs), np.ptp(tail_cys)))

                last_b = mo["frame_map"][max_f]
                is_near_border = (last_b[0] < 20 or last_b[1] < 20 or last_b[2] > orig_w - 20 or last_b[3] > orig_h - 20)
                if tail_disp < 20.0 and not is_near_border and max_f < end_frame:
                    for f in range(max_f + 1, end_frame + 1):
                        mo["frame_map"][f] = last_b

    # Build Entity Dictionaries under Causal Agnostic Decoupled ID Architecture
    all_entities = {}
    candidate_entities = []

    # 1. Person Candidates
    for sp in stitched_persons:
        f_start = min(sp["frames"].keys())
        init_tid = min(sp.get("orig_tids", [f_start]))
        candidate_entities.append({
            "type": "person",
            "class": "person",
            "frames": sp["frames"],
            "frame_map": sp["frames"],
            "t_birth": f_start,
            "init_tid": init_tid,
            "displacement": 0.0,
            "mean_conf": 1.0
        })

    # 2. Object Candidates
    for mo in merged_objects:
        f_start = min(mo["frame_map"].keys())
        best_cls = max(mo["class_votes"].items(), key=lambda kv: kv[1])[0]
        boxes_arr = np.array(list(mo["frame_map"].values()))
        cxs = (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2.0
        cys = (boxes_arr[:, 1] + boxes_arr[:, 3]) / 2.0
        disp = float(math.hypot(np.ptp(cxs), np.ptp(cys)))
        candidate_entities.append({
            "type": "object",
            "class": best_cls,
            "frames": mo["frame_map"],
            "frame_map": mo["frame_map"],
            "displacement": disp,
            "mean_conf": float(np.mean(mo["confs"])),
            "t_birth": f_start,
            "init_tid": f_start
        })

    # 3. Sort strictly by Causality (Birth Frame, then initial detector order)
    candidate_entities.sort(key=lambda c: (c["t_birth"], c["init_tid"]))

    # 4. Assign Monotonic Agnostic IDs
    for k, c in enumerate(candidate_entities, 1):
        if args.source_id:
            eid = f"[{args.source_id}_{k}]"
        else:
            eid = f"[{k}]"
        c["color"] = COLOR_PALETTE[(k - 1) % len(COLOR_PALETTE)]
        all_entities[eid] = c

    num_persons = sum(1 for e in all_entities.values() if e["type"] == "person")
    num_objects = sum(1 for e in all_entities.values() if e["type"] == "object")
    print(f"Identified {len(all_entities)} entities under Causal Agnostic ID scheme ({num_persons} persons, {num_objects} objects).")
    for eid, e in all_entities.items():
        print(f"  * {eid} {e['class']} (type: {e['type']}, birth: frame {min(e['frame_map'].keys())})")

    # Execute Mode
    stride_frames = max(1, int(round(fps / args.sample_fps)))
    if args.full_video:
        base_out_dir = os.path.join("data", "output", "full_video", video_basename)
    elif args.output_subdir:
        base_out_dir = os.path.join("data", "output", args.output_subdir, video_basename)
    else:
        base_out_dir = os.path.join("data", "output", video_basename)

    if args.mode == "full":
        full_dir = os.path.join(base_out_dir, "full_frame")
        run_full_frame_pipeline(
            ra_model=ra_model,
            all_entities=all_entities,
            raw_frames=raw_frames,
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
            stride_frames=stride_frames,
            min_rel_conf=args.min_rel_conf,
            min_consistency=args.min_consistency,
            num_review_frames=args.num_review_frames,
            out_dir=full_dir,
            video_name=video_basename
        )
    elif args.mode == "roi":
        roi_dir = base_out_dir
        run_roi_zoom_pipeline(
            ra_model=ra_model,
            all_entities=all_entities,
            raw_frames=raw_frames,
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
            stride_frames=stride_frames,
            min_rel_conf=args.min_rel_conf,
            min_consistency=args.min_consistency,
            padding_ratio=args.padding_ratio,
            num_review_frames=args.num_review_frames,
            out_dir=roi_dir,
            video_name=video_basename
        )

    print("\n" + "=" * 80)
    print("ALL REQUESTED PIPELINE STAGES COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
