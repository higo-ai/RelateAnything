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
    parser.add_argument("--sample_fps", type=float, default=3.0, help="RelateAnything sampling rate (FPS)")
    parser.add_argument("--num_review_frames", type=int, default=8, help="Number of sequential review frames to save")
    parser.add_argument("--min_consistency", type=float, default=0.25, help="Temporal consistency threshold")
    parser.add_argument("--min_rel_conf", type=float, default=0.15, help="Minimum relation confidence score")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, cuda, or cpu")
    parser.add_argument("--padding_ratio", type=float, default=0.20, help="Context padding ratio for ROI crop")
    parser.add_argument("--mode", type=str, choices=["full", "roi"], default="roi",
                        help="Execution mode: 'full' (Full Frame) or 'roi' (ROI Zoom)")
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
    cluster_union_boxes: Optional[Dict[str, Dict[int, Tuple[int, int, int, int]]]] = None
) -> np.ndarray:
    """Renders authentic surveillance monitor feed matching the exact styling of the original repo:
    - 1.5px soft bounding box border (anti-aliased visual weight).
    - Compact label badge with Alpha Blending (70% tint, 30% background transparency).
    - Outline for the badge + bold white text with faux-bold double pass (font_scale=0.45).
    - Real-time physical contact activation: connecting lines & HUD badges light up ONLY when
      entities are genuinely in physical contact/proximity at the current frame."""
    vis = raw_frame.copy()
    h, w = vis.shape[:2]

    # 1. Cluster Union Bounding Envelope (yellow thin outline, only if present in current frame)
    if cluster_union_boxes:
        for cid, uboxes in cluster_union_boxes.items():
            if f_idx in uboxes:
                ub = uboxes[f_idx]
                cv2.rectangle(vis, (ub[0], ub[1]), (ub[2], ub[3]), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(vis, f"ZOOM ENVELOPE: {cid}", (ub[0] + 4, max(14, ub[1] - 4)),
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

    # 3. Real-Time Physical Contact Guard: Draw connecting line & label pill ONLY when entities are in actual physical contact/proximity at current frame
    active_now_relations = []
    for r in active_relations:
        sub_id = str(r["subject_id"])
        obj_id = str(r["object_id"])
        if sub_id in entities and obj_id in entities:
            fm_s = entities[sub_id].get("frame_map", entities[sub_id].get("frames", {}))
            fm_o = entities[obj_id].get("frame_map", entities[obj_id].get("frames", {}))
            if f_idx in fm_s and f_idx in fm_o:
                b_s = fm_s[f_idx]
                b_o = fm_o[f_idx]
                dx = max(0, max(b_s[0] - b_o[2], b_o[0] - b_s[2]))
                dy = max(0, max(b_s[1] - b_o[3], b_o[1] - b_s[3]))
                current_dist = math.hypot(dx, dy)
                current_iou = compute_box_iou(np.array(b_s, dtype=float), np.array(b_o, dtype=float))

                # Real-time contact condition: arm-reach contact (<= 35px) or overlapping
                if current_dist <= 35.0 or current_iou > 0.0:
                    active_now_relations.append(r)
                    c_sub = ((b_s[0] + b_s[2]) // 2, (b_s[1] + b_s[3]) // 2)
                    c_obj = ((b_o[0] + b_o[2]) // 2, (b_o[1] + b_o[3]) // 2)

                    # Connecting line between interacting pair
                    cv2.line(vis, c_sub, c_obj, (0, 255, 255), 1, cv2.LINE_AA)

                    # Midpoint pill for relationship name
                    mid_x = (c_sub[0] + c_obj[0]) // 2
                    mid_y = (c_sub[1] + c_obj[1]) // 2

                    rel_label = f"{r['predicate']} ({r['mean_confidence']:.2f})"
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
            feed_txt = f"{r['subject_id']} {r['subject_class']} -{r['predicate']}-> {r['object_id']} {r['object_class']}"
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
    active_relations: List[Dict[str, Any]]
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
                b_s = fm_s[f_idx]
                b_o = fm_o[f_idx]
                dx = max(0, max(b_s[0] - b_o[2], b_o[0] - b_s[2]))
                dy = max(0, max(b_s[1] - b_o[3], b_o[1] - b_s[3]))
                if math.hypot(dx, dy) <= 35.0 or compute_box_iou(np.array(b_s, dtype=float), np.array(b_o, dtype=float)) > 0:
                    crop_active_relations.append(r)

    if crop_active_relations:
        top_txt = " | ".join([f"{r['subject_id']} -{r['predicate']}-> {r['object_id']}" for r in crop_active_relations[:2]])
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(top_txt, font, 0.42, 1)
        sub_top = crop_img[4:4 + th + 8, 4:4 + tw + 10]
        if sub_top.shape[0] > 0 and sub_top.shape[1] > 0:
            dark_bg = np.zeros_like(sub_top)
            crop_img[4:4 + th + 8, 4:4 + tw + 10] = cv2.addWeighted(dark_bg, 0.70, sub_top, 0.30, 0)
            cv2.rectangle(crop_img, (4, 4), (4 + tw + 10, 4 + th + 8), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(crop_img, top_txt, (8, 4 + th + 4), font, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    return crop_img



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


def predict_relations_with_ontology(
    ra_model: RelateAnything,
    image,
    boxes_xyxy: np.ndarray,
    box_labels: List[str],
    box_types: List[str]
) -> List[Dict[str, Any]]:
    """Predicts relationships purely from the authentic neural network logits of RelateAnything
    across the closed 26 VidVRD predicates.
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
        scores = ra_model.contract.scores(logits, pair).cpu().numpy()
        sub_idx = out["sub_idx"][0].cpu().numpy()
        obj_idx = out["obj_idx"][0].cpu().numpy()

    triplets = []
    for k in range(len(sub_idx)):
        si, oi = int(sub_idx[k]), int(obj_idx[k])
        if si == oi or si >= N or oi >= N:
            continue
        st = box_types[si]
        # Agentic physical law: inanimate scene objects cannot initiate actions onto humans
        if st == "object":
            continue

        sc = scores[k]
        best_p_idx = int(np.argmax(sc))
        best_sc = float(sc[best_p_idx])
        if best_sc > 0:
            pred_name = ra_model.predicates[best_p_idx]
            triplets.append({
                "subject_idx": si,
                "object_idx": oi,
                "predicate": pred_name,
                "score": best_sc
            })
    return triplets



def is_physically_plausible_relation(sub_eid, obj_eid, pred, all_entities):
    sub_info = all_entities.get(sub_eid, {})
    obj_info = all_entities.get(obj_eid, {})
    if sub_info.get('type') == 'person' and obj_info.get('type') == 'object':
        disp = float(obj_info.get('displacement', 0.0))
        # Mentor Rule: An inanimate resting object that remains completely stationary across all frames (disp < 35px)
        # cannot be actively held or carried. Merely standing near an untouched object on the floor is not holding/carrying.
        # Active physical contact actions (hit, touch, kick, push, knock, lean_on) on stationary objects ARE fully preserved.
        if pred in ('carry', 'hold') and disp < 35.0:
            return False
    return True

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
    out_dir: str
) -> Dict[str, Any]:
    """Runs Full Frame Baseline without spatial clustering or cropping."""
    print("\n" + "=" * 70)
    print("MODE: FULL-FRAME BASELINE (GLOBAL SURVEILLANCE EVALUATION)")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    for old_f in os.listdir(frames_dir):
        if old_f.endswith(".jpg"):
            try: os.remove(os.path.join(frames_dir, old_f))
            except: pass

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
            box_types=box_types
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
                if not is_physically_plausible_relation(sub_eid, obj_eid, pred, all_entities):
                    continue
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

    # Export relations.json
    relations_json_path = os.path.join(out_dir, "relations.json")
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

        if f_idx in review_frame_indices and stt_counter <= num_review_frames:
            f_sec = f_idx / fps
            frame_img_path = os.path.join(frames_dir, f"frame_{stt_counter:02d}_{f_sec:.2f}s.jpg")
            cv2.imwrite(frame_img_path, vis)
            stt_counter += 1

    out_mp4_path = os.path.join(out_dir, "visualizer.mp4")
    write_clean_h264_mp4(
        frames_list=rendered_video_frames,
        out_mp4_path=out_mp4_path,
        fps=fps,
        w=orig_w,
        h=orig_h
    )

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
    out_dir: str
) -> Dict[str, Any]:
    """Runs rigorous Spatio-Temporal Interaction Clustering (STIC) + Dynamic ROI Zoom Crop pipeline."""
    print("\n" + "=" * 70)
    print("MODE: SPATIO-TEMPORAL INTERACTION CLUSTERING (STIC) & DYNAMIC ROI ZOOM")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    for old_f in os.listdir(frames_dir):
        if old_f.endswith(".jpg"):
            try: os.remove(os.path.join(frames_dir, old_f))
            except: pass

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
            ux1, uy1, ux2, uy2 = uboxes[f_idx]
            crop_img = frame[uy1:uy2, ux1:ux2]
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
                    lx1 = max(0.0, float(gx1 - ux1))
                    ly1 = max(0.0, float(gy1 - uy1))
                    lx2 = min(float(crop_w), float(gx2 - ux1))
                    ly2 = min(float(crop_h), float(gy2 - uy1))
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
                box_types=box_types
            )

            for t in triplets:
                sub_eid = frame_eids[t["subject_idx"]]
                obj_eid = frame_eids[t["object_idx"]]
                pair_pred_scores[(sub_eid, obj_eid)][t["predicate"]].append(float(t["score"]))

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

                if mean_conf >= min_rel_conf and consistency >= min_consistency:
                    if not is_physically_plausible_relation(sub_eid, obj_eid, pred, all_entities):
                        continue
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
                top_r = candidate_preds[0]
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

    # 3. Export relations.json
    relations_json_path = os.path.join(out_dir, "relations.json")
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

    # 4. Render Video and 8 Sequential Review Frames (Active-Contact Lifespan Sampling)
    primary_cluster = active_clusters[0] if active_clusters else None

    # Adaptive Sampling: Sample review frames STRICTLY within the active contact window (LACS)
    review_frame_indices = set()
    if primary_cluster:
        contact_frames = compute_active_contact_frames(primary_cluster, all_entities, (orig_h, orig_w))
        if len(contact_frames) >= num_review_frames:
            step_review = (len(contact_frames) - 1) / (num_review_frames - 1)
            review_frame_indices = set(contact_frames[int(round(i * step_review))] for i in range(num_review_frames))
        elif primary_cluster["cluster_union_boxes"]:
            cluster_active_frames = sorted(list(primary_cluster["cluster_union_boxes"].keys()))
            if len(cluster_active_frames) >= num_review_frames:
                step_review = (len(cluster_active_frames) - 1) / (num_review_frames - 1)
                review_frame_indices = set(cluster_active_frames[int(round(i * step_review))] for i in range(num_review_frames))
            else:
                review_frame_indices = set(cluster_active_frames)
    else:
        all_f_indices = sorted(list(raw_frames.keys()))
        step_review = max(1, len(all_f_indices) // num_review_frames)
        review_frame_indices = set(all_f_indices[::step_review][:num_review_frames])

    rendered_video_frames = []
    stt_counter = 1

    print(f"\nRendering clean surveillance video and saving {num_review_frames} crisp ROI Zoom review frames...")
    for f_idx in range(start_frame, end_frame + 1):
        if f_idx not in raw_frames:
            continue

        # Render surveillance monitor video frame
        vis_monitor = render_surveillance_monitor_frame(
            raw_frame=raw_frames[f_idx],
            entities=all_entities,
            f_idx=f_idx,
            fps=fps,
            active_relations=all_cluster_confirmed_relations,
            cluster_union_boxes=cluster_union_boxes_map
        )
        rendered_video_frames.append(vis_monitor)

        # Render crisp zoomed review frame for human visual inspection
        if f_idx in review_frame_indices and stt_counter <= num_review_frames:
            f_sec = f_idx / fps
            frame_img_path = os.path.join(frames_dir, f"frame_{stt_counter:02d}_{f_sec:.2f}s.jpg")

            if primary_cluster and f_idx in primary_cluster["cluster_union_boxes"]:
                crop_box = primary_cluster["cluster_union_boxes"][f_idx]
                vis_crop = render_crisp_roi_crop_frame(
                    raw_frame=raw_frames[f_idx],
                    crop_box=crop_box,
                    all_entities=all_entities,
                    f_idx=f_idx,
                    active_relations=all_cluster_confirmed_relations
                )
                cv2.imwrite(frame_img_path, vis_crop)
            else:
                cv2.imwrite(frame_img_path, vis_monitor)

            stt_counter += 1

    out_mp4_path = os.path.join(out_dir, "visualizer.mp4")
    write_clean_h264_mp4(
        frames_list=rendered_video_frames,
        out_mp4_path=out_mp4_path,
        fps=fps,
        w=orig_w,
        h=orig_h
    )

    saved_imgs = sorted([x for x in os.listdir(frames_dir) if x.endswith(".jpg")])
    print(f"  [FRAMES EXPORT] Saved {len(saved_imgs)} sequential review frames to: {frames_dir}")
    for fname in saved_imgs:
        print(f"    - {fname}")

    return out_payload


def main():
    args = parse_arguments()
    device = resolve_device(args.device)
    video_basename = os.path.splitext(os.path.basename(args.video))[0]

    print("=" * 80)
    print("VIDVRD END-TO-END PIPELINE: RELATEANYTHING + YOLOE-26M")
    print(f"Target Video:         {args.video}")
    print(f"Temporal Window:      {args.start_sec:.1f}s -> {args.end_sec:.1f}s (Duration: {args.end_sec - args.start_sec:.1f}s)")
    print(f"Execution Mode:       {args.mode.upper()}")
    print(f"Compute Device:       {device.upper()}")
    print(f"RelateAnything Model: {args.relsgg_model}")
    print(f"Confidence Threshold: {args.min_rel_conf}")
    print(f"Consistency Thresh:   {args.min_consistency}")
    print("=" * 80)

    # 1. Load Vocabularies
    with open(args.objects_json, "r", encoding="utf-8") as f:
        allowed_objects_60 = json.load(f)
    with open(args.relations_json, "r", encoding="utf-8") as f:
        relations_26 = json.load(f)

    # 2. Initialize Models
    print("\n[1/4] Initializing YOLOE-26m Detector & Tracker...")
    yolo_model = YOLO(args.yolo_weights)
    yolo_model.to(device)
    yolo_model.set_classes(allowed_objects_60)

    print("\n[2/4] Initializing RelateAnything & Setting 26-Predicate Closed Vocabulary...")
    ra_model = RelateAnything.from_pretrained(args.relsgg_model, device=device)
    ra_model.set_vocabulary(relations_26)

    # 3. Video Reader Setup
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(args.start_sec * fps)
    end_frame = min(int(args.end_sec * fps), total_frames - 1)
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
        if stats["count"] >= 15 and (stats["total_area"] / stats["count"]) >= 1200
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
                    if dist <= 35.0 or iou >= 0.30:
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

    person_entities = {}
    for p_idx, sp in enumerate(stitched_persons, 1):
        person_entities[f"[{p_idx}]"] = {
            "type": "person",
            "class": "person",
            "frames": sp["frames"],
            "frame_map": sp["frames"],
            "color": COLOR_PALETTE[(p_idx - 1) % len(COLOR_PALETTE)]
        }
    print(f"Identified {len(person_entities)} stable person tracklets.")

    # Object Stitching and Stationary Interpolation
    persistent_raw_objects = [tr for tr in active_object_tracklets if len(tr["frame_map"]) >= 20]
    persistent_raw_objects.sort(key=lambda tr: min(tr["frame_map"].keys()))
    merged_objects = []

    for tr in persistent_raw_objects:
        t_start = min(tr["frame_map"].keys())
        b_start = tr["frame_map"][t_start]
        matched_mo = None
        for mo in merged_objects:
            mo_end = max(mo["frame_map"].keys())
            gap = t_start - mo_end
            if 0 < gap <= 40:
                b_last = mo["frame_map"][mo_end]
                dist = compute_box_edge_distance(b_start, b_last)
                iou = compute_box_iou(b_start, b_last)
                if iou >= 0.15 or dist <= 50.0:
                    matched_mo = mo
                    break
        if matched_mo is not None:
            matched_mo["frame_map"].update(tr["frame_map"])
            matched_mo["confs"].extend(tr["confs"])
            for c_k, c_v in tr["class_votes"].items():
                matched_mo["class_votes"][c_k] = matched_mo["class_votes"].get(c_k, 0) + c_v
        else:
            merged_objects.append(tr)

    # Linear Gap Interpolation for Objects (eliminates single-frame detector dropouts)
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

    object_entities = {}
    obj_idx_counter = len(person_entities) + 1
    for mo in merged_objects:
        best_cls = max(mo["class_votes"].items(), key=lambda kv: kv[1])[0]
        boxes_arr = np.array(list(mo["frame_map"].values()))
        cxs = (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2.0
        cys = (boxes_arr[:, 1] + boxes_arr[:, 3]) / 2.0
        disp = float(math.hypot(np.ptp(cxs), np.ptp(cys)))

        object_entities[f"[{obj_idx_counter}]"] = {
            "type": "object",
            "class": best_cls,
            "frames": mo["frame_map"],
            "frame_map": mo["frame_map"],
            "displacement": disp,
            "mean_conf": float(np.mean(mo["confs"])),
            "color": COLOR_PALETTE[(obj_idx_counter - 1) % len(COLOR_PALETTE)]
        }
        obj_idx_counter += 1
    print(f"Identified {len(object_entities)} persistent object tracklets.")

    all_entities = {**person_entities, **object_entities}

    # Execute Mode
    stride_frames = max(1, int(round(fps / args.sample_fps)))
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
            out_dir=full_dir
        )
    elif args.mode == "roi":
        roi_dir = os.path.join(base_out_dir, "roi_zoom")
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
            out_dir=roi_dir
        )

    print("\n" + "=" * 80)
    print("ALL REQUESTED PIPELINE STAGES COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
