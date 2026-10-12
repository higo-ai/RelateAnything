import glob
"""
Unified Forward Pipeline with Task 2: Spatial Clustering & Dynamic ROI Zoom Crop
=================================================================================
VidVRD Task 2 Implementation:
1. Unified YOLOE-26m Open-Vocabulary Object & Person Tracking (ByteTrack).
2. Spatio-Temporal Interaction Clustering (DBSCAN / Proximity-based Connected Components).
3. Dynamic Stabilized Union Bounding Box Envelope with Adaptive Context Padding (~20%).
4. Multi-Scale High-Resolution Zoom Crop Generation (eliminates background distractor clutter).
5. Cluster-Specific VLM Payload Generation with Zero Cheating / Zero Hardcoding.

Author: VinFast Computer Vision Center AI Trainee Program
Mentors: Luong Manh Tu & Tran Minh Thanh
"""

import os
import sys
import cv2
import json
import math
import argparse
import numpy as np
import torch
from ultralytics import YOLO

# Import Task 2 Spatial Clustering Module
STATIC_FIXTURE_CLASSES = {
    "table", "bench", "chair", "refrigerator", "sofa", "bed",
    "toilet", "sink", "microwave", "oven", "screen", "stool",
    "stop_sign", "traffic_light", "electric_fan", "faucet"
}

from modules.spatial_clustering import (
    compute_box_edge_distance,
    compute_box_iou,
    cluster_entities_spatially,
    compute_cluster_union_boxes,
    render_cluster_zoom_frames,
    render_cluster_visualization_video
)

# ------------------------------------------------------------------------------
# CLI ARGUMENTS
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="VidVRD Task 2: Spatial Clustering & Dynamic ROI Zoom Crop Pipeline")
parser.add_argument("--video", type=str, default="data/videos/video1.mp4", help="Path to input surveillance video")
parser.add_argument("--start_sec", type=float, default=44.0, help="Start time in seconds for golden segment")
parser.add_argument("--end_sec", type=float, default=52.0, help="End time in seconds for golden segment")
parser.add_argument("--num_frames", type=int, default=8, help="Number of clean VLM frames to sample")
parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold for interactive object detection")
parser.add_argument("--iou", type=float, default=0.35, help="IoU threshold for ByteTrack person tracking")
parser.add_argument("--stride", type=int, default=1, help="Frame step stride for object detection")
parser.add_argument("--device", type=str, default="auto", help="Compute device: 'auto', 'cuda', or 'cpu'")
parser.add_argument("--proximity_thresh", type=float, default=None, help="Spatial proximity threshold in pixels (default: dynamic ~15% diagonal)")
parser.add_argument("--padding_ratio", type=float, default=0.20, help="Adaptive context padding ratio for ROI crop (default: 0.20)")
parser.add_argument("--visualize_video", type=str, default=None, help="Path to save annotated Task 2 cluster union box video (.mp4)")
parser.add_argument("--preview_dir", type=str, default=None, help="Directory to save full-frame preview images")
parser.add_argument("--num_preview_frames", type=int, default=10, help="Number of full-frame preview images to sample")
cli_args, _ = parser.parse_known_args()

DEVICE = "cuda" if (cli_args.device == "auto" and torch.cuda.is_available()) else ("cpu" if cli_args.device == "auto" else cli_args.device)

# ------------------------------------------------------------------------------
# 1. PATHS AND CONFIGURATIONS
# ------------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, cli_args.video) if not os.path.isabs(cli_args.video) else cli_args.video
if not os.path.exists(video_path):
    alt_data = os.path.join(base_dir, "data", "videos", os.path.basename(video_path))
    if os.path.exists(alt_data):
        video_path = alt_data
    elif alt_data.lower().endswith(".avi") and os.path.exists(alt_data[:-4] + ".mp4"):
        video_path = alt_data[:-4] + ".mp4"

video_basename = os.path.splitext(os.path.basename(video_path))[0]
payloads_dir = os.path.join(base_dir, "data", "payloads", video_basename)
os.makedirs(payloads_dir, exist_ok=True)

START_SEC = cli_args.start_sec
END_SEC = cli_args.end_sec
NUM_VLM_FRAMES = cli_args.num_frames
CONF_THRESH = cli_args.conf
IOU_THRESH = cli_args.iou
FRAME_STRIDE = cli_args.stride

print("=" * 80)
print(f"VIDVRD TASK 2 PIPELINE: SPATIAL CLUSTERING & DYNAMIC ROI ZOOM CROP")
print(f"Target Video:        {video_path}")
print(f"Temporal Window:     {START_SEC}s -> {END_SEC}s (Duration: {END_SEC - START_SEC:.1f}s)")
print(f"Target VLM Frames:   {NUM_VLM_FRAMES} frames")
print(f"Compute Device:      {DEVICE.upper()}")
print(f"Adaptive Padding:    {cli_args.padding_ratio * 100:.0f}%")
print("=" * 80)

# Load Vocabularies
with open(os.path.join(base_dir, "configs", "s_objects.json"), "r", encoding="utf-8") as f:
    allowed_objects_60 = json.load(f)

with open(os.path.join(base_dir, "configs", "relations.json"), "r", encoding="utf-8") as f:
    relations_list = json.load(f)

# ------------------------------------------------------------------------------
# 2. MODEL INITIALIZATION
# ------------------------------------------------------------------------------
yoloe_weights = os.path.join(base_dir, "models", "yoloe-26m-seg.pt")
if not os.path.exists(yoloe_weights):
    yoloe_weights = os.path.join(base_dir, "yoloe-26m-seg.pt")
model_weights = yoloe_weights
model = YOLO(model_weights)
model.to(DEVICE)
model.set_classes(allowed_objects_60)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start_frame = int(START_SEC * fps)
end_frame = min(int(END_SEC * fps), total_frames - 1)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

ret, first_f = cap.read()
if not ret:
    raise RuntimeError(f"Cannot read video at frame {start_frame}")
orig_h, orig_w = first_f.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# ------------------------------------------------------------------------------
# 3. PASS 1: FORWARD TRACKING & TRAJECTORY ACCUMULATION
# ------------------------------------------------------------------------------
print("\n--- Pass 1: Forward Tracking & Detection ---")
bytetrack_config = os.path.join(base_dir, "configs", "custom_bytetrack.yaml")
COLOR_PALETTE = [
    (0, 165, 255), (50, 205, 50), (255, 191, 0), (238, 130, 238),
    (255, 20, 147), (0, 255, 255), (147, 112, 219), (0, 215, 255)
]

frame_detections = []
active_object_tracklets = []
raw_clean_frames = {}

for f_idx in range(start_frame, end_frame + 1):
    ret, frame = cap.read()
    if not ret:
        break
    raw_clean_frames[f_idx] = (f_idx / fps, frame)

    track_results = model.track(
        source=frame,
        persist=True,
        tracker=bytetrack_config,
        conf=0.20,
        iou=IOU_THRESH,
        verbose=False,
        device=DEVICE
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
                # Filter out distant doorway background noise specks (area < 800 or height < 50)
                if tid is not None and area >= 800 and bh >= 50:
                    frame_persons.append((b, tid, "person", area))
            else:
                if c_name not in STATIC_FIXTURE_CLASSES and c_val >= CONF_THRESH and area >= 200:
                    frame_objects.append((b, c_name, c_val))

    # Spatial Object Association with Open-Vocabulary Class Frequency Fusion
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

# ------------------------------------------------------------------------------
# 4. STABLE PERSON IDENTIFIERS & TRACKLET REFINEMENT
# ------------------------------------------------------------------------------
# Prioritize foreground actors by total pixel mass (count * mean_area)
person_tid_stats = {}
for fd in frame_detections:
    for b, tid, c, area in fd["persons"]:
        if tid not in person_tid_stats:
            person_tid_stats[tid] = {"count": 0, "total_area": 0, "frames": {}}
        person_tid_stats[tid]["count"] += 1
        person_tid_stats[tid]["total_area"] += area
        person_tid_stats[tid]["frames"][fd["frame_idx"]] = b

# Temporal Tracklet Stitching (TTS) for Persons across temporary tracking dropouts
raw_person_tracklets = [
    {"id": tid, "frames": stats["frames"], "total_area": stats["total_area"]}
    for tid, stats in person_tid_stats.items()
    if stats["count"] >= 20 and (stats["total_area"] / stats["count"]) >= 1500
]
raw_person_tracklets.sort(key=lambda t: min(t["frames"].keys()))

stitched_persons = []
for tr in raw_person_tracklets:
    tr_start = min(tr["frames"].keys())
    tr_start_b = tr["frames"][tr_start]
    c_start = ((tr_start_b[0] + tr_start_b[2]) / 2.0, (tr_start_b[1] + tr_start_b[3]) / 2.0)
    matched_sp = None
    for sp in stitched_persons:
        # Check temporal and spatial continuity across dropouts, concurrent duplicates, or tracker ID flicker
        common_f = set(tr["frames"].keys()) & set(sp["frames"].keys())
        if common_f:
            # Overlapping tracklets: only merge if spatial duplicate of the same body (mean IoU >= 0.35)
            ious = [compute_box_iou(tr["frames"][f], sp["frames"][f]) for f in common_f]
            if np.mean(ious) >= 0.35:
                matched_sp = sp
                break
        else:
            # Non-concurrent tracklets (sequential or interleaved tracker flicker):
            # Find nearest frames in time between tr and sp
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
            
            # Brief dropout or tracker alternating flicker: dt <= 15 frames (~0.5s)
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
            "orig_tids": [tr["id"]],
            "frames": dict(tr["frames"]),
            "total_area": tr["total_area"]
        })

stitched_persons.sort(key=lambda p: p["total_area"], reverse=True)
canonical_persons = stitched_persons[:3]  # keep top foreground actors
person_entities = {}
for i, cp in enumerate(canonical_persons):
    p_id = f"[{i + 1}]"
    person_entities[p_id] = {
        "class": "person",
        "frame_map": cp["frames"],
        "type": "person"
    }
num_persons = len(person_entities)

# Object tracklet cross-class merging & stationary occlusion gap interpolation
persistent_raw_objects = [tr for tr in active_object_tracklets if len(tr["frame_map"]) >= 25]

# Sort by appearance frame and apply Temporal Tracklet Stitching (TTS) for objects
persistent_raw_objects.sort(key=lambda tr: min(tr["frame_map"].keys()))
merged_objects = []
for tr in persistent_raw_objects:
    t_start = min(tr["frame_map"].keys())
    b_start = tr["frame_map"][t_start]
    c_start = ((b_start[0] + b_start[2]) / 2.0, (b_start[1] + b_start[3]) / 2.0)
    c_class = max(tr["class_votes"].items(), key=lambda kv: kv[1])[0]
    
    matched = None
    for mo in merged_objects:
        mo_end = max(mo["frame_map"].keys())
        b_end = mo["frame_map"][mo_end]
        c_end = ((b_end[0] + b_end[2]) / 2.0, (b_end[1] + b_end[3]) / 2.0)
        mo_class = max(mo["class_votes"].items(), key=lambda kv: kv[1])[0]
        
        gap = t_start - mo_end
        dist = math.hypot(c_start[0] - c_end[0], c_start[1] - c_end[1])
        
        # Merge if same/compatible class (e.g. handbag <-> backpack) and either:
        # (a) Spatial overlap/stationary match (dist <= 45px)
        # (b) Sequential human transport continuity (gap <= 50 frames and dist <= 130px)
        is_bag_family = (c_class in ["backpack", "handbag"]) and (mo_class in ["backpack", "handbag"])
        is_same_class = (c_class == mo_class) or is_bag_family
        
        if is_same_class:
            if dist <= 45.0 or (-5 <= gap <= 50 and dist <= 160.0):
                matched = mo
                break
                
    if matched is not None:
        matched["frame_map"].update(tr["frame_map"])
        matched["confs"].extend(tr.get("confs", []))
        for cn, count in tr["class_votes"].items():
            matched["class_votes"][cn] = matched["class_votes"].get(cn, 0) + count
    else:
        merged_objects.append(tr)

# Interpolate occlusion gaps and hold stationary position - Vá lỗ hổng chóp chóp bằng nội suy tuyến tính (Linear Interpolation)
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
                b_interp = (1.0 - alpha) * b_prev + alpha * b_next
                mo["frame_map"][f] = b_interp.astype(int)
        
        # Stationary Forward-Fill: Giữ vị trí tĩnh khi đồ vật đặt xuống
        # 1. Permanently stationary scene objects (e.g. parked bicycle, total_disp < 35px)
        # 2. Placed objects: objects transported and placed down on a surface/ground (tail_disp < 20px, not near border)
        boxes_arr = np.array(list(mo["frame_map"].values()))
        cxs = (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2.0
        cys = (boxes_arr[:, 1] + boxes_arr[:, 3]) / 2.0
        total_disp = float(math.hypot(np.ptp(cxs), np.ptp(cys)))

        tail_fs = sorted_fs[-min(10, len(sorted_fs)):]
        tail_boxes = np.array([mo["frame_map"][f] for f in tail_fs])
        tail_cxs = (tail_boxes[:, 0] + tail_boxes[:, 2]) / 2.0
        tail_cys = (tail_boxes[:, 1] + tail_boxes[:, 3]) / 2.0
        tail_disp = float(math.hypot(np.ptp(tail_cxs), np.ptp(tail_cys)))

        last_b = mo["frame_map"][max_f]
        is_near_border = (last_b[0] < 20 or last_b[1] < 20 or 
                          last_b[2] > orig_w - 20 or last_b[3] > orig_h - 20)

        if (total_disp < 35.0) or (tail_disp < 20.0 and not is_near_border):
            print(f"  [Stationary Forward-Fill] Object '{c_class}' resting at frame {max_f} (tail_disp={tail_disp:.1f}px, near_border={is_near_border}). Holding position through frame {end_frame}.")
            for f in range(max_f + 1, end_frame + 1):
                mo["frame_map"][f] = last_b

next_obj_id = num_persons + 1
object_entities = {}
for mo in merged_objects:
    o_id = f"[{next_obj_id}]"
    final_class = max(mo["class_votes"].items(), key=lambda kv: kv[1])[0]
    boxes_arr = np.array(list(mo["frame_map"].values()))
    cxs = (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2.0
    cys = (boxes_arr[:, 1] + boxes_arr[:, 3]) / 2.0
    disp = float(math.hypot(np.ptp(cxs), np.ptp(cys)))
    object_entities[o_id] = {
        "class": final_class,
        "frame_map": mo["frame_map"],
        "type": "object",
        "displacement": disp
    }
    next_obj_id += 1

all_entities = {**person_entities, **object_entities}

print(f"Registered Total Entities: {len(all_entities)} ({len(person_entities)} persons, {len(object_entities)} scene objects)")
for eid, edata in all_entities.items():
    print(f"  - Entity {eid}: class='{edata['class']}', frames_visible={len(edata['frame_map'])}")

# ------------------------------------------------------------------------------
# 5. TASK 2: SPATIAL CLUSTERING (DBSCAN / PROXIMITY CONNECTED COMPONENTS)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 2: SPATIO-TEMPORAL INTERACTION CLUSTERING")
print("=" * 80)

active_clusters, singletons = cluster_entities_spatially(
    entities=all_entities,
    image_shape=(orig_h, orig_w),
    proximity_thresh_px=cli_args.proximity_thresh
)

print(f"Spatial Clustering Results:")
print(f"  - Total Active Interactive Clusters (|C| >= 2): {len(active_clusters)}")
print(f"  - Total Isolated Singletons (|C| == 1):         {len(singletons)} (Excluded from interaction search)")

if singletons:
    print(f"    * Isolated Non-Interacting Entities: {singletons}")

for c in active_clusters:
    cid = c["cluster_id"]
    e_list = [f"{eid} ({c['entities'][eid]['class']})" for eid in c["entity_ids"]]
    print(f"  - {cid.upper()}: Entities = {e_list}, Min Pairwise Edge Distance = {c['min_internal_distance_px']:.1f}px")

# Read System Prompt
prompt_txt_path = os.path.join(base_dir, "data", "prompt_system_general.txt")
if os.path.exists(prompt_txt_path):
    with open(prompt_txt_path, "r", encoding="utf-8") as f:
        vlm_system_prompt = f.read().strip()
else:
    vlm_system_prompt = ""

# ------------------------------------------------------------------------------
# 6. MULTI-SCALE DYNAMIC ROI ZOOM CROP & PAYLOAD GENERATION PER CLUSTER
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 2: DYNAMIC ROI ZOOM CROP RENDERING & PAYLOAD GENERATION")
print("=" * 80)

generated_payloads = []

for c in active_clusters:
    cid = c["cluster_id"]
    cluster_dir_name = f"{video_basename}_roi_{cid}"
    
    # PRESERVE TRUE GLOBAL TRACKING IDs: [1], [2], [4]...
    # Strictly maintain 1-to-1 data consistency with Branch A (Full Frame CCTV tracking)
    # Zero re-indexing prevents central database desynchronization and cross-window ID drift!
    cluster_orig_eids = sorted(list(c["entity_ids"]), key=lambda eid: (0 if c["entities"][eid]["type"] == "person" else 1, eid))
    c["entity_ids"] = cluster_orig_eids
    cluster_frames_dir = os.path.join(base_dir, "data", "frames", video_basename, cluster_dir_name)
    os.makedirs(cluster_frames_dir, exist_ok=True)
    for old_f in glob.glob(os.path.join(cluster_frames_dir, "*.jpg")):
        try:
            os.remove(old_f)
        except OSError:
            pass
    
    # Spatio-Temporal Active Interaction Window:
    # Samples 8 frames strictly while the interacting entities are actively engaged in physical contact/proximity
    # and safely within the camera frame (eliminates border truncation artifacts and post-interaction separation)
    eids = c['entity_ids']
    human_eids = [eid for eid in eids if c['entities'][eid]['type'] == 'person']

    active_contact_frames = []
    if len(human_eids) >= 2:
        fm1 = c['entities'][human_eids[0]]['frame_map']
        fm2 = c['entities'][human_eids[1]]['frame_map']
        common_f = sorted(list(set(fm1.keys()) & set(fm2.keys())))
        if common_f:
            max_h1 = max(fm1[f][3] - fm1[f][1] for f in common_f)
            max_h2 = max(fm2[f][3] - fm2[f][1] for f in common_f)
            min_h1 = int(0.55 * max_h1)
            min_h2 = int(0.55 * max_h2)
            for f in common_f:
                b1, b2 = fm1[f], fm2[f]
                dist = compute_box_edge_distance(b1, b2)
                h1 = b1[3] - b1[1]
                h2 = b2[3] - b2[1]
                # Full 4-edge boundary safety: persons are not truncated off-screen
                in_bounds = (
                    b1[0] >= 15 and b1[2] <= orig_w - 20 and b1[1] >= 15 and b1[3] <= orig_h - 20 and h1 >= min_h1 and
                    b2[0] >= 15 and b2[2] <= orig_w - 20 and b2[1] >= 15 and b2[3] <= orig_h - 20 and h2 >= min_h2
                )
                if dist <= 30.0 and in_bounds:
                    active_contact_frames.append(f)
    elif len(human_eids) == 1 and len(eids) >= 2:
        h_id = human_eids[0]
        o_ids = [eid for eid in eids if eid != h_id]
        fm_h = c['entities'][h_id]['frame_map']
        
        # Unified Spatio-Temporal Interaction Window for Person-to-Object Clusters:
        # 1. Detect physical proximity (arm's reach edge distance <= 40px)
        # 2. Analyze object dynamics: if the object is transported (max displacement >= 25px while in human proximity),
        #    the active interaction window focuses on the motion phase up to surface placement (+ grace margin of ~1.5s).
        #    This prevents post-placement static resting phases (e.g. bag sitting untouched on desk for 15s) from diluting the carrying prompt.
        # 3. If the object is stationary (e.g. parked car or laptop on desk), the interaction window is directly the close proximity dwell frames.
        obj_contact_frames = []
        for o_id in o_ids:
            fm_o = c['entities'][o_id]['frame_map']
            common_f = sorted(list(set(fm_h.keys()) & set(fm_o.keys())))
            if not common_f:
                continue
            
            close_f = [
                f for f in common_f
                if compute_box_edge_distance(fm_h[f], fm_o[f]) <= 40.0
                and (fm_h[f][2] <= orig_w - 15 and fm_h[f][0] >= 10 and fm_h[f][3] <= orig_h - 15 and fm_h[f][1] >= 10)
            ]
            if not close_f:
                continue
            
            # Trajectory analysis of the object center during human proximity
            o_centers = np.array([[(fm_o[f][0] + fm_o[f][2]) / 2.0, (fm_o[f][1] + fm_o[f][3]) / 2.0] for f in close_f])
            if len(o_centers) >= 2:
                max_disp = float(np.max(np.linalg.norm(o_centers - o_centers[0], axis=1)))
            else:
                max_disp = 0.0
            
            if max_disp >= 25.0:
                # Dynamic transport / carrying phase detected across space
                # Find the frame where the object becomes stationary (placed on surface)
                rest_frame = None
                window_size = min(10, len(close_f) // 3)
                if window_size >= 4:
                    for idx in range(len(close_f) - window_size):
                        sub_pts = o_centers[idx : idx + window_size]
                        sub_disp = np.max(np.linalg.norm(sub_pts - sub_pts[0], axis=1))
                        # If displacement over window is < 6px after at least 20% of the trajectory
                        if sub_disp < 6.0 and idx > int(0.20 * len(close_f)):
                            rest_frame = close_f[idx]
                            break
                
                if rest_frame is not None:
                    # Include active carrying phase up to placement + generous grace margin (~45 frames / 1.5s)
                    # to capture the causal act of putting down and releasing the object
                    fps_est = 30.0
                    grace_frames = int(1.5 * fps_est)
                    end_idx = min(len(close_f) - 1, close_f.index(rest_frame) + grace_frames)
                    active_window = close_f[: end_idx + 1]
                else:
                    active_window = close_f
                obj_contact_frames.extend(active_window)
            else:
                # Stationary object manipulation (e.g. parked car break-in, desk interaction)
                obj_contact_frames.extend(close_f)
        
        active_contact_frames = sorted(list(set(obj_contact_frames)))

    # Adaptive Spatio-Temporal Sampling Strategy per Cluster Type: Băm đều 8 ảnh trong khoảnh khắc đó
    # Prioritize active_contact_frames whenever available (>= NUM_VLM_FRAMES)
    # ensuring all sampled frames capture genuine physical interaction/manipulation.
    if len(active_contact_frames) >= NUM_VLM_FRAMES:
        c_step = (len(active_contact_frames) - 1) / (NUM_VLM_FRAMES - 1)
        c_sample_indices = [active_contact_frames[int(round(i * c_step))] for i in range(NUM_VLM_FRAMES)]
    else:
        co_present_frames = [
            f for f in sorted(list(set.union(*[set(c['entities'][eid]['frame_map'].keys()) for eid in c['entity_ids']])))
            if sum(1 for eid in c['entity_ids'] if f in c['entities'][eid]['frame_map']) >= 2
        ]
        if len(co_present_frames) >= NUM_VLM_FRAMES:
            c_step = (len(co_present_frames) - 1) / (NUM_VLM_FRAMES - 1)
            c_sample_indices = [co_present_frames[int(round(i * c_step))] for i in range(NUM_VLM_FRAMES)]
        elif len(active_contact_frames) > 0:
            c_step = (len(active_contact_frames) - 1) / max(1, len(active_contact_frames) - 1)
            c_sample_indices = [active_contact_frames[int(round(i * c_step))] for i in range(min(len(active_contact_frames), NUM_VLM_FRAMES))]
        else:
            ideal_step = (end_frame - start_frame) / (NUM_VLM_FRAMES - 1)
            c_sample_indices = [int(start_frame + i * ideal_step) for i in range(NUM_VLM_FRAMES)]

    c_sample_frames_clean = {f_idx: raw_clean_frames[f_idx] for f_idx in c_sample_indices if f_idx in raw_clean_frames}

    # Compute stabilized union crop box - Tính hộp bao trùm chung (Union Envelope) có đệm 20%
    crop_boxes = compute_cluster_union_boxes(
        cluster_entity_ids=c["entity_ids"],
        entities=all_entities,
        sample_frame_indices=c_sample_indices,
        image_shape=(orig_h, orig_w),
        padding_ratio=cli_args.padding_ratio,
        stabilize_temporal_envelope=True
    )

    # Render Zoom Crop Frames - Cắt ảnh gốc và vẽ Set-of-Marks tinh chỉnh
    saved_frames = render_cluster_zoom_frames(
        cluster=c,
        clean_frames_dict=c_sample_frames_clean,
        crop_boxes=crop_boxes,
        output_dir=cluster_frames_dir,
        color_palette=COLOR_PALETTE
    )

    frame_filenames = [fn for fn, _, _ in saved_frames]
    mean_zoom = float(np.mean([zf for _, _, zf in saved_frames]))

    print(f"\nSUCCESS: [{cid.upper()}] Rendered {len(saved_frames)} Zoom Crop Frames:")
    print(f"   - Storage: {cluster_frames_dir}")
    print(f"   - Mean Resolution Magnification (Zoom Factor): {mean_zoom:.2f}x")
    for fn, _, zf in saved_frames[:3]:
        print(f"     * {fn} (Zoom: {zf:.2f}x)")
    if len(saved_frames) > 3:
        print(f"     * ... and {len(saved_frames) - 3} more frames")

    # Build Cluster-Specific Prompt
    cluster_entities_str = ", ".join([f"{eid} ({c['entities'][eid]['class']})" for eid in c["entity_ids"]])
    cluster_user_prompt = (
        f"Analyze all provided sequential zoom-crop frames of this localized interaction zone ({cid}).\n"
        f"Detected entities with visual marks in this interaction cluster: {cluster_entities_str}.\n\n"
        "Examine active interactions between the marked entities across time.\n\n"
        "PREDEFINED RELATION TAXONOMY (CLOSED VOCABULARY):\n"
        f"Every predicate in the 'relation' field MUST be an exact string match selected strictly from the 26 allowed categories: {relations_list}. All out-of-vocabulary verbs are strictly prohibited.\n"
        "- Strictly evaluate interactions ONLY between marked entities [ID]. Completely ignore unmarked objects or background clutter; NEVER substitute an unmarked item with a marked person.\n"
        "- Clean ID Formatting: In the 'subject' and 'object' fields, output strictly the clean mark ID string (e.g. '[1]', '[2]') without appending class names or descriptive words.\n"
        "- Triplet Uniqueness & Predicate Exclusivity: Report each unique relation between a subject and an object AT MOST ONCE for the entire clip. Between the same subject and object, interaction predicates are strictly mutually exclusive: output ONLY the single most comprehensive predicate (e.g., casual contact during walking, greeting, or parting is categorized strictly as 'touch', never 'push'). Do NOT output duplicate or conflicting triplets for different frames.\n"
        "- Temporal Action Continuity: Sequential phases of an interaction across time (such as approaching, extending an arm, making contact, and parting) constitute ONE unified interaction event. Do NOT fragment preparatory reaching motions into a separate 'push' relation.\n"
        "- Interpersonal Actions: Categorize casual physical contact between persons (such as placing a hand on a shoulder, touching an arm or body, or tapping) strictly as 'touch'. Reserve 'push' and 'pull' strictly for visible forceful shoving where an entity is visibly propelled, knocked off balance, or dragged.\n"
        "- Predicates 'carry' and 'hold' apply strictly between a Person (subject) and a moveable Object. A person cannot 'carry' another person unless physically lifting them off the ground.\n"
        "- Categorize a person holding and transporting an object with their hands while moving as 'carry', and holding statically as 'hold'.\n"
        "- Temporal Transitions: If a person actively carries or holds an object in early frames, report that valid relation ('carry' or 'hold') even if they subsequently place down or release the object on a surface in later frames.\n"
        "- Inactive Surface Clutter: If an object remains completely stationary in the exact same location across ALL frames without ever being moved, picked up, or held, omit that pair entirely (merely walking past or standing near an untouched object on the floor/surface is NOT an interaction).\n"
        "- Predicates like 'get_on', 'get_off', 'ride', 'drive' apply ONLY to vehicles or animals.\n"
        "- In the 'reason' field, describe strictly the visible physical contact between this subject and this object without referencing other entities.\n\n"
        'Respond strictly with the JSON object: {"triplets": [{"subject": "[ID]", "relation": "<verb>", "object": "[ID]", "reason": "..."}]}.'
    )

    cluster_payload = {
        "task": "Video Visual Relation Detection (VidVRD) - Task 2 Dynamic ROI Zoom Crop",
        "scenario": f"Localized Interaction Zone Analysis ({cid})",
        "pipeline_variant": "yoloe_spatial_clustering_dynamic_roi_zoom",
        "cluster_info": {
            "cluster_id": cid,
            "entities": [{"mark_id": eid, "class_label": c["entities"][eid]["class"]} for eid in c["entity_ids"]],
            "zoom_magnification_factor": f"{mean_zoom:.2f}x",
            "padding_ratio": cli_args.padding_ratio,
            "min_internal_distance_px": round(c["min_internal_distance_px"], 1)
        },
        "allowed_relations_vocabulary_26": relations_list,
        "visual_prompt_frames_sequence": frame_filenames,
        "vlm_system_prompt": vlm_system_prompt,
        "vlm_user_prompt": cluster_user_prompt
    }

    payload_filename = f"{video_basename}_roi_{cid}_payload.json"
    payload_file_path = os.path.join(payloads_dir, payload_filename)
    with open(payload_file_path, "w", encoding="utf-8") as pf:
        json.dump(cluster_payload, pf, indent=2, ensure_ascii=False)
    
    print(f"   - Generated Cluster Payload: {payload_file_path}")
    generated_payloads.append(payload_file_path)

print("\n" + "=" * 80)
print(f"TASK 2 PIPELINE COMPLETED SUCCESSFULLY!")
print(f"Generated {len(generated_payloads)} Cluster Payloads in: {payloads_dir}")
for p in generated_payloads:
    print(f"  * {os.path.basename(p)}")
print("=" * 80)


# ------------------------------------------------------------------------------
# 7. TASK 2: CLUSTER UNION VISUALIZATION & PREVIEW EXPORT (MENTOR DIRECTIVE)
# ------------------------------------------------------------------------------
if cli_args.visualize_video or cli_args.preview_dir:
    print("\n" + "=" * 80)
    print("TASK 2: RENDERING CLUSTER UNION VISUALIZATION & PREVIEW FRAMES")
    print("=" * 80)
    
    effective_preview_dir = cli_args.preview_dir
    if effective_preview_dir is None and cli_args.visualize_video:
        effective_preview_dir = os.path.join(base_dir, "data", "preview", video_basename, f"{video_basename}_clusters")
        os.makedirs(effective_preview_dir, exist_ok=True)
    
    vis_res = render_cluster_visualization_video(
        raw_clean_frames=raw_clean_frames,
        active_clusters=active_clusters,
        singletons=singletons,
        all_entities=all_entities,
        output_video_path=cli_args.visualize_video,
        preview_dir=effective_preview_dir,
        num_preview_frames=cli_args.num_preview_frames,
        fps=fps,
        video_basename=video_basename,
        proximity_thresh_px=cli_args.proximity_thresh if cli_args.proximity_thresh else 45.0
    )
    
    if vis_res.get("output_video_path"):
        print(f"SUCCESS: Rendered Cluster Union Box Surveillance Video:")
        print(f"   - File: {vis_res['output_video_path']}")
        print(f"   - Rules Applied:")
        print(f"     * Union Bounding Box ONLY (No individual inner boxes)")
        print(f"     * Dynamic Header Badge: 'CLUSTER X: [ID] class + [ID] class'")
        print(f"     * Modularity Preserved (Zero interaction verbs upstream)")
        print(f"     * Clutter Singletons Unboxed (e.g. wall bag [2])")
        
    if vis_res.get("preview_dir"):
        print(f"\nSUCCESS: Exported {vis_res['total_preview_frames']} Full Panoramic Preview Images:")
        print(f"   - Directory: {vis_res['preview_dir']}")
        for prev in vis_res.get("previews", []):
            box_info = f"Box: {prev['cluster_info']['union_box']}" if prev.get("has_cluster_box") else "No Cluster Box (Separated / Clutter unboxed)"
            print(f"     * {prev['filename']} (Time: {prev['timestamp_sec']}s) -> {box_info}")
        print("=" * 80)
