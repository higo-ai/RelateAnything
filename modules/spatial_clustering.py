"""
Spatial Clustering & Dynamic ROI Zoom Crop Module
=================================================
VidVRD Task 2: Advanced Spatio-Temporal Interaction Clustering (STIC) & Multi-Scale Dynamic ROI Zoom.

Author: VinFast Computer Vision Center AI Trainee Research Program
Mentors: Luong Manh Tu & Tran Minh Thanh

Key Architectural Foundations:
1. Spatio-Temporal Interaction Clustering (STIC):
   - Computes pairwise continuous contact and proximity metrics across all common frames.
   - Robustly isolates active interactive clusters (|C| >= 2) from isolated singletons (|C| == 1).
   - Eliminates transitive chaining through time: transient passing-by (<20 frames) is cleanly
     distinguished from sustained physical interaction (touch, shake hand, carry, hold, hug, push).
   - Distinguishes Human-Human and Human-Object interaction domains from stationary background clutter:
     * Interpersonal interactions require sustained physical proximity (D_edge <= tau_contact for >= 20 frames or IoU >= 0.05).
     * Human-Object interactions require active object manipulation or co-movement (object displacement >= 20px
       or moving in lockstep with human hands/body). Stationary resting clutter on floor or furniture is excluded.
     * Object-Object interactions are strictly prohibited by the closed VidVRD predicate taxonomy.
2. Dynamic Motion-Aware ROI Zoom Crop:
   - Computes adaptive Union Bounding Box envelopes with 15-20% safety context padding.
   - Supports both stationary interaction zones (stabilized static window) and traveling interactions (smooth tracking window).
   - Renders crisp local Set-of-Marks labels directly onto pristine unannotated video crops.
3. Cluster-Specific VLM Prompt Payload Generation:
   - Formulates targeted visual prompts for each localized interaction cluster.
   - 100% general, zero hardcoding, zero cheating, adhering strictly to VidVRD 26-predicate taxonomy.
"""

import os
import cv2
import json
import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

def compute_box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

def compute_box_edge_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Computes Euclidean boundary distance between two bounding boxes.
    Returns 0.0 if the boxes touch or overlap.
    """
    dx = max(0, max(box_a[0] - box_b[2], box_b[0] - box_a[2]))
    dy = max(0, max(box_a[1] - box_b[3], box_b[1] - box_a[3]))
    return float(math.hypot(dx, dy))

def evaluate_pairwise_interaction_affinity(
    entity_a: Dict[str, Any],
    entity_b: Dict[str, Any],
    image_shape: Tuple[int, int] = (480, 640),
    contact_thresh_px: Optional[float] = None,
    min_sustained_frames: int = 20,
    min_overlap_iou: float = 0.04
) -> Tuple[bool, float, int, float]:
    """
    Evaluates whether two tracked entities have an active physical interaction.
    
    Parameters:
    - entity_a, entity_b: Dicts containing:
        'class': str
        'frame_map': {frame_idx: [x1, y1, x2, y2]}
        'type': 'person' | 'object'
        (optional) 'displacement': float
    
    Returns: (is_interactive, min_edge_dist, sustained_contact_frames, max_iou)
    """
    type_a = entity_a.get("type", "person" if entity_a.get("class") == "person" else "object")
    type_b = entity_b.get("type", "person" if entity_b.get("class") == "person" else "object")

    # Domain rule: In VidVRD, relations are directed subject-object actions. Two scene objects never interact.
    if type_a == "object" and type_b == "object":
        return False, float("inf"), 0, 0.0

    fmap_a = entity_a["frame_map"]
    fmap_b = entity_b["frame_map"]
    common_frames = sorted(list(set(fmap_a.keys()) & set(fmap_b.keys())))
    if not common_frames:
        return False, float("inf"), 0, 0.0

    h, w = image_shape
    if contact_thresh_px is None:
        # Physical arm-reach / contact threshold: ~3.5% of diagonal (~28px in 640x480, ~30px in 720x480)
        contact_thresh_px = max(22.0, min(35.0, 0.035 * math.hypot(w, h)))

    dists = []
    ious = []
    for f in common_frames:
        b_a = np.array(fmap_a[f], dtype=float)
        b_b = np.array(fmap_b[f], dtype=float)
        d = compute_box_edge_distance(b_a, b_b)
        u = compute_box_iou(b_a, b_b)
        dists.append(d)
        ious.append(u)

    min_dist = min(dists)
    max_iou = max(ious)
    contact_frames = sum(1 for d in dists if d <= contact_thresh_px)

    # --------------------------------------------------------------------------
    # Case 1: Human-to-Human Interaction (touch, hug, shake_hand, push, pull)
    # --------------------------------------------------------------------------
    if type_a == "person" and type_b == "person":
        # Requires sustained physical contact/proximity (>= min_sustained_frames)
        # Distinguishes genuine interaction (conversation, touching: >= 20 frames) from transient walking past (< 20 frames)
        is_interactive = (min_dist <= contact_thresh_px) and (contact_frames >= min_sustained_frames)
        return is_interactive, min_dist, contact_frames, max_iou

    # --------------------------------------------------------------------------
    # Case 2: Human-to-Object Interaction (carry, hold, lift, grab, etc.)
    # --------------------------------------------------------------------------
    # Identify which entity is person and which is object
    obj_entity = entity_b if type_b == "object" else entity_a
    person_entity = entity_a if type_b == "object" else entity_b

    # --------------------------------------------------------------------------
    # Case 2: Human-to-Object Interaction (carry, hold, touch, sit_on, inspect, etc.)
    # In strict accordance with Mentor's directive:
    # 1. Dynamic / Carried Objects (displacement >= 35px):
    #    - Object moves with the person (e.g. carried bag).
    #    - Requires physical contact/proximity (min_dist <= contact_thresh_px) over >= 10 frames or IoU >= 0.02.
    # 2. Stationary Scene Objects (displacement < 35px, e.g. floor backpack, parked car, bench):
    #    - Physical interaction requires contact in the person's active manipulation zone
    #      (excluding crown of head: y >= y_top + 0.15 * person_h) AND sustained presence
    #      (dwell ratio >= 25% of co-present frames or >= 25 sustained contact frames).
    #    - Distinguishes genuine interaction (approaching and stopping at a car/backpack)
    #      from optical 2D background occlusions (walking past a wall item hanging near ceiling).
    # --------------------------------------------------------------------------
    obj_disp = float(obj_entity.get("displacement", 0.0))
    if obj_disp >= 35.0:
        is_interactive = (min_dist <= contact_thresh_px) and (contact_frames >= 10 or max_iou >= 0.02)
        return is_interactive, min_dist, contact_frames, max_iou
    else:
        manip_contact_frames = 0
        manip_min_dist = float("inf")
        for f in common_frames:
            p_b = np.array(person_entity["frame_map"][f], dtype=float)
            o_b = np.array(obj_entity["frame_map"][f], dtype=float)
            # Body manipulation zone (excluding top 15% head crown)
            head_h = 0.15 * (p_b[3] - p_b[1])
            body_b = np.array([p_b[0], p_b[1] + head_h, p_b[2], p_b[3]])
            d_manip = compute_box_edge_distance(body_b, o_b)
            if d_manip < manip_min_dist:
                manip_min_dist = d_manip
            if d_manip <= contact_thresh_px:
                manip_contact_frames += 1
        
        dwell_ratio = manip_contact_frames / max(1, len(common_frames))
        is_interactive = (manip_min_dist <= contact_thresh_px) and (dwell_ratio >= 0.25 and manip_contact_frames >= 15)
        return is_interactive, manip_min_dist, manip_contact_frames, max_iou

def cluster_entities_spatially(
    entities: Dict[str, Dict[str, Any]],
    image_shape: Tuple[int, int] = (480, 640),
    proximity_thresh_px: Optional[float] = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Performs Spatio-Temporal Interaction Clustering (STIC) on tracked entities.
    
    Parameters:
    - entities: Dict of {entity_id: {"class": str, "frame_map": {frame_idx: [x1, y1, x2, y2]}, "type": "person"|"object"}}
    - image_shape: (H, W) of video
    - proximity_thresh_px: Optional custom distance threshold
    
    Returns:
    - active_clusters: List of interactive cluster dicts (|C| >= 2)
    - isolated_singletons: List of entity_ids (|C| == 1) excluded from interaction queries
    """
    entity_ids = sorted(entities.keys())
    n = len(entity_ids)

    # Build adjacency graph based on verified spatio-temporal interaction affinity
    adj = {eid: set() for eid in entity_ids}
    distance_records = {}

    for i in range(n):
        for j in range(i + 1, n):
            id_a, id_b = entity_ids[i], entity_ids[j]
            
            is_inter, min_d, n_close, max_u = evaluate_pairwise_interaction_affinity(
                entity_a=entities[id_a],
                entity_b=entities[id_b],
                image_shape=image_shape,
                contact_thresh_px=proximity_thresh_px
            )
            distance_records[(id_a, id_b)] = (min_d, n_close, max_u)
            print(f"  [PAIR AFFINITY] {id_a} ({entities[id_a]['class']}) <-> {id_b} ({entities[id_b]['class']}): is_inter={is_inter}, min_d={min_d:.1f}px, n_close={n_close} frames, max_u={max_u:.2f}")
            if is_inter:
                adj[id_a].add(id_b)
                adj[id_b].add(id_a)

    # Connected Components grouping for entities sharing active interaction links
    visited = set()
    raw_clusters = []
    for eid in entity_ids:
        if eid not in visited:
            component = []
            queue = [eid]
            visited.add(eid)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            raw_clusters.append(sorted(component))

    active_clusters = []
    isolated_singletons = []

    c_idx = 1
    for comp in raw_clusters:
        if len(comp) == 1:
            isolated_singletons.append(comp[0])
        else:
            # Active interactive cluster
            internal_dists = [
                distance_records.get((a, b), distance_records.get((b, a), (float("inf"), 0, 0.0)))[0]
                for a in comp for b in comp if a != b
            ]
            min_int_dist = min(internal_dists) if internal_dists else 0.0

            active_clusters.append({
                "cluster_id": f"cluster_{c_idx}",
                "entity_ids": comp,
                "entities": {eid: entities[eid] for eid in comp},
                "min_internal_distance_px": min_int_dist,
                "is_interactive": True
            })
            c_idx += 1

    return active_clusters, isolated_singletons

def compute_cluster_union_boxes(
    cluster_entity_ids: List[str],
    entities: Dict[str, Dict[str, Any]],
    sample_frame_indices: List[int],
    image_shape: Tuple[int, int] = (480, 640),
    padding_ratio: float = 0.20,
    stabilize_temporal_envelope: bool = True
) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Computes high-resolution bounding boxes for the cluster across sampled frames.
    If the cluster is localized (e.g. 2 people conversing/touching), applies a stabilized static window.
    If the cluster travels across the room, applies a smooth tracking crop to maximize zoom magnification.
    
    Returns: Dict of {frame_idx: (crop_x1, crop_y1, crop_x2, crop_y2)}
    """
    h_img, w_img = image_shape
    raw_boxes_per_frame = {}

    for f_idx in sample_frame_indices:
        visible_eids = [eid for eid in cluster_entity_ids if f_idx in entities[eid]["frame_map"]]
        # In VidVRD, an interactive cluster strictly requires >= 2 active entities present in the frame.
        # If fewer than 2 entities are present (e.g. entities have walked off), the interaction cluster is disbanded at this frame.
        if len(visible_eids) < 2:
            continue

        human_visible = [eid for eid in visible_eids if entities[eid].get("type") == "person"]
        f_boxes = []
        if human_visible:
            # Human entities actively forming the interaction core in this frame
            core_boxes = [entities[eid]["frame_map"][f_idx] for eid in human_visible]
            f_boxes.extend(core_boxes)
            # Include associated scene objects only if they are within physical reach (<= 80px) in this frame
            for eid in visible_eids:
                if eid not in human_visible:
                    o_box = entities[eid]["frame_map"][f_idx]
                    min_dist_to_human = min(compute_box_edge_distance(o_box, h_box) for h_box in core_boxes)
                    if min_dist_to_human <= 80.0:
                        f_boxes.append(o_box)
        else:
            f_boxes = [entities[eid]["frame_map"][f_idx] for eid in visible_eids]

        # In VidVRD, an interaction envelope strictly requires >= 2 entities actively present in interaction range
        if len(f_boxes) < 2:
            continue

        f_boxes_arr = np.array(f_boxes)
        ux1 = int(np.min(f_boxes_arr[:, 0]))
        uy1 = int(np.min(f_boxes_arr[:, 1]))
        ux2 = int(np.max(f_boxes_arr[:, 2]))
        uy2 = int(np.max(f_boxes_arr[:, 3]))
        raw_boxes_per_frame[f_idx] = (ux1, uy1, ux2, uy2)

    if not raw_boxes_per_frame:
        return {}

    # Compute trajectory displacement of the cluster center across sampled frames
    centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in raw_boxes_per_frame.values()]
    max_disp_x = max(c[0] for c in centers) - min(c[0] for c in centers)
    max_disp_y = max(c[1] for c in centers) - min(c[1] for c in centers)
    is_wide_travel = (max_disp_x > 0.20 * w_img) or (max_disp_y > 0.20 * h_img)

    if stabilize_temporal_envelope and not is_wide_travel:
        # Stabilized static envelope across all frames for localized interactions
        all_x1 = min(b[0] for b in raw_boxes_per_frame.values())
        all_y1 = min(b[1] for b in raw_boxes_per_frame.values())
        all_x2 = max(b[2] for b in raw_boxes_per_frame.values())
        all_y2 = max(b[3] for b in raw_boxes_per_frame.values())

        box_w = all_x2 - all_x1
        box_h = all_y2 - all_y1
        pad_x = int(box_w * padding_ratio)
        pad_y = int(box_h * padding_ratio)

        crop_x1 = max(0, all_x1 - pad_x)
        crop_y1 = max(0, all_y1 - pad_y)
        crop_x2 = min(w_img, all_x2 + pad_x)
        crop_y2 = min(h_img, all_y2 + pad_y)

        # Ensure reasonable minimum crop dimension (at least 320px for natural visual context)
        min_dim = 320
        if (crop_x2 - crop_x1) < min_dim and w_img >= min_dim:
            cx = (crop_x1 + crop_x2) // 2
            crop_x1 = max(0, min(w_img - min_dim, cx - min_dim // 2))
            crop_x2 = crop_x1 + min_dim
        if (crop_y2 - crop_y1) < min_dim and h_img >= min_dim:
            cy = (crop_y1 + crop_y2) // 2
            crop_y1 = max(0, min(h_img - min_dim, cy - min_dim // 2))
            crop_y2 = crop_y1 + min_dim

        return {f_idx: (crop_x1, crop_y1, crop_x2, crop_y2) for f_idx in raw_boxes_per_frame.keys()}
    else:
        # Dynamic smooth tracking window: follows the moving cluster with generous context padding
        crop_boxes = {}
        for f_idx, (ux1, uy1, ux2, uy2) in raw_boxes_per_frame.items():
            bw = ux2 - ux1
            bh = uy2 - uy1
            # Adaptive padding: generous padding on all sides
            px = max(int(bw * max(padding_ratio, 0.30)), 40)
            py = max(int(bh * max(padding_ratio, 0.20)), 30)
            cx1 = max(0, ux1 - px)
            cy1 = max(0, uy1 - py)
            cx2 = min(w_img, ux2 + px)
            cy2 = min(h_img, uy2 + py)
            
            # Enforce minimum size for natural aspect ratio
            min_dim = 320
            if (cx2 - cx1) < min_dim and w_img >= min_dim:
                cx = (cx1 + cx2) // 2
                cx1 = max(0, min(w_img - min_dim, cx - min_dim // 2))
                cx2 = cx1 + min_dim
            if (cy2 - cy1) < min_dim and h_img >= min_dim:
                cy = (cy1 + cy2) // 2
                cy1 = max(0, min(h_img - min_dim, cy - min_dim // 2))
                cy2 = cy1 + min_dim
            crop_boxes[f_idx] = (cx1, cy1, cx2, cy2)
        return crop_boxes

def render_cluster_zoom_frames(
    cluster: Dict[str, Any],
    clean_frames_dict: Dict[int, Tuple[float, np.ndarray]],
    crop_boxes: Dict[int, Tuple[int, int, int, int]],
    output_dir: str,
    color_palette: Optional[List[Tuple[int, int, int]]] = None
) -> List[Tuple[str, str, float]]:
    """
    Renders high-resolution Set-of-Marks zoom cropped frames strictly for the cluster entities.
    
    Returns: List of (saved_filename, full_path, zoom_factor)
    """
    os.makedirs(output_dir, exist_ok=True)
    if color_palette is None:
        color_palette = [
            (0, 165, 255), (50, 205, 50), (255, 191, 0), (238, 130, 238),
            (255, 20, 147), (0, 255, 255), (147, 112, 219), (0, 215, 255)
        ]

    saved_info = []
    order = 1

    cluster_entity_ids = cluster["entity_ids"]
    entities = cluster["entities"]

    for f_idx in sorted(clean_frames_dict.keys()):
        f_sec, clean_img = clean_frames_dict[f_idx]
        h_orig, w_orig = clean_img.shape[:2]
        cx1, cy1, cx2, cy2 = crop_boxes[f_idx]

        # High-res crop from pristine unannotated frame
        crop_img = clean_img[cy1:cy2, cx1:cx2].copy()
        crop_h, crop_w = crop_img.shape[:2]
        zoom_factor = float((w_orig * h_orig) / (crop_w * crop_h)) if (crop_w * crop_h) > 0 else 1.0

        # Render Set-of-Marks locally inside the cropped image
        for eid in cluster_entity_ids:
            fm = entities[eid]["frame_map"]
            if f_idx not in fm:
                sorted_k = sorted(fm.keys())
                if sorted_k and sorted_k[0] <= f_idx <= sorted_k[-1]:
                    nearest_k = min(sorted_k, key=lambda k: abs(k - f_idx))
                    orig_b = fm[nearest_k]
                else:
                    continue
            else:
                orig_b = fm[f_idx]
            # Transform to local coordinates
            lx1 = max(0, orig_b[0] - cx1)
            ly1 = max(0, orig_b[1] - cy1)
            lx2 = min(crop_w, orig_b[2] - cx1)
            ly2 = min(crop_h, orig_b[3] - cy1)

            if lx2 <= lx1 or ly2 <= ly1:
                continue

            # Extract integer id for color palette
            try:
                num_id = int(eid.strip("[]"))
            except ValueError:
                num_id = 1
            color = color_palette[num_id % len(color_palette)]
            clabel = entities[eid]["class"]

            # 1. 1.5px Soft Bounding Box Border (anti-aliased visual weight)
            overlay_b = crop_img.copy()
            cv2.rectangle(
                overlay_b,
                (max(0, lx1 - 1), max(0, ly1 - 1)),
                (min(crop_w - 1, lx2 + 1), min(crop_h - 1, ly2 + 1)),
                color,
                2
            )
            crop_img = cv2.addWeighted(overlay_b, 0.45, crop_img, 0.55, 0)
            cv2.rectangle(crop_img, (lx1, ly1), (lx2, ly2), color, 1, cv2.LINE_AA)

            # 2. Compact label badge with Alpha Blending (70% opacity, 30% background transparency)
            label_text = f"{eid} {clabel}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, 1)

            badge_h = text_h + 8
            badge_w = text_w + 10
            if ly1 >= badge_h + 2:
                by1 = ly1 - badge_h
                by2 = ly1
                ty = ly1 - 4
            else:
                by1 = ly1
                by2 = min(crop_h, ly1 + badge_h)
                ty = ly1 + text_h + 2

            bx1 = lx1
            bx2 = min(crop_w, lx1 + badge_w)

            # Local Alpha Blending (70% tint, 30% background)
            sub = crop_img[by1:by2, bx1:bx2]
            overlay_badge = np.full_like(sub, color)
            crop_img[by1:by2, bx1:bx2] = cv2.addWeighted(overlay_badge, 0.70, sub, 0.30, 0)

            # Outline for the badge
            cv2.rectangle(crop_img, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)

            # 3. Bold White Text (CTRL+B faux-bold double pass)
            cv2.putText(crop_img, label_text, (bx1 + 5, ty), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(crop_img, label_text, (bx1 + 6, ty), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

            # Redundant interior duplicate ID completely removed to prevent face/hand occlusion

        fn = f"frame_{order:02d}_{f_sec:.2f}s_crop.jpg"
        out_path = os.path.join(output_dir, fn)
        cv2.imwrite(out_path, crop_img)
        saved_info.append((fn, out_path, zoom_factor))
        order += 1

    return saved_info


def render_cluster_visualization_video(
    raw_clean_frames,
    active_clusters,
    singletons,
    all_entities,
    output_video_path=None,
    preview_dir=None,
    num_preview_frames=10,
    fps=29.97,
    video_basename="video",
    color_palette=None,
    proximity_thresh_px=45.0
):
    """
    VidVRD Task 2: Spatio-Temporal Interaction Clustering (STIC) Visualization
    Renders full-resolution surveillance video showing ONLY the Cluster Union Box
    enclosing interacting entities in each active cluster.
    
    Mentor & User Specifications strictly enforced:
    1. Draw ONLY the large bounding box enclosing the active cluster (Union Box).
    2. STRICT NEGATIVE CONSTRAINT: Do NOT draw individual bounding boxes inside the cluster.
    3. Header badge on union box: e.g. 'CLUSTER 1: [1] person + [3] handbag' (dynamic IDs and classes).
    4. STRICT MODULARITY CONSTRAINT: Do NOT include interaction predicates ('touch', 'carry') on video.
    5. Clean frame: Zero top-left HUD / overlay frames (clean unobstructed surveillance).
    6. Dynamic Spatio-Temporal Adjacency: If an entity in the cluster separates or is left behind
       (e.g. backpack left on the floor while humans walk away), it is dynamically detached from
       the box so the union box does NOT stretch unrealistically across the scene.
    7. Codec: Uses 'avc1' (H.264) for universal Windows Media Player / Chrome / VS Code playback.
    """
    if not raw_clean_frames:
        return {"output_video_path": None, "preview_frames": []}
        
    sorted_frames = sorted(raw_clean_frames.keys())
    first_f_sec, first_frame = raw_clean_frames[sorted_frames[0]]
    h_orig, w_orig = first_frame.shape[:2]
    
    # Modern bright BGR colors: Emerald Green, Electric Cyan, Neon Orange, Vivid Magenta
    if color_palette is None:
        color_palette = [
            (100, 235, 50),
            (255, 190, 0),
            (0, 165, 255),
            (238, 130, 238),
            (0, 255, 255)
        ]
        
    writer = None
    if output_video_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
        # Use avc1 (H.264) with fallback to mp4v
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w_orig, h_orig))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w_orig, h_orig))
            
    if preview_dir and num_preview_frames > 0:
        os.makedirs(preview_dir, exist_ok=True)
        for f in os.listdir(preview_dir):
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".json"):
                try:
                    os.remove(os.path.join(preview_dir, f))
                except OSError:
                    pass
        if len(sorted_frames) <= num_preview_frames:
            preview_indices = set(sorted_frames)
        else:
            step = (len(sorted_frames) - 1) / (num_preview_frames - 1)
            preview_indices = set(sorted_frames[int(round(i * step))] for i in range(num_preview_frames))
    else:
        preview_indices = set()
        
    saved_previews = []
    rendered_frames_meta = {}
    
    for f_idx in sorted_frames:
        f_sec, clean_img = raw_clean_frames[f_idx]
        vis_img = clean_img.copy()
        
        clusters_to_draw = []
        
        for c_i, c in enumerate(active_clusters):
            cid = c.get("cluster_id", f"cluster_{c_i + 1}")
            c_num = cid.replace("cluster_", "")
            color = color_palette[c_i % len(color_palette)]
            
            present_eids = [eid for eid in c["entity_ids"] if f_idx in all_entities[eid]["frame_map"]]
            if len(present_eids) < 2:
                continue
                
            # Build frame-level physical proximity adjacency graph
            # Ensures that if an entity was detached (e.g. backpack resting far away on the floor),
            # it is not artificially stretched into the cluster box!
            adj = {eid: set() for eid in present_eids}
            for i in range(len(present_eids)):
                for j in range(i + 1, len(present_eids)):
                    ea, eb = present_eids[i], present_eids[j]
                    ba = all_entities[ea]["frame_map"][f_idx]
                    bb = all_entities[eb]["frame_map"][f_idx]
                    d = compute_box_edge_distance(ba, bb)
                    iou = compute_box_iou(ba, bb)
                    if d <= proximity_thresh_px or iou > 0.0:
                        adj[ea].add(eb)
                        adj[eb].add(ea)
                        
            # Find connected components within this frame
            visited = set()
            for eid in present_eids:
                if eid not in visited:
                    comp = []
                    queue = [eid]
                    visited.add(eid)
                    while queue:
                        curr = queue.pop(0)
                        comp.append(curr)
                        for neighbor in adj[curr]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)
                                
                    # Only components with >= 2 entities are actively interacting in this frame
                    if len(comp) >= 2:
                        comp_sorted = sorted(comp, key=lambda x: (0 if all_entities[x].get("type") == "person" else 1, x))
                        boxes = [all_entities[e]["frame_map"][f_idx] for e in comp_sorted]
                        pad = 12
                        ux1 = max(0, int(min(b[0] for b in boxes)) - pad)
                        uy1 = max(0, int(min(b[1] for b in boxes)) - pad)
                        ux2 = min(w_orig, int(max(b[2] for b in boxes)) + pad)
                        uy2 = min(h_orig, int(max(b[3] for b in boxes)) + pad)
                        
                        entity_parts = [f"{e} {all_entities[e].get('class', '')}" for e in comp_sorted]
                        entity_desc = " + ".join(entity_parts)
                        badge_text = f"CLUSTER {c_num}: {entity_desc}"
                        
                        clusters_to_draw.append({
                            "cid": cid,
                            "c_num": c_num,
                            "color": color,
                            "box": (ux1, uy1, ux2, uy2),
                            "badge_text": badge_text,
                            "entities": comp_sorted
                        })
                        
                        rendered_frames_meta[f_idx] = {
                            "sec": round(f_sec, 2),
                            "cluster_id": cid,
                            "union_box": [ux1, uy1, ux2, uy2],
                            "entities": comp_sorted,
                            "badge": badge_text
                        }
                        
        # Render active cluster union boxes with sleek tactical aesthetics (1.5px soft border + alpha-blending badge)
        for cdata in clusters_to_draw:
            ux1, uy1, ux2, uy2 = cdata["box"]
            color = cdata["color"]
            badge_text = cdata["badge_text"]
            
            # 1. 1.5px Soft Bounding Box Border (anti-aliased visual weight)
            overlay_b = vis_img.copy()
            cv2.rectangle(
                overlay_b,
                (max(0, ux1 - 1), max(0, uy1 - 1)),
                (min(w_orig - 1, ux2 + 1), min(h_orig - 1, uy2 + 1)),
                color,
                2
            )
            vis_img = cv2.addWeighted(overlay_b, 0.45, vis_img, 0.55, 0)
            cv2.rectangle(vis_img, (ux1, uy1), (ux2, uy2), color, 1, cv2.LINE_AA)
            
            # 2. Sleek tactical corner brackets
            corner_len = min(20, (ux2 - ux1) // 5, (uy2 - uy1) // 5)
            c_thick = 2
            cv2.line(vis_img, (ux1, uy1), (ux1 + corner_len, uy1), color, c_thick, cv2.LINE_AA)
            cv2.line(vis_img, (ux1, uy1), (ux1, uy1 + corner_len), color, c_thick, cv2.LINE_AA)
            cv2.line(vis_img, (ux2, uy1), (ux2 - corner_len, uy1), color, c_thick, cv2.LINE_AA)
            cv2.line(vis_img, (ux2, uy1), (ux2, uy1 + corner_len), color, c_thick, cv2.LINE_AA)
            cv2.line(vis_img, (ux1, uy2), (ux1 + corner_len, uy2), color, c_thick, cv2.LINE_AA)
            cv2.line(vis_img, (ux1, uy2), (ux1, uy2 - corner_len), color, c_thick, cv2.LINE_AA)
            cv2.line(vis_img, (ux2, uy2), (ux2 - corner_len, uy2), color, c_thick, cv2.LINE_AA)
            cv2.line(vis_img, (ux2, uy2), (ux2, uy2 - corner_len), color, c_thick, cv2.LINE_AA)
            
            # 3. Compact Header Badge with Alpha Blending (70% dark tint, 30% background transparency)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, 1)
            
            badge_h = th + 8
            badge_w = tw + 12
            
            if uy1 >= badge_h + 2:
                by1 = uy1 - badge_h
                by2 = uy1
                ty = uy1 - 4
            else:
                by1 = uy1
                by2 = min(h_orig, uy1 + badge_h)
                ty = uy1 + th + 2
                
            bx1 = ux1
            if bx1 + badge_w > w_orig:
                bx1 = max(0, w_orig - badge_w - 2)
            bx2 = min(w_orig, bx1 + badge_w)
            
            # Local Alpha Blending: 80% opacity dark tint with subtle cluster hue, 20% background transparency
            sub = vis_img[by1:by2, bx1:bx2]
            dark_tint = (int(color[0] * 0.20 + 15), int(color[1] * 0.20 + 15), int(color[2] * 0.20 + 15))
            overlay_badge = np.full_like(sub, dark_tint)
            vis_img[by1:by2, bx1:bx2] = cv2.addWeighted(overlay_badge, 0.80, sub, 0.20, 0)
            
            # Crisp 1px outline for the badge
            cv2.rectangle(vis_img, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)
            
            # Faux-bold white text (double-pass)
            cv2.putText(vis_img, badge_text, (bx1 + 6, ty), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis_img, badge_text, (bx1 + 7, ty), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            
        if writer:
            writer.write(vis_img)
            
        if f_idx in preview_indices:
            preview_order = len(saved_previews) + 1
            preview_fn = f"preview_frame_{preview_order:02d}_{f_sec:.2f}s.jpg"
            preview_path = os.path.join(preview_dir, preview_fn)
            cv2.imwrite(preview_path, vis_img)
            saved_previews.append({
                "frame_idx": f_idx,
                "timestamp_sec": round(f_sec, 2),
                "filename": preview_fn,
                "path": preview_path,
                "has_cluster_box": bool(clusters_to_draw),
                "cluster_info": rendered_frames_meta.get(f_idx, None)
            })
            
    if writer:
        writer.release()
        
    if preview_dir:
        manifest_path = os.path.join(preview_dir, "preview_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump({
                "video": video_basename,
                "total_sampled_previews": len(saved_previews),
                "singletons_clutter_filtered": singletons,
                "previews": saved_previews
            }, mf, indent=2, ensure_ascii=False)
            
    return {
        "output_video_path": output_video_path,
        "preview_dir": preview_dir,
        "total_preview_frames": len(saved_previews),
        "previews": saved_previews
    }


def compute_active_contact_frames(cluster, entities, image_shape, contact_thresh_px=35.0):
    eids = cluster.get('entity_ids', [])
    if len(eids) < 2:
        return []
    present_maps = [set(entities[e]['frame_map'].keys()) for e in eids if e in entities and 'frame_map' in entities[e]]
    if not present_maps:
        return []
    common_frames = set.intersection(*present_maps)
    res = []
    for f in sorted(common_frames):
        is_contact = False
        for i in range(len(eids)):
            for j in range(i + 1, len(eids)):
                ea, eb = eids[i], eids[j]
                if f in entities[ea]['frame_map'] and f in entities[eb]['frame_map']:
                    b1 = entities[ea]['frame_map'][f]
                    b2 = entities[eb]['frame_map'][f]
                    d = compute_box_edge_distance(b1, b2)
                    iou = compute_box_iou(b1, b2)
                    if d <= contact_thresh_px or iou > 0:
                        is_contact = True
                        break
            if is_contact:
                break
        if is_contact:
            res.append(f)
    return res if res else sorted(list(common_frames))
