"""
Offline Video Renderer from VidVRD Relations JSON
=================================================
Renders visualizer surveillance video directly from structured JSON output 
and raw video frames, without requiring deep learning models, PyTorch, or GPU.

Usage:
    python render_from_json.py --video data/videos/video2.mp4 --json data/output/full_video/video2/video2_relations.json
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np


def render_frame_from_data(frame: np.ndarray, f_idx: int, fps: float, tracks: dict, relations: list) -> np.ndarray:
    h, w = frame.shape[:2]
    vis = frame.copy()
    
    scale = max(0.5, min(w / 1280.0, h / 720.0))
    line_w = max(1, int(round(1.5 * scale)))
    font_main = 0.45 * scale
    font_sub = 0.40 * scale
    txt_th = max(1, int(round(1.0 * scale)))

    # 1. Draw Track Bounding Boxes
    present_boxes = {}
    f_str = str(f_idx)
    for tid, tinfo in tracks.items():
        trajs = tinfo.get("trajectories", {})
        if f_str in trajs:
            b = trajs[f_str]
            bx1, by1, bx2, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            color = tuple(tinfo.get("color", [0, 255, 0]))
            clabel = tinfo.get("class", "")
            present_boxes[tid] = (bx1, by1, bx2, by2)

            # Soft border
            overlay_b = vis.copy()
            cv2.rectangle(
                overlay_b,
                (max(0, bx1 - line_w), max(0, by1 - line_w)),
                (min(w - 1, bx2 + line_w), min(h - 1, by2 + line_w)),
                color,
                line_w + 2
            )
            vis = cv2.addWeighted(overlay_b, 0.40, vis, 0.60, 0)
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, line_w, cv2.LINE_AA)

            # Badge
            label_text = f"{tid} {clabel}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label_text, font, font_main, txt_th)
            badge_h = th + int(round(8 * scale))
            badge_w = tw + int(round(10 * scale))
            badge_y1 = max(0, by1 - badge_h) if by1 >= badge_h + 2 else by1
            badge_y2 = badge_y1 + badge_h
            badge_x1 = bx1
            badge_x2 = min(w, bx1 + badge_w)

            sub = vis[badge_y1:badge_y2, badge_x1:badge_x2]
            if sub.shape[0] > 0 and sub.shape[1] > 0:
                overlay_badge = np.full_like(sub, color)
                vis[badge_y1:badge_y2, badge_x1:badge_x2] = cv2.addWeighted(overlay_badge, 0.70, sub, 0.30, 0)
                cv2.rectangle(vis, (badge_x1, badge_y1), (badge_x2, badge_y2), color, line_w, cv2.LINE_AA)
                ty = badge_y2 - int(round(4 * scale))
                cv2.putText(vis, label_text, (badge_x1 + int(round(5 * scale)), ty), font, font_main, (255, 255, 255), txt_th, cv2.LINE_AA)

    # 2. Draw Active Relations for this frame
    active_now = []
    for r in relations:
        act_fs = r.get("active_frames", [])
        if f_idx in act_fs:
            sub_id = str(r["subject_id"])
            obj_id = str(r["object_id"])
            if sub_id in present_boxes and obj_id in present_boxes:
                active_now.append(r)
                b_s = present_boxes[sub_id]
                b_o = present_boxes[obj_id]
                c_s = ((b_s[0] + b_s[2]) // 2, (b_s[1] + b_s[3]) // 2)
                c_o = ((b_o[0] + b_o[2]) // 2, (b_o[1] + b_o[3]) // 2)

                # Yellow connecting line
                cv2.line(vis, c_s, c_o, (0, 255, 255), line_w, cv2.LINE_AA)

                # Midpoint pill
                mid_x = (c_s[0] + c_o[0]) // 2
                mid_y = (c_s[1] + c_o[1]) // 2
                rel_label = f"{r['predicate']} ({r.get('confidence', 0.0):.2f})"
                (rw, rh), _ = cv2.getTextSize(rel_label, cv2.FONT_HERSHEY_SIMPLEX, font_sub, txt_th)
                rx1 = max(0, mid_x - rw // 2 - int(round(4 * scale)))
                ry1 = max(0, mid_y - rh // 2 - int(round(3 * scale)))
                rx2 = min(w, rx1 + rw + int(round(8 * scale)))
                ry2 = min(h, ry1 + rh + int(round(6 * scale)))

                pill_sub = vis[ry1:ry2, rx1:rx2]
                if pill_sub.shape[0] > 0 and pill_sub.shape[1] > 0:
                    pill_bg = np.zeros_like(pill_sub)
                    vis[ry1:ry2, rx1:rx2] = cv2.addWeighted(pill_bg, 0.70, pill_sub, 0.30, 0)
                    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 255), line_w, cv2.LINE_AA)
                    cv2.putText(vis, rel_label, (rx1 + int(round(4 * scale)), ry2 - int(round(4 * scale))),
                                cv2.FONT_HERSHEY_SIMPLEX, font_sub, (0, 255, 255), txt_th, cv2.LINE_AA)

    # 3. Timestamp OSD (Top Right)
    sec = f_idx / fps
    osd_text = f"REC {int(sec // 60):02d}:{sec % 60:05.2f}"
    (tw, th), _ = cv2.getTextSize(osd_text, cv2.FONT_HERSHEY_SIMPLEX, font_main, txt_th)
    osd_x = w - tw - int(round(16 * scale))
    osd_y = int(round(22 * scale))
    sub_osd = vis[osd_y - th - int(round(4 * scale)):osd_y + int(round(4 * scale)), osd_x - int(round(6 * scale)):w - int(round(8 * scale))]
    if sub_osd.shape[0] > 0 and sub_osd.shape[1] > 0:
        dark_bg = np.zeros_like(sub_osd)
        vis[osd_y - th - int(round(4 * scale)):osd_y + int(round(4 * scale)), osd_x - int(round(6 * scale)):w - int(round(8 * scale))] = cv2.addWeighted(dark_bg, 0.60, sub_osd, 0.40, 0)
        cv2.putText(vis, osd_text, (osd_x, osd_y), cv2.FONT_HERSHEY_SIMPLEX, font_main, (255, 255, 255), txt_th, cv2.LINE_AA)

    # 4. Live Relation Feed (Top Left)
    if active_now:
        feed_y = int(round(22 * scale))
        for r in active_now[:2]:
            feed_txt = f"{r['subject_id']} {r.get('subject_class', '')} -{r['predicate']} ({r.get('confidence', 0.0):.2f})-> {r['object_id']} {r.get('object_class', '')}"
            (fw, fh), _ = cv2.getTextSize(feed_txt, cv2.FONT_HERSHEY_SIMPLEX, font_sub, txt_th)
            feed_sub = vis[feed_y - fh - int(round(3 * scale)):feed_y + int(round(4 * scale)), int(round(10 * scale)):int(round(14 * scale)) + fw + int(round(6 * scale))]
            if feed_sub.shape[0] > 0 and feed_sub.shape[1] > 0:
                dark_pill = np.zeros_like(feed_sub)
                vis[feed_y - fh - int(round(3 * scale)):feed_y + int(round(4 * scale)), int(round(10 * scale)):int(round(14 * scale)) + fw + int(round(6 * scale))] = cv2.addWeighted(dark_pill, 0.70, feed_sub, 0.30, 0)
                cv2.putText(vis, feed_txt, (int(round(14 * scale)), feed_y), cv2.FONT_HERSHEY_SIMPLEX, font_sub, (80, 255, 120), txt_th, cv2.LINE_AA)
            feed_y += int(round(20 * scale))

    return vis


def render_video_from_json(video_path: str, json_path: str, output_path: str = None) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("video_metadata", {})
    fps = float(meta.get("fps", 25.0))
    tracks = data.get("tracks", data.get("entities", {}))
    relations = data.get("relations", [])

    if output_path is None:
        video_dir = os.path.dirname(json_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{base_name}_rendered_from_json.mp4")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(output_path, fourcc, fps_video, (w, h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps_video, (w, h))

    print(f"[OFFLINE RENDER] Rendering {total_frames} frames from JSON...")
    print(f"  Input Video:  {video_path}")
    print(f"  Input JSON:   {json_path}")
    print(f"  Output MP4:   {output_path}")

    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        vis_frame = render_frame_from_data(frame, f_idx, fps_video, tracks, relations)
        writer.write(vis_frame)
        f_idx += 1

    cap.release()
    writer.release()
    print(f"[OFFLINE RENDER] Completed successfully ({f_idx} frames). Saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Render surveillance video directly from VidVRD Relations JSON")
    parser.add_argument("--video", type=str, required=True, help="Path to raw input video MP4")
    parser.add_argument("--json", type=str, required=True, help="Path to input relations JSON")
    parser.add_argument("--output", type=str, default=None, help="Optional output MP4 path")
    args = parser.parse_args()

    render_video_from_json(video_path=args.video, json_path=args.json, output_path=args.output)


if __name__ == "__main__":
    main()
