"""
VidVRD RelateAnything Surveillance Platform
=============================================
Compact industrial surveillance dashboard for video visual relationship detection.
Single-screen, zero-scroll layout with entity-centric interaction tracking.

Usage:
    streamlit run app.py
"""

import os
import sys
import json
import subprocess
import streamlit as st
import pandas as pd
from render_from_json import render_video_from_json

# Page configuration
st.set_page_config(
    page_title="VidVRD RelateAnything | Giám Sát Hành Vi Video",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Zero-Scroll, Solid Header, Clean Entity Inspector
st.markdown("""
<style>
    /* Completely hide Streamlit header and deploy button to reclaim top space */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
    }

    /* Dark Theme & Zero-Scroll Container */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
    }
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 100% !important;
    }

    /* Compact Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0E131F !important;
        border-right: 1px solid #1F293D !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem !important;
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
    }
    section[data-testid="stSidebar"] h3 {
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        margin-bottom: 4px !important;
        color: #F8FAFC !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] .stRadio div {
        font-size: 0.82rem !important;
        color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] .stRadio,
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stSlider {
        margin-bottom: -10px !important;
    }
    section[data-testid="stSidebar"] hr {
        margin: 10px 0 !important;
        border-color: #1F293D !important;
    }

    /* Top Persistent Header */
    .dash-header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 8px;
        margin-bottom: 10px;
        border-bottom: 1px solid #1F293D;
    }
    .dash-main-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
    }
    .dash-sub-title {
        font-size: 0.92rem;
        font-weight: normal;
        color: #94A3B8;
        margin-left: 6px;
    }

    /* Compact Metric Cards */
    .metrics-bar {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-bottom: 10px;
    }
    .stat-card {
        background: #111827;
        border: 1px solid #1F293D;
        border-radius: 6px;
        padding: 6px 12px;
        text-align: center;
    }
    .stat-label {
        font-size: 0.72rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 2px;
    }
    .stat-val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #38BDF8;
        line-height: 1.2;
    }

    /* Panel Headers */
    .panel-header-compact {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.92rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 6px;
        padding-bottom: 4px;
        border-bottom: 1px solid #1F293D;
    }
    .cam-tag {
        background: #2563EB;
        color: #FFFFFF;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 1px 6px;
        border-radius: 3px;
    }

    /* Video Player Constraints */
    video {
        max-height: 380px !important;
        width: 100% !important;
        object-fit: contain !important;
        border-radius: 6px !important;
        background: #000000 !important;
        border: 1px solid #1F293D !important;
    }
</style>
""", unsafe_allow_html=True)


def get_available_benchmark_videos():
    """Returns sorted list of benchmark videos available in data/videos/."""
    video_dir = os.path.join("data", "videos")
    if not os.path.exists(video_dir):
        return []
    videos = [f for f in sorted(os.listdir(video_dir)) if f.endswith(".mp4")]
    def sort_key(name):
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 999
    return sorted(videos, key=sort_key)


def main():
    # 1. Guaranteed Visible Top Header
    st.markdown("""
    <div class="dash-header-bar">
        <div class="dash-main-title">
            🎥 VidVRD RelateAnything <span class="dash-sub-title">&bull; Hệ Thống Giám Sát Hành Vi & Quan Hệ Thị Giác</span>
        </div>
        <div style="font-size: 0.8rem; color: #10B981; font-weight: 600;">
            ● Trực Tuyến (Online)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Compact Sidebar
    st.sidebar.markdown("### ⚙️ Nguồn Video")
    input_source = st.sidebar.radio(
        "Chọn chế độ:",
        ["🎬 Video mẫu (Benchmark)", "📤 Tải lên video (Upload)"],
        label_visibility="collapsed"
    )

    selected_video_path = None
    video_name = ""

    if "Benchmark" in input_source:
        benchmark_videos = get_available_benchmark_videos()
        if benchmark_videos:
            default_choice = "video2.mp4" if "video2.mp4" in benchmark_videos else benchmark_videos[0]
            default_idx = benchmark_videos.index(default_choice)
            selected_choice = st.sidebar.selectbox("Chọn Video Mẫu:", benchmark_videos, index=default_idx)
            selected_video_path = os.path.join("data", "videos", selected_choice)
            video_name = os.path.splitext(selected_choice)[0]
        else:
            st.sidebar.warning("Không có video trong data/videos/")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Tải file video lên:",
            type=["mp4", "avi", "mov"]
        )
        if uploaded_file is not None:
            upload_dir = os.path.join("data", "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            selected_video_path = os.path.join(upload_dir, uploaded_file.name)
            with open(selected_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_name = os.path.splitext(uploaded_file.name)[0]
            st.sidebar.success(f"Đã nạp: {uploaded_file.name}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Tham Số AI")
    conf_thresh = st.sidebar.slider("YOLO Conf:", min_value=0.15, max_value=0.70, value=0.35, step=0.05)
    device = st.sidebar.selectbox("Thiết bị (Device):", ["auto", "cuda", "cpu"], index=0)

    st.sidebar.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    run_pipeline_btn = st.sidebar.button("🚀 Bắt Đầu Phân Tích", type="primary", use_container_width=True)

    # Output paths
    expected_out_dir = os.path.join("data", "output", "full_video", video_name)
    expected_json = os.path.join(expected_out_dir, f"{video_name}_relations.json")
    expected_mp4 = os.path.join(expected_out_dir, f"{video_name}_visualizer.mp4")

    # Pipeline Execution Action
    if run_pipeline_btn:
        if not selected_video_path or not os.path.exists(selected_video_path):
            st.error("Vui lòng chọn hoặc tải lên một tệp video hợp lệ.")
            return

        with st.spinner(f"Đang phân tích RelateAnything trên {os.path.basename(selected_video_path)}..."):
            cmd = [
                sys.executable, "pipeline_relate_anything.py",
                "--video", selected_video_path,
                "--mode", "roi",
                "--conf", str(conf_thresh),
                "--device", device,
                "--full_video"
            ]
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode == 0:
                st.success("Phân tích hoàn tất!")
                st.rerun()
            else:
                st.error(f"Lỗi phân tích (Code: {process.returncode})")
                with st.expander("Chi tiết lỗi"):
                    st.code(process.stderr or process.stdout)

    # Main Display Area
    if os.path.exists(expected_json) and os.path.exists(expected_mp4):
        with open(expected_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("video_metadata", {})
        tracks = data.get("tracks", data.get("entities", {}))
        relations = data.get("relations", [])
        fps = meta.get("fps", 25.0)
        dur = meta.get("duration_sec", 0.0)

        # 3. Compact Metrics Row
        st.markdown(f"""
        <div class="metrics-bar">
            <div class="stat-card">
                <div class="stat-label">Thời Lượng Video</div>
                <div class="stat-val">{dur:.1f}s</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Thực Thể Theo Dõi</div>
                <div class="stat-val">{len(tracks)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Quan Hệ Phát Hiện</div>
                <div class="stat-val">{len(relations)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Tốc Độ Khung Hình</div>
                <div class="stat-val">{fps:.1f} FPS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4. Main 2-Column Layout (Balanced for 1 Screen)
        col_video, col_table = st.columns([1.15, 1.0], gap="medium")

        # --- LEFT: Visualized Surveillance Video ---
        with col_video:
            st.markdown(f"""
            <div class="panel-header-compact">
                <span>📹 Video Kết Quả Phân Tích (Detected Visualizer)</span>
                <span class="cam-tag">CAM: {video_name.upper()}</span>
            </div>
            """, unsafe_allow_html=True)

            st.video(expected_mp4)

            # Compact Action Bar right under video
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            b1, b2 = st.columns([1.1, 1.0])
            with b1:
                st.download_button(
                    label="📥 Tải Tệp JSON (Full)",
                    data=json.dumps(data, indent=2, ensure_ascii=False),
                    file_name=f"{video_name}_relations.json",
                    mime="application/json",
                    use_container_width=True
                )
            with b2:
                with st.popover("⚡ Render lại từ JSON"):
                    st.caption("Tạo video từ toạ độ JSON không cần chạy lại AI (~3 giây CPU).")
                    rerender_out = os.path.join(expected_out_dir, f"{video_name}_cpu_render.mp4")
                    if st.button("Chạy Render CPU", key="btn_render_cpu"):
                        with st.spinner("Đang render..."):
                            try:
                                res_p = render_video_from_json(selected_video_path, expected_json, rerender_out)
                                st.success(f"Xong! Lưu: {res_p}")
                            except Exception as e:
                                st.error(f"Lỗi: {e}")

        # --- RIGHT: Intuitive Entity & Interaction Table (No fluff, no repetition) ---
        with col_table:
            # Build clean entity rows mapped with relations and first detected second
            entity_rows = []
            for tid, tinfo in sorted(tracks.items(), key=lambda x: int(x[0].strip('[]')) if x[0].strip('[]').isdigit() else 999):
                cls_name = tinfo.get("class", "unknown")
                traj = tinfo.get("trajectories", {})
                frames = [int(f) for f in traj.keys()] if traj else [0]
                first_sec = min(frames) / fps
                last_sec = max(frames) / fps

                # Deduplicate actions
                actions = []
                seen_preds = set()
                for r in relations:
                    pred = r.get("predicate", "")
                    sub = r.get("subject_id", "")
                    obj = r.get("object_id", "")
                    if sub == tid:
                        key = (pred, obj)
                        if key not in seen_preds:
                            seen_preds.add(key)
                            actions.append(f"{pred} {obj}")
                    elif obj == tid:
                        key = (f"be_{pred}", sub)
                        if key not in seen_preds:
                            seen_preds.add(key)
                            actions.append(f"được {sub} {pred}")

                act_str = ", ".join(actions) if actions else "— (Đơn lẻ)"
                status = "Có tương tác" if actions else "Tĩnh / Độc lập"

                entity_rows.append({
                    "ID": tid,
                    "Đối tượng": cls_name,
                    "Phát hiện lúc": f"{first_sec:.1f}s",
                    "Khoảng thời gian": f"{first_sec:.1f}s ➔ {last_sec:.1f}s",
                    "Hành vi tương tác": act_str,
                    "Trạng thái": status,
                    "has_action": bool(actions)
                })

            df_all = pd.DataFrame(entity_rows)
            num_interactive = sum(1 for r in entity_rows if r["has_action"])

            # Filter Header
            st.markdown(f"""
            <div class="panel-header-compact">
                <span>📋 Danh Sách Đối Tượng & Hành Vi ({len(entity_rows)} thực thể)</span>
                <span style="font-size: 0.78rem; color: #38BDF8;">
                    {num_interactive} đối tượng có tương tác
                </span>
            </div>
            """, unsafe_allow_html=True)

            filter_mode = st.radio(
                "Lọc:",
                ["Tất cả thực thể", "Chỉ đối tượng có tương tác"],
                horizontal=True,
                label_visibility="collapsed"
            )

            if filter_mode == "Chỉ đối tượng có tương tác":
                df_display = df_all[df_all["has_action"] == True].copy()
            else:
                df_display = df_all.copy()

            # Display clean, intuitive columns
            df_display = df_display[["ID", "Đối tượng", "Phát hiện lúc", "Khoảng thời gian", "Hành vi tương tác", "Trạng thái"]]

            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                height=355
            )

    else:
        # Empty State (Compact)
        st.markdown(f"""
        <div style="background: #111827; border: 1px solid #1F293D; border-radius: 8px; text-align: center; padding: 30px 20px; margin-top: 20px;">
            <div style="font-size: 2rem; margin-bottom: 8px;">🔍</div>
            <h4 style="color: #F8FAFC; margin-bottom: 6px;">Chưa Có Kết Quả Cho Video {video_name}</h4>
            <p style="color: #94A3B8; font-size: 0.85rem; max-width: 450px; margin: 0 auto 16px auto;">
                Tệp kết quả chưa được tạo. Hãy bấm nút <b>🚀 Bắt Đầu Phân Tích</b> ở thanh bên trái để hệ thống xử lý tự động.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
