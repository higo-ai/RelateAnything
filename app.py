"""
VidVRD RelateAnything Web Application
=====================================
Interactive Streamlit interface for surveillance video relation detection.

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
    page_title="VidVRD RelateAnything Surveillance Platform",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def get_available_benchmark_videos():
    video_dir = os.path.join("data", "videos")
    if not os.path.exists(video_dir):
        return []
    videos = [f for f in sorted(os.listdir(video_dir)) if f.endswith(".mp4")]
    # Custom numeric sort
    def sort_key(name):
        num_part = "".join(ch for ch in name if ch.isdigit())
        return int(num_part) if num_part else 999
    return sorted(videos, key=sort_key)


def main():
    st.markdown('<div class="main-header">🎥 VidVRD RelateAnything Video Relation Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">End-to-End Visual Relationship Detection & Spatio-Temporal Interaction Clustering (STIC)</div>', unsafe_allow_html=True)

    # Sidebar: Input Configuration
    st.sidebar.header("Video Configuration")
    input_source = st.sidebar.radio("Select Input Source:", ["Benchmark Video Library", "Upload Custom Video"])

    selected_video_path = None
    video_name = ""

    if input_source == "Benchmark Video Library":
        benchmark_videos = get_available_benchmark_videos()
        if benchmark_videos:
            default_idx = benchmark_videos.index("video2.mp4") if "video2.mp4" in benchmark_videos else 0
            selected_choice = st.sidebar.selectbox("Choose Benchmark Video:", benchmark_videos, index=default_idx)
            selected_video_path = os.path.join("data", "videos", selected_choice)
            video_name = os.path.splitext(selected_choice)[0]
        else:
            st.sidebar.warning("No benchmark videos found in data/videos/")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Video File (MP4):", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            upload_dir = os.path.join("data", "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            selected_video_path = os.path.join(upload_dir, uploaded_file.name)
            with open(selected_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_name = os.path.splitext(uploaded_file.name)[0]
            st.sidebar.success(f"Uploaded: {uploaded_file.name}")

    st.sidebar.markdown("---")
    st.sidebar.header("Pipeline Parameters")
    mode = st.sidebar.selectbox("Inference Mode:", ["roi", "full"], index=0, help="'roi': Spatio-temporal dynamic crop; 'full': Global surveillance frame")
    conf_thresh = st.sidebar.slider("Detection Confidence:", min_value=0.15, max_value=0.70, value=0.35, step=0.05)
    device = st.sidebar.selectbox("Device:", ["auto", "cuda", "cpu"], index=0)

    run_pipeline_btn = st.sidebar.button("🚀 Run Pipeline Analysis", type="primary", use_container_width=True)

    # Check for existing results
    expected_out_dir = os.path.join("data", "output", "full_video", video_name)
    expected_json = os.path.join(expected_out_dir, f"{video_name}_relations.json")
    expected_mp4 = os.path.join(expected_out_dir, f"{video_name}_visualizer.mp4")

    # Execution Action
    if run_pipeline_btn:
        if not selected_video_path or not os.path.exists(selected_video_path):
            st.error("Please select or upload a valid video file first.")
            return

        with st.spinner(f"Running RelateAnything Pipeline on {os.path.basename(selected_video_path)}..."):
            cmd = [
                sys.executable, "pipeline_relate_anything.py",
                "--video", selected_video_path,
                "--mode", mode,
                "--conf", str(conf_thresh),
                "--device", device,
                "--full_video"
            ]
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode == 0:
                st.success("Analysis completed successfully!")
            else:
                st.error(f"Pipeline execution failed (Exit code: {process.returncode})")
                with st.expander("View Error Log"):
                    st.code(process.stderr or process.stdout)

    # Display Results
    if os.path.exists(expected_json) and os.path.exists(expected_mp4):
        with open(expected_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("video_metadata", {})
        tracks = data.get("tracks", data.get("entities", {}))
        relations = data.get("relations", [])
        fps = meta.get("fps", 25.0)
        dur = meta.get("duration_sec", 0.0)

        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Duration", f"{dur:.1f}s")
        with m2:
            st.metric("Tracked Entities", f"{len(tracks)}")
        with m3:
            st.metric("Detected Relations", f"{len(relations)}")
        with m4:
            st.metric("Frame Rate", f"{fps:.1f} FPS")

        # Tabs Layout
        tab_video, tab_rels, tab_tracks, tab_json, tab_render = st.tabs([
            "🎬 Visualizer Video",
            "📊 Detected Relationships",
            "🚶 Tracked Entities",
            "📄 JSON Specification",
            "⚡ Fast Offline Re-render"
        ])

        with tab_video:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.subheader("Raw Surveillance Source")
                if selected_video_path and os.path.exists(selected_video_path):
                    st.video(selected_video_path)
            with col_v2:
                st.subheader("RelateAnything Visualizer")
                st.video(expected_mp4)

        with tab_rels:
            st.subheader("Confirmed Visual Relationships")
            if relations:
                table_rows = []
                for idx, r in enumerate(relations, 1):
                    t_range = r.get("time_range", {})
                    t_str = f"{t_range.get('start_sec', 0.0):.1f}s -> {t_range.get('end_sec', 0.0):.1f}s"
                    table_rows.append({
                        "No.": idx,
                        "Subject": f"{r['subject_id']} ({r.get('subject_class', '')})",
                        "Predicate": r["predicate"],
                        "Object": f"{r['object_id']} ({r.get('object_class', '')})",
                        "Confidence": f"{r.get('confidence', 0.0):.2f}",
                        "Consistency": f"{r.get('temporal_consistency', 1.0):.2f}",
                        "Active Time": t_str,
                        "Frames Count": len(r.get("active_frames", []))
                    })
                df_rels = pd.DataFrame(table_rows)
                st.dataframe(df_rels, use_container_width=True, hide_index=True)
            else:
                st.info("No interactive visual relationships detected for this temporal window.")

        with tab_tracks:
            st.subheader("Entity Trajectories & Attributes")
            track_rows = []
            for tid, tinfo in tracks.items():
                track_rows.append({
                    "Track ID": tid,
                    "Class": tinfo.get("class", ""),
                    "Type": tinfo.get("type", ""),
                    "Tracked Frames": tinfo.get("total_frames_tracked", len(tinfo.get("trajectories", {})))
                })
            df_tracks = pd.DataFrame(track_rows)
            st.dataframe(df_tracks, use_container_width=True, hide_index=True)

        with tab_json:
            st.subheader("Structured JSON Payload")
            st.download_button(
                label="📥 Download Structured JSON",
                data=json.dumps(data, indent=2, ensure_ascii=False),
                file_name=f"{video_name}_relations.json",
                mime="application/json"
            )
            st.json(data)

        with tab_render:
            st.subheader("Offline Renderer (Decoupled Pure OpenCV)")
            st.write("Renders the visualizer video directly from the JSON trajectories without executing neural network models.")
            out_rerender_name = st.text_input("Output Re-rendered Video Path:", value=os.path.join(expected_out_dir, f"{video_name}_from_json.mp4"))
            if st.button("Render MP4 from JSON"):
                with st.spinner("Rendering MP4 from JSON..."):
                    try:
                        res_p = render_video_from_json(selected_video_path, expected_json, out_rerender_name)
                        st.success(f"Rendered in ~3 seconds! Saved to: {res_p}")
                        st.video(res_p)
                    except Exception as e:
                        st.error(f"Render failed: {e}")
    else:
        st.info("No prior results found for this video. Please click 'Run Pipeline Analysis' in the sidebar to generate results.")


if __name__ == "__main__":
    main()
