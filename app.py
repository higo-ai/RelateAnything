import os
import sys
import json
import urllib.parse
import subprocess
import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="VidVRD RelateAnything",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling
st.markdown("""
<style>
    /* Ẩn header và footer mặc định của Streamlit */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
    }

    /* Chuẩn hóa nền và khống chế khung nhìn vừa khít 1 màn hình */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
        height: 100vh !important;
    }
    .stApp {
        background: radial-gradient(circle at 10% 10%, #1e293b 0%, #111827 50%, #0a0e17 100%) !important;
        color: #F8FAFC !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        height: 100vh !important;
        overflow: hidden !important;
    }
    .block-container {
        padding-top: 0.25rem !important;
        padding-bottom: 0.25rem !important;
        padding-left: 0.6rem !important;
        padding-right: 0.6rem !important;
        max-width: 100% !important;
        height: 100vh !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }

    /* Reset margin & padding của các phần tử con */
    div[data-testid="stMarkdownContainer"] p {
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stElementContainer"] {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Ẩn các marker thẻ định danh khỏi hiển thị */
    div[data-testid="stElementContainer"]:has(.master-box-marker),
    div[data-testid="stElementContainer"]:has(.header-box-marker),
    div[data-testid="stElementContainer"]:has(.zone-left-marker),
    div[data-testid="stElementContainer"]:has(.zone-right-marker),
    div[data-testid="stElementContainer"]:has(.left-subbox1-marker),
    div[data-testid="stElementContainer"]:has(.left-subbox2-marker),
    div[data-testid="stElementContainer"]:has(.right-subbox1-marker),
    div[data-testid="stElementContainer"]:has(.right-subbox2-marker) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* =========================================================================
       1. BOX TỔNG (VÙNG VIỀN XANH LÁ) - BAO TRÙM TOÀN BỘ, CHUẨN 1 MÀN HÌNH
       ========================================================================= */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .master-box-marker) {
        background: rgba(13, 20, 36, 0.75) !important;
        border: 2px solid #22c55e !important; /* Viền xanh lá công nghệ rực rỡ, sắc nét đúng bản vẽ */
        border-radius: 12px !important;
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.35), inset 0 0 15px rgba(34, 197, 94, 0.08) !important;
        padding: 6px 8px !important;
        height: calc(100vh - 0.6rem) !important;
        min-height: 575px !important;
        max-height: calc(100vh - 0.6rem) !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        overflow: hidden !important;
        margin: 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .master-box-marker) > div[data-testid="stVerticalBlock"] {
        gap: 6px !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }

    /* =========================================================================
       2. BOX TRỌNG TÂM 1: TIÊU ĐỀ (HEADER BOX)
          Gồm 2 box con: Box con tiêu đề bên trái & Box con thiết lập bên phải
       ========================================================================= */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .header-box-marker) {
        background: rgba(20, 28, 46, 0.60) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        padding: 0 12px !important;
        min-height: 50px !important;
        max-height: 50px !important;
        height: 50px !important;
        box-sizing: border-box !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        flex-shrink: 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .header-box-marker) > div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
        width: 100% !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .header-box-marker) div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        width: 100% !important;
        height: 100% !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .header-box-marker) div[data-testid="column"] {
        display: flex !important;
        align-items: center !important;
        height: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .header-box-marker) div[data-testid="column"]:first-child {
        justify-content: flex-start !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .header-box-marker) div[data-testid="column"]:last-child {
        justify-content: flex-end !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .header-box-marker) div[data-testid="stMarkdownContainer"],
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .header-box-marker) div[data-testid="stMarkdownContainer"] p {
        display: flex !important;
        align-items: center !important;
        height: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Box con bên trái: Tiêu đề - CĂN CHÍNH GIỮA TUYỆT ĐỐI */
    .header-subbox-title {
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 8px !important;
        padding: 0 16px !important;
        height: 38px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 10px !important;
        box-sizing: border-box !important;
        box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.05) !important;
        vertical-align: middle !important;
    }
    .header-title-text {
        font-size: 1.18rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        line-height: 1 !important;
        display: inline-flex !important;
        align-items: center !important;
        letter-spacing: -0.01em !important;
        transform: translateY(-1px) !important;
    }
    .header-dot-sep {
        color: #64748B !important;
        font-size: 1.05rem !important;
        line-height: 1 !important;
        display: inline-flex !important;
        align-items: center !important;
        transform: translateY(-1px) !important;
    }
    .header-subtitle-text {
        font-size: 0.86rem !important;
        font-weight: 400 !important;
        color: #94A3B8 !important;
        line-height: 1 !important;
        display: inline-flex !important;
        align-items: center !important;
        transform: translateY(-1px) !important;
    }

    /* Box con bên phải: Nút Thiết Lập */
    div[data-testid="stPopover"] {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
        margin: 0 !important;
        width: 100% !important;
        height: 100% !important;
    }
    div[data-testid="stPopover"] > button {
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        padding: 0 14px !important;
        margin: 0 !important;
        height: 38px !important;
        line-height: 38px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.05) !important;
        transition: border-color 0.2s ease, background 0.2s ease !important;
    }
    div[data-testid="stPopover"] > button:hover {
        background: rgba(56, 189, 248, 0.20) !important;
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
    }
    div[data-testid="stPopoverBody"] {
        background: rgba(15, 23, 42, 0.96) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 12px !important;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.65) !important;
        width: 860px !important;
        max-width: 92vw !important;
        padding: 16px 20px !important;
    }

    /* =========================================================================
       3. HAI BOX TRỌNG TÂM DƯỚI: TRÁI & PHẢI (CHUẨN ĐỐI XỨNG TUYỆT ĐỐI)
       ========================================================================= */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .zone-left-marker),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .zone-right-marker) {
        background: rgba(20, 28, 46, 0.55) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        min-height: 508px !important;
        max-height: 508px !important;
        height: 508px !important;
        box-sizing: border-box !important;
        padding: 6px 8px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        margin: 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .zone-left-marker) > div[data-testid="stVerticalBlock"],
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .zone-right-marker) > div[data-testid="stVerticalBlock"] {
        gap: 6px !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* Sub-box 1 (Trên): Cả trái & phải cao chuẩn 58px, thoáng đãng không bị cắt lẹm viền */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .left-subbox1-marker),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .right-subbox1-marker) {
        background: rgba(15, 23, 42, 0.60) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 8px !important;
        min-height: 58px !important;
        max-height: 58px !important;
        height: 58px !important;
        box-sizing: border-box !important;
        padding: 6px 8px !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-shrink: 0 !important;
        overflow: hidden !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .left-subbox1-marker) > div[data-testid="stVerticalBlock"],
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .right-subbox1-marker) > div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
        height: 100% !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* Sub-box 2 (Dưới): Cả trái & phải cao chuẩn 426px */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .left-subbox2-marker),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .right-subbox2-marker) {
        background: rgba(15, 23, 42, 0.60) !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        border-radius: 8px !important;
        min-height: 426px !important;
        max-height: 426px !important;
        height: 426px !important;
        box-sizing: border-box !important;
        padding: 8px 10px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        overflow: hidden !important;
        margin: 0 !important;
        flex-grow: 1 !important;
    }

    /* =========================================================================
       SUB-BOX 1 (TRÁI): 4 THẺ CHỈ SỐ BỰ VÀ RỘNG NHƯ BÊN PHẢI (CAO 44PX)
       ========================================================================= */
    .metrics-grid {
        display: grid !important;
        grid-template-columns: repeat(4, 1fr) !important;
        gap: 8px !important;
        width: 100% !important;
        align-items: center !important;
        box-sizing: border-box !important;
    }
    .metric-card {
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 6px !important;
        padding: 3px 6px !important;
        height: 44px !important;
        text-align: center !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        box-sizing: border-box !important;
        transition: border-color 0.2s ease, background 0.2s ease !important;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.40) !important;
        background: rgba(30, 41, 59, 0.80) !important;
    }
    .metric-label {
        font-size: 0.66rem !important;
        color: #94A3B8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
        font-weight: 600 !important;
        margin-bottom: 2px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        line-height: 1 !important;
    }
    .metric-value {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #38BDF8 !important;
        line-height: 1 !important;
    }

    /* =========================================================================
       SUB-BOX 2 (TRÁI): HEADER & KHUNG VIDEO THU NHỎ, CHỪA KHOẢNG TRỐNG VÙNG TRÊN
       ========================================================================= */
    .video-header-inline {
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        height: 26px !important;
        margin-bottom: 0 !important;
        flex-shrink: 0 !important;
    }
    .video-header-title {
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        color: #F1F5F9 !important;
        line-height: 1 !important;
    }
    .video-header-sep {
        color: #64748B !important;
        font-size: 0.85rem !important;
        line-height: 1 !important;
    }
    .video-download-link {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #38BDF8 !important;
        text-decoration: none !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 4px !important;
        transition: color 0.15s ease !important;
    }
    .video-download-link:hover {
        color: #7DD3FC !important;
        text-decoration: underline !important;
    }

    /* Khung Video: Đúng y chang khung chữ nhật trắng trong ảnh minh họa (cao 248px, chừa khoảng trên) */
    div[data-testid="stVideo"] {
        width: 86% !important;
        max-width: 440px !important;
        height: 248px !important;
        max-height: 248px !important;
        margin: 20px auto 0 auto !important; /* Chừa khoảng trống vùng trên đúng như minh họa */
        background: #000000 !important;
        border: 2px solid rgba(255, 255, 255, 0.35) !important; /* Viền sáng rõ nét như ảnh vẽ */
        border-radius: 8px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.7) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
        box-sizing: border-box !important;
        aspect-ratio: unset !important;
    }
    div[data-testid="stVideo"] video {
        width: 100% !important;
        height: 100% !important;
        max-width: 100% !important;
        max-height: 248px !important;
        object-fit: contain !important;
        background: #000000 !important;
        border-radius: 6px !important;
        display: block !important;
    }

    /* =========================================================================
       SUB-BOX 1 (PHẢI): 2 NÚT SWITCHER TƯƠNG ĐƯƠNG 4 METRICS (CAO 44PX)
       ========================================================================= */
    .switcher-btn-secondary button {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 6px !important;
        color: #94A3B8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        height: 44px !important;
        padding: 0 10px !important;
        width: 100% !important;
    }
    .switcher-btn-secondary button:hover {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #F1F5F9 !important;
    }
    .switcher-btn-primary button {
        background: rgba(56, 189, 248, 0.18) !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        border-radius: 6px !important;
        color: #38BDF8 !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        height: 44px !important;
        padding: 0 10px !important;
        width: 100% !important;
    }

    /* Thẻ hiển thị quan hệ trong danh sách */
    .rel-card {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 5px 8px;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.15s ease;
    }
    .rel-card:hover {
        background: rgba(30, 41, 59, 0.65);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .rel-entity {
        font-weight: 600;
        color: #F8FAFC;
        font-size: 0.80rem;
    }
    .rel-badge {
        display: inline-block;
        padding: 1px 6px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.72rem;
        letter-spacing: 0.02em;
    }
    .badge-combat {
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-transport {
        background: rgba(56, 189, 248, 0.2);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }
    .badge-luggage {
        background: rgba(52, 211, 153, 0.2);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.4);
    }
    .badge-posture {
        background: rgba(192, 132, 252, 0.2);
        color: #C084FC;
        border: 1px solid rgba(192, 132, 252, 0.4);
    }
    .badge-default {
        background: rgba(148, 163, 184, 0.2);
        color: #E2E8F0;
        border: 1px solid rgba(148, 163, 184, 0.4);
    }
    .rel-time {
        font-size: 0.68rem;
        color: #94A3B8;
        font-family: monospace;
    }

    /* Nút tua video nhỏ gọn */
    div[data-testid="stButton"] > button {
        border-radius: 6px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        padding: 3px 6px !important;
    }
</style>
""", unsafe_allow_html=True)


def get_available_benchmark_videos():
    video_dir = os.path.join("data", "videos")
    if not os.path.exists(video_dir):
        return []
    videos = [f for f in sorted(os.listdir(video_dir)) if f.endswith(".mp4")]
    def sort_key(name):
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 999
    return sorted(videos, key=sort_key)


def get_badge_class(pred: str) -> str:
    p = pred.lower()
    if p in {"hit", "kick", "knock", "fight", "push"}:
        return "badge-combat"
    if p in {"ride", "drive", "get_on", "get_off"}:
        return "badge-transport"
    if p in {"carry", "hold", "lift", "grab", "pull"}:
        return "badge-luggage"
    if p in {"stand", "sit", "walk", "touch"}:
        return "badge-posture"
    return "badge-default"


def main():
    if "video_seek_time" not in st.session_state:
        st.session_state["video_seek_time"] = 0
    if "seek_toast_msg" not in st.session_state:
        st.session_state["seek_toast_msg"] = None
    if "active_view_mode" not in st.session_state:
        st.session_state["active_view_mode"] = "cards"
    if "input_mode" not in st.session_state:
        st.session_state["input_mode"] = "Video mẫu"
    if "benchmark_choice" not in st.session_state:
        st.session_state["benchmark_choice"] = "video1.mp4"
    if "conf_thresh" not in st.session_state:
        st.session_state["conf_thresh"] = 0.20
    if "rel_conf_thresh" not in st.session_state:
        st.session_state["rel_conf_thresh"] = 0.15
    if "consistency_thresh" not in st.session_state:
        st.session_state["consistency_thresh"] = 0.25
    if "device" not in st.session_state:
        st.session_state["device"] = "auto"

    if st.session_state["seek_toast_msg"]:
        st.toast(st.session_state["seek_toast_msg"], icon="⏱")
        st.session_state["seek_toast_msg"] = None

    benchmark_videos = get_available_benchmark_videos()

    # =========================================================================
    # 1. VÙNG BOX TỔNG (VIỀN XANH LÁ) - CHUẨN MÀN HÌNH 1 TRANG
    # =========================================================================
    with st.container(border=True):
        st.markdown('<div class="master-box-marker"></div>', unsafe_allow_html=True)

        # =====================================================================
        # 2. BOX TRỌNG TÂM 1: TIÊU ĐỀ (HEADER BOX)
        #    Gồm 2 box con: Box con tiêu đề bên trái & Box con thiết lập bên phải
        # =====================================================================
        with st.container(border=True):
            st.markdown('<div class="header-box-marker"></div>', unsafe_allow_html=True)
            col_head_title, col_head_pop = st.columns([4.2, 0.8], vertical_alignment="center")

            with col_head_title:
                # Box con bên trái: Tiêu đề (Căn chính giữa tuyệt đối)
                st.markdown("""
                <div class="header-subbox-title">
                    <span class="header-title-text">VidVRD RelateAnything</span>
                    <span class="header-dot-sep">•</span>
                    <span class="header-subtitle-text">Hệ Thống Giám Sát Hành Vi & Quan Hệ Thị Giác</span>
                </div>
                """, unsafe_allow_html=True)

            with col_head_pop:
                # Box con bên phải: Nút Thiết Lập
                settings_pop = st.popover("☰ Thiết Lập", use_container_width=True)

        # Popover Settings Dropdown Menu (Overlay Popup)
        selected_video_path = None
        video_name = ""

        with settings_pop:
            st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #F8FAFC; margin-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 4px;'>⚙️ THIẾT LẬP HỆ THỐNG & MÔ HÌNH</div>", unsafe_allow_html=True)
            col_s1, col_s2, col_s3 = st.columns([1.0, 1.15, 1.05], gap="medium")

            # Column 1: Video Source
            with col_s1:
                st.markdown("<b style='font-size: 0.80rem; color: #38BDF8;'>📦 NGUỒN VIDEO</b>", unsafe_allow_html=True)
                src_choice = st.radio(
                    "Nguồn video:",
                    ["Video mẫu", "Tải video mới lên"],
                    index=0 if st.session_state["input_mode"] == "Video mẫu" else 1,
                    label_visibility="collapsed",
                    key="cfg_src_choice"
                )
                st.session_state["input_mode"] = src_choice

                if src_choice == "Video mẫu":
                    if benchmark_videos:
                        def_idx = benchmark_videos.index(st.session_state["benchmark_choice"]) if st.session_state["benchmark_choice"] in benchmark_videos else 0
                        sel_bench = st.selectbox("Chọn video mẫu:", benchmark_videos, index=def_idx, key="cfg_bench_select")
                        st.session_state["benchmark_choice"] = sel_bench
                        selected_video_path = os.path.join("data", "videos", sel_bench)
                        video_name = os.path.splitext(sel_bench)[0]
                    else:
                        st.warning("Không có video trong data/videos/")
                else:
                    up_file = st.file_uploader("Tải video lên (mp4, avi, mov):", type=["mp4", "avi", "mov"], key="cfg_up_file")
                    if up_file is not None:
                        up_dir = os.path.join("data", "uploads")
                        os.makedirs(up_dir, exist_ok=True)
                        selected_video_path = os.path.join(up_dir, up_file.name)
                        with open(selected_video_path, "wb") as f:
                            f.write(up_file.getbuffer())
                        video_name = os.path.splitext(up_file.name)[0]
                        st.success(f"Đã nạp: {up_file.name}")

            # Column 2: Thresholds
            with col_s2:
                st.markdown("<b style='font-size: 0.80rem; color: #38BDF8;'>🎛️ NGƯỠNG THIẾT LẬP</b>", unsafe_allow_html=True)
                yolo_conf = st.slider("YOLO Detection Conf:", 0.10, 0.70, float(st.session_state["conf_thresh"]), 0.05, key="cfg_yolo_conf")
                st.session_state["conf_thresh"] = yolo_conf

                rel_conf = st.slider("Relation Conf Thresh:", 0.10, 0.50, float(st.session_state["rel_conf_thresh"]), 0.05, key="cfg_rel_conf")
                st.session_state["rel_conf_thresh"] = rel_conf

                cons_thresh = st.slider("Consistency Thresh (Lọc flicker):", 0.10, 0.60, float(st.session_state["consistency_thresh"]), 0.05, key="cfg_cons_thresh")
                st.session_state["consistency_thresh"] = cons_thresh

            # Column 3: Compute Device
            with col_s3:
                st.markdown("<b style='font-size: 0.80rem; color: #38BDF8;'>⚡ THIẾT BỊ TÍNH TOÁN</b>", unsafe_allow_html=True)
                dev_options = ["auto", "cuda", "cpu"]
                cur_dev_idx = dev_options.index(st.session_state["device"]) if st.session_state["device"] in dev_options else 0
                sel_dev = st.radio(
                    "Chọn thiết bị:",
                    dev_options,
                    index=cur_dev_idx,
                    horizontal=True,
                    key="cfg_dev_radio",
                    format_func=lambda x: "Auto" if x == "auto" else ("CUDA" if x == "cuda" else "CPU")
                )
                st.session_state["device"] = sel_dev

                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
                run_btn = st.button("🚀 Bắt Đầu Phân Tích", type="primary", use_container_width=True, key="cfg_run_btn")

        # Resolve active video
        if not selected_video_path:
            if st.session_state["input_mode"] == "Video mẫu" and benchmark_videos:
                cur_b = st.session_state["benchmark_choice"] if st.session_state["benchmark_choice"] in benchmark_videos else benchmark_videos[0]
                selected_video_path = os.path.join("data", "videos", cur_b)
                video_name = os.path.splitext(cur_b)[0]

        # Pipeline Trigger
        if "cfg_run_btn" in st.session_state and st.session_state.cfg_run_btn:
            if not selected_video_path or not os.path.exists(selected_video_path):
                st.error("Vui lòng chọn hoặc tải lên một tệp video hợp lệ.")
                return

            with st.spinner(f"Đang phân tích RelateAnything trên {os.path.basename(selected_video_path)}..."):
                cmd = [
                    sys.executable, "pipeline_relate_anything.py",
                    "--video", selected_video_path,
                    "--mode", "roi",
                    "--conf", str(st.session_state["conf_thresh"]),
                    "--min_rel_conf", str(st.session_state["rel_conf_thresh"]),
                    "--min_consistency", str(st.session_state["consistency_thresh"]),
                    "--device", st.session_state["device"],
                    "--full_video"
                ]
                process = subprocess.run(cmd, capture_output=True, text=True)
                if process.returncode == 0:
                    st.session_state["video_seek_time"] = 0
                    st.success("Phân tích hoàn tất!")
                    st.rerun()
                else:
                    st.error(f"Lỗi phân tích (Mã: {process.returncode})")
                    with st.expander("Chi tiết lỗi"):
                        st.code(process.stderr or process.stdout)

        # Output paths
        expected_out_dir = os.path.join("data", "output", "full_video", video_name)
        expected_json = os.path.join(expected_out_dir, f"{video_name}_relations.json")
        expected_mp4 = os.path.join(expected_out_dir, f"{video_name}_visualizer.mp4")

        # =====================================================================
        # 3. DUAL ZONES: BOX TRỌNG TÂM TRÁI & BOX TRỌNG TÂM PHẢI (50/50 TWINS)
        # =====================================================================
        col_zone_left, col_zone_right = st.columns([1.0, 1.0], gap="medium")

        # =====================================================================
        # BOX TRỌNG TÂM BÊN TRÁI: (Sub-box 1: 4 Metrics | Sub-box 2: Video Box)
        # =====================================================================
        with col_zone_left:
            st.markdown('<div class="zone-left-marker"></div>', unsafe_allow_html=True)
            with st.container(border=True):
                if os.path.exists(expected_json) and os.path.exists(expected_mp4):
                    with open(expected_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    meta = data.get("video_metadata", {})
                    tracks = data.get("tracks", data.get("entities", {}))
                    relations = data.get("relations", [])
                    fps = meta.get("fps", 25.0)
                    dur = meta.get("duration_sec", 0.0)

                    # Sub-box 1 (Trên): 4 Thẻ chỉ số Metrics (Cao 58px, Thẻ cao 44px, Bự & Rộng)
                    with st.container(border=True):
                        st.markdown('<div class="left-subbox1-marker"></div>', unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metrics-grid">
                            <div class="metric-card">
                                <div class="metric-label">Thời Lượng Video</div>
                                <div class="metric-value">{dur:.1f}s</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Thực Thể Theo Dõi</div>
                                <div class="metric-value">{len(tracks)}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Quan Hệ Phát Hiện</div>
                                <div class="metric-value">{len(relations)}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Tốc Độ Khung Hình</div>
                                <div class="metric-value">{fps:.1f} FPS</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Sub-box 2 (Dưới): Box Video lớn (Cao 426px)
                    with st.container(border=True):
                        st.markdown('<div class="left-subbox2-marker"></div>', unsafe_allow_html=True)
                        json_str = json.dumps(data, indent=2, ensure_ascii=False)
                        json_encoded = urllib.parse.quote(json_str)

                        # Dòng tiêu đề inline header
                        st.markdown(f"""
                        <div class="video-header-inline">
                            <span class="video-header-title">Video Giám Sát Tương Tác</span>
                            <span class="video-header-sep">|</span>
                            <a href="data:application/json;charset=utf-8,{json_encoded}" download="{video_name}_relations.json" class="video-download-link">
                                Tải file json <span style="font-size: 0.95em;">📥</span>
                            </a>
                        </div>
                        """, unsafe_allow_html=True)

                        # KHUNG VIDEO NHỎ GỌN CÓ VIỀN SÁNG, CHỪA KHOẢNG TRỐNG PHÍA TRÊN ĐÚNG ẢNH MINH HỌA
                        current_seek = int(st.session_state.get("video_seek_time", 0))
                        st.video(expected_mp4, start_time=current_seek)

                else:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 36px 16px;">
                        <div style="font-size: 2.0rem; margin-bottom: 8px;">🔍</div>
                        <h4 style="color: #F8FAFC; margin-bottom: 6px;">Chưa Có Kết Quả Phân Tích Cho: {video_name}</h4>
                        <p style="color: #94A3B8; font-size: 0.85rem; max-width: 400px; margin: 0 auto;">
                            Bấm vào nút <b>☰ Thiết Lập</b> ở trên để chọn tham số và bấm <b>🚀 Bắt Đầu Phân Tích</b>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        # =====================================================================
        # BOX TRỌNG TÂM BÊN PHẢI: (Sub-box 1: 2 Switchers | Sub-box 2: Filters & Data)
        # =====================================================================
        with col_zone_right:
            st.markdown('<div class="zone-right-marker"></div>', unsafe_allow_html=True)
            with st.container(border=True):
                if os.path.exists(expected_json) and os.path.exists(expected_mp4):
                    # Sub-box 1 (Trên): 2 Nút chuyển đổi tab (Cao 58px, Nút cao 44px khớp với 4 metrics bên trái)
                    with st.container(border=True):
                        st.markdown('<div class="right-subbox1-marker"></div>', unsafe_allow_html=True)
                        c_sw1, c_sw2 = st.columns(2, gap="small")
                        is_cards = (st.session_state["active_view_mode"] == "cards")

                        with c_sw1:
                            st.markdown(f'<div class="{"switcher-btn-primary" if is_cards else "switcher-btn-secondary"}">', unsafe_allow_html=True)
                            if st.button("📋 Bộ Lọc & Danh Sách Quan Hệ", key="sw_btn_cards", use_container_width=True):
                                st.session_state["active_view_mode"] = "cards"
                                st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)

                        with c_sw2:
                            st.markdown(f'<div class="{"switcher-btn-secondary" if is_cards else "switcher-btn-primary"}">', unsafe_allow_html=True)
                            if st.button("📊 Xem Dưới Dạng Bảng Dữ Liệu Chi Tiết", key="sw_btn_table", use_container_width=True):
                                st.session_state["active_view_mode"] = "table"
                                st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)

                    # Sub-box 2 (Dưới): Bộ lọc & Dữ liệu Quan hệ (Cao 426px, khớp hoàn hảo với Video Box bên trái)
                    with st.container(border=True):
                        st.markdown('<div class="right-subbox2-marker"></div>', unsafe_allow_html=True)
                        # Build filter sets
                        subject_options = ["Tất cả"]
                        predicate_options = ["Tất cả"]
                        object_options = ["Tất cả"]

                        if relations:
                            raw_subs = {f"{r.get('subject_id', '')} {r.get('subject_class', '')}".strip() for r in relations}
                            raw_preds = {r.get("predicate", "").strip() for r in relations if r.get("predicate")}
                            raw_objs = {f"{r.get('object_id', '')} {r.get('object_class', '')}".strip() for r in relations}

                            subject_options.extend(sorted(list(raw_subs)))
                            predicate_options.extend(sorted(list(raw_preds)))
                            object_options.extend(sorted(list(raw_objs)))

                        # 3 Dropdowns bộ lọc
                        fc1, fc2, fc3 = st.columns(3)
                        with fc1:
                            sel_sub = st.selectbox("Chủ thể (Subject):", subject_options, index=0)
                        with fc2:
                            sel_pred = st.selectbox("Hành vi (Action):", predicate_options, index=0)
                        with fc3:
                            sel_obj = st.selectbox("Đối tượng (Object):", object_options, index=0)

                        # Filter relations
                        filtered_relations = []
                        for r in relations:
                            sub_str = f"{r.get('subject_id', '')} {r.get('subject_class', '')}".strip()
                            pred_str = r.get("predicate", "").strip()
                            obj_str = f"{r.get('object_id', '')} {r.get('object_class', '')}".strip()

                            if sel_sub != "Tất cả" and sub_str != sel_sub:
                                continue
                            if sel_pred != "Tất cả" and pred_str != sel_pred:
                                continue
                            if sel_obj != "Tất cả" and obj_str != sel_obj:
                                continue
                            filtered_relations.append(r)

                        st.caption(f"Hiển thị **{len(filtered_relations)}** / {len(relations)} quan hệ phù hợp:")

                        if not filtered_relations:
                            st.markdown("""
                            <div style="background: rgba(15, 23, 42, 0.45); border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 8px; padding: 20px 14px; text-align: center; color: #94A3B8; font-size: 0.82rem;">
                                🔍 Không có quan hệ nào khớp với bộ lọc hiện tại.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            if st.session_state["active_view_mode"] == "cards":
                                with st.container(height=275):
                                    for idx, r in enumerate(filtered_relations):
                                        sub_str = f"{r.get('subject_id', '')} {r.get('subject_class', '')}".strip()
                                        pred = r.get("predicate", "")
                                        conf = r.get("confidence", 0.0)
                                        obj_str = f"{r.get('object_id', '')} {r.get('object_class', '')}".strip()
                                        time_range = r.get("time_range", {})
                                        start_sec = time_range.get("start_sec", 0.0)
                                        end_sec = time_range.get("end_sec", 0.0)
                                        dur_rel = max(0.0, end_sec - start_sec)
                                        badge_cls = get_badge_class(pred)

                                        col_card, col_btn = st.columns([3.8, 1.2], vertical_alignment="center")
                                        with col_card:
                                            st.markdown(f"""
                                            <div class="rel-card">
                                                <div>
                                                    <span class="rel-entity">{sub_str}</span>
                                                    <span style="color: #64748B; margin: 0 4px;">&rarr;</span>
                                                    <span class="rel-badge {badge_cls}">{pred} ({conf:.2f})</span>
                                                    <span style="color: #64748B; margin: 0 4px;">&rarr;</span>
                                                    <span class="rel-entity">{obj_str}</span>
                                                </div>
                                                <div class="rel-time">
                                                    ⏱ {start_sec:.2f}s &rarr; {end_sec:.2f}s ({dur_rel:.2f}s)
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        with col_btn:
                                            if st.button(f"▶ Tua {int(start_sec)}s", key=f"seek_btn_{idx}", use_container_width=True):
                                                st.session_state["video_seek_time"] = int(start_sec)
                                                st.session_state["seek_toast_msg"] = f"Đang phát tại thời điểm: {int(start_sec)}s"
                                                st.rerun()

                            else:
                                table_rows = []
                                for r in filtered_relations:
                                    t = r.get("time_range", {})
                                    table_rows.append({
                                        "Chủ thể (Subject)": f"{r.get('subject_id', '')} {r.get('subject_class', '')}",
                                        "Hành vi (Action)": r.get("predicate", ""),
                                        "Đối tượng (Object)": f"{r.get('object_id', '')} {r.get('object_class', '')}",
                                        "Độ tin cậy": round(r.get("confidence", 0.0), 3),
                                        "Bắt đầu": f"{t.get('start_sec', 0.0):.2f}s",
                                        "Kết thúc": f"{t.get('end_sec', 0.0):.2f}s"
                                    })
                                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True, height=275)

                else:
                    st.markdown("""
                    <div style="text-align: center; padding: 40px 16px; color: #94A3B8;">
                        Dữ liệu quan hệ sẽ xuất hiện ở đây sau khi phân tích video.
                    </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

