import os
import sys
import json
import urllib.parse
import subprocess
import streamlit as st
import streamlit.components.v1 as components
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
        background: 
            radial-gradient(circle at 14% 15%, rgba(37, 99, 235, 0.16) 0%, transparent 45%),
            radial-gradient(circle at 86% 18%, rgba(129, 140, 248, 0.12) 0%, transparent 45%),
            radial-gradient(circle at 50% 88%, rgba(6, 182, 212, 0.12) 0%, transparent 50%),
            radial-gradient(circle at 35% 55%, rgba(168, 85, 247, 0.08) 0%, transparent 42%),
            linear-gradient(160deg, #0b111e 0%, #070c16 50%, #050810 100%) !important;
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
       1. BOX TỔNG (VÙNG VIỀN XANH LÁ) - LIQUID GLASS ĐỘNG CHUẨN 1 MÀN HÌNH
       ========================================================================= */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .master-box-marker) {
        background: linear-gradient(160deg, rgba(14, 22, 40, 0.72) 0%, rgba(9, 15, 28, 0.85) 100%) !important;
        backdrop-filter: blur(16px) saturate(180%) contrast(105%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(180%) contrast(105%) !important;
        border: 2px solid #22c55e !important; /* Viền xanh lá công nghệ rực rỡ */
        border-radius: 12px !important;
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.35), inset 0 1.5px 1px rgba(255, 255, 255, 0.25), inset 0 0 15px rgba(34, 197, 94, 0.08) !important;
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
       2. BOX TRỌNG TÂM 1: TIÊU ĐỀ (HEADER BOX) - LIQUID GLASS
          Gồm 2 box con: Box con tiêu đề bên trái & Box con thiết lập bên phải
       ========================================================================= */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .header-box-marker) {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(20, 30, 52, 0.50) 50%, rgba(10, 16, 30, 0.70) 100%) !important;
        backdrop-filter: blur(14px) saturate(180%) contrast(105%) !important;
        -webkit-backdrop-filter: blur(14px) saturate(180%) contrast(105%) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
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
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25), inset 0 1.5px 0.5px rgba(255, 255, 255, 0.30) !important;
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
    }

    /* Box con bên trái: Tiêu đề - Viền tím công nghệ, rê chuột LÓA SÁNG LIQUID GLASS */
    .header-subbox-title {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(168, 85, 247, 0.06) 35%, rgba(12, 18, 34, 0.70) 100%) !important;
        backdrop-filter: blur(14px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(14px) saturate(180%) !important;
        border: 1px solid rgba(168, 85, 247, 0.50) !important; /* Viền tím công nghệ */
        border-radius: 8px !important;
        padding: 0 18px !important;
        height: 38px !important;
        line-height: 38px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 12px !important;
        box-sizing: border-box !important;
        box-shadow: 0 0 8px rgba(168, 85, 247, 0.18), inset 0 1.5px 1px rgba(255, 255, 255, 0.45) !important;
        margin: 0 !important;
        transform: translateY(-8px) !important; /* Bù trừ chính xác 8px lệch do margin âm của Streamlit Markdown */
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: default !important;
        white-space: nowrap !important;
    }
    .header-subbox-title:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.24) 0%, rgba(192, 132, 252, 0.22) 35%, rgba(255, 255, 255, 0.06) 60%, rgba(35, 20, 55, 0.72) 100%) !important;
        border-color: #C084FC !important; /* Màu viền tím sáng rực rỡ khi rê chuột */
        box-shadow: 0 0 20px rgba(192, 132, 252, 0.65), inset 0 2px 1.5px rgba(255, 255, 255, 0.90), inset 0 0 16px rgba(192, 132, 252, 0.30) !important;
    }
    .header-title-text {
        font-size: 1.30rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        line-height: 1 !important;
        display: inline-flex !important;
        align-items: center !important;
        letter-spacing: -0.015em !important;
    }
    .header-dot-sep {
        color: #64748B !important;
        font-size: 1.15rem !important;
        line-height: 1 !important;
        display: inline-flex !important;
        align-items: center !important;
    }
    .header-subtitle-text {
        font-size: 0.94rem !important;
        font-weight: 500 !important;
        color: #94A3B8 !important;
        line-height: 1 !important;
        display: inline-flex !important;
        align-items: center !important;
    }

    /* Box con bên phải: Nút Thiết Lập */
    div[data-testid="stPopover"],
    .stPopover {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
        margin: 0 !important;
        width: 100% !important;
        height: 100% !important;
    }
    div[data-testid="stPopover"] > div,
    .stPopover > div {
        width: 100% !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
        margin: 0 !important;
    }
    button[data-testid="stPopoverButton"],
    div[data-testid="stPopover"] button,
    .stPopover button {
        height: 38px !important;
        line-height: 38px !important;
        padding: 0 14px !important;
        margin: 0 !important;
        border-radius: 8px !important;
        width: 100% !important;
        white-space: nowrap !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        border-style: solid !important;
        border-width: 1px !important;
        cursor: pointer !important;
        backdrop-filter: blur(18px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(190%) !important;
        animation: settings-radar-pulse 2.2s infinite ease-in-out !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    @keyframes settings-radar-pulse {
        0%, 100% {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(14, 165, 233, 0.12) 40%, rgba(15, 23, 42, 0.65) 100%);
            border-color: rgba(56, 189, 248, 0.35);
            box-shadow: 0 0 6px rgba(56, 189, 248, 0.15), inset 0 1.2px 0.8px rgba(255, 255, 255, 0.4);
            color: #E2E8F0;
        }
        50% {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.18) 0%, rgba(56, 189, 248, 0.25) 40%, rgba(15, 23, 42, 0.70) 100%);
            border-color: #38BDF8;
            box-shadow: 0 0 18px 2px rgba(56, 189, 248, 0.50), inset 0 1.8px 1px rgba(255, 255, 255, 0.7);
            color: #FFFFFF;
        }
    }

    button[data-testid="stPopoverButton"]:hover,
    div[data-testid="stPopover"] button:hover,
    .stPopover button:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.25) 0%, rgba(56, 189, 248, 0.25) 35%, rgba(255, 255, 255, 0.06) 60%, rgba(15, 25, 45, 0.75) 100%) !important;
        border-color: #38BDF8 !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.65), inset 0 2px 1.5px rgba(255, 255, 255, 0.90), inset 0 0 14px rgba(56, 189, 248, 0.30) !important;
    }

    /* Khi đang mở popup cài đặt thì giữ sáng cố định, dừng nhấp nháy */
    button[data-testid="stPopoverButton"][aria-expanded="true"],
    div[data-testid="stPopover"] button[aria-expanded="true"],
    .stPopover button[aria-expanded="true"] {
        animation: none !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.22) 0%, rgba(56, 189, 248, 0.22) 40%, rgba(15, 23, 42, 0.75) 100%) !important;
        border-color: #38BDF8 !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.45), inset 0 1.5px 1px rgba(255, 255, 255, 0.75) !important;
    }

    /* Khi đã mở ra rồi thì tắt nhấp nháy vĩnh viễn trong phiên (trở về mặc định viền xanh như 4 box) */
    button[data-testid="stPopoverButton"].pulse-stopped,
    div[data-testid="stPopover"] button.pulse-stopped,
    .stPopover button.pulse-stopped {
        animation: none !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(15, 23, 42, 0.65) 100%) !important;
        border-color: rgba(56, 189, 248, 0.45) !important;
        box-shadow: 0 0 6px rgba(56, 189, 248, 0.12), inset 0 1.2px 0.8px rgba(255, 255, 255, 0.4) !important;
        color: #F8FAFC !important;
    }
    button[data-testid="stPopoverButton"].pulse-stopped:hover,
    div[data-testid="stPopover"] button.pulse-stopped:hover,
    .stPopover button.pulse-stopped:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.25) 0%, rgba(56, 189, 248, 0.25) 35%, rgba(255, 255, 255, 0.06) 60%, rgba(15, 25, 45, 0.75) 100%) !important;
        border-color: #38BDF8 !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.65), inset 0 2px 1.5px rgba(255, 255, 255, 0.90), inset 0 0 14px rgba(56, 189, 248, 0.30) !important;
    }

    /* =========================================================================
       ANIMATED HAMBURGER MENU ICON (Biến hình ☰ thành ✕ khi mở menu)
       ========================================================================= */
    .custom-hamburger {
        display: inline-flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        width: 15px !important;
        height: 12px !important;
        position: relative !important;
        margin-right: 9px !important;
        vertical-align: middle !important;
        transition: transform 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
        flex-shrink: 0 !important;
    }
    .custom-hamburger .hb-line {
        display: block !important;
        height: 2px !important;
        width: 100% !important;
        background: #38BDF8 !important;
        border-radius: 2px !important;
        box-shadow: 0 0 5px rgba(56, 189, 248, 0.6) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        transform-origin: center !important;
    }
    /* Khi rê chuột vào nút: Thanh giữa thu ngắn tạo hiệu ứng nhún nhảy sinh động */
    button[data-testid="stPopoverButton"]:hover .custom-hamburger .hb-mid {
        width: 60% !important;
        box-shadow: 0 0 8px rgba(56, 189, 248, 0.9) !important;
    }
    /* KHI MỞ MENU (aria-expanded="true"): Xoay 90 độ và biến hình mượt mà thành dấu ✕ */
    button[data-testid="stPopoverButton"][aria-expanded="true"] .custom-hamburger,
    .custom-hamburger.is-open {
        transform: rotate(90deg) !important;
    }
    button[data-testid="stPopoverButton"][aria-expanded="true"] .custom-hamburger .hb-top,
    .custom-hamburger.is-open .hb-top {
        transform: translateY(5px) rotate(45deg) !important;
        background: #7DD3FC !important;
        box-shadow: 0 0 8px rgba(125, 211, 252, 0.8) !important;
    }
    button[data-testid="stPopoverButton"][aria-expanded="true"] .custom-hamburger .hb-mid,
    .custom-hamburger.is-open .hb-mid {
        opacity: 0 !important;
        transform: scale(0) !important;
    }
    button[data-testid="stPopoverButton"][aria-expanded="true"] .custom-hamburger .hb-bot,
    .custom-hamburger.is-open .hb-bot {
        transform: translateY(-5px) rotate(-45deg) !important;
        background: #7DD3FC !important;
        box-shadow: 0 0 8px rgba(125, 211, 252, 0.8) !important;
    }

    /* Ẩn hoàn toàn mũi tên chevron bên phải (cả Material Symbols font icon lẫn SVG) */
    button[data-testid="stPopoverButton"] [data-testid="stIconMaterial"],
    button[data-testid="stPopoverButton"] svg,
    button[data-testid="stPopoverButton"] div[aria-hidden="true"],
    button[data-testid="stPopoverButton"] span[class*="stIconMaterial"],
    div[data-testid="stPopover"] button [data-testid="stIconMaterial"],
    div[data-testid="stPopover"] button svg,
    div[data-testid="stPopover"] button div[aria-hidden="true"],
    .stPopover button [data-testid="stIconMaterial"],
    .stPopover button svg,
    .stPopover button div[aria-hidden="true"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        pointer-events: none !important;
        font-size: 0 !important;
        line-height: 0 !important;
    }

    /* Bảng thiết lập mở rộng (Popover Body): Hiệu ứng kính mờ xuyên thấu (Glassmorphism) */
    div[data-testid="stPopoverBody"] {
        z-index: 1000000 !important;
        background: linear-gradient(145deg, rgba(28, 45, 75, 0.65) 0%, rgba(15, 25, 46, 0.76) 50%, rgba(10, 16, 32, 0.84) 100%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(190%) contrast(105%) !important;
        backdrop-filter: blur(20px) saturate(190%) contrast(105%) !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        border-radius: 14px !important;
        box-shadow: 
            0 24px 60px rgba(0, 0, 0, 0.65), 
            0 0 20px rgba(56, 189, 248, 0.20), 
            inset 0 1.8px 1px rgba(255, 255, 255, 0.60), 
            inset 0 -1px 1px rgba(0, 0, 0, 0.40) !important;
        width: 860px !important;
        max-width: 92vw !important;
        padding: 16px 20px !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stPopoverBody"]:hover {
        background: linear-gradient(145deg, rgba(36, 58, 95, 0.74) 0%, rgba(56, 189, 248, 0.14) 35%, rgba(14, 24, 45, 0.82) 100%) !important;
        border-color: #38BDF8 !important; /* Rê chuột vào bảng mở rộng thì viền sáng rực rỡ màu xanh và lóa nền */
        box-shadow: 
            0 28px 70px rgba(0, 0, 0, 0.70), 
            0 0 35px rgba(56, 189, 248, 0.45), 
            inset 0 2px 1.5px rgba(255, 255, 255, 0.85), 
            inset 0 0 20px rgba(56, 189, 248, 0.20) !important;
    }

    /* Widget bên trong Bảng thiết lập Popover */
    div[data-testid="stPopoverBody"] div[data-testid="stRadio"] > div {
        background: rgba(15, 23, 42, 0.45) !important;
        padding: 6px 12px !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        box-shadow: inset 0 1px 0.5px rgba(255, 255, 255, 0.15) !important;
    }
    div[data-testid="stPopoverBody"] div[data-testid="stFileUploaderDropzone"] {
        background: rgba(15, 23, 42, 0.45) !important;
        border: 1px dashed rgba(56, 189, 248, 0.40) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stPopoverBody"] div[data-testid="stFileUploaderDropzone"]:hover {
        background: rgba(56, 189, 248, 0.10) !important;
        border-color: #38BDF8 !important;
    }
    div[data-testid="stPopoverBody"] button[kind="primary"],
    button[kind="primary"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.22) 0%, rgba(37, 99, 235, 0.85) 40%, rgba(14, 116, 144, 0.85) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.60) !important;
        box-shadow: 0 4px 18px rgba(37, 99, 235, 0.45), inset 0 1.5px 1px rgba(255, 255, 255, 0.70) !important;
        backdrop-filter: blur(12px) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stPopoverBody"] button[kind="primary"]:hover,
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.32) 0%, rgba(59, 130, 246, 0.95) 40%, rgba(6, 182, 212, 0.95) 100%) !important;
        border-color: #38BDF8 !important;
        box-shadow: 0 6px 24px rgba(56, 189, 248, 0.60), inset 0 2px 1.5px rgba(255, 255, 255, 0.90) !important;
        transform: translateY(-1px) !important;
    }

    /* Component iframe controller: ẩn hoàn toàn, không chiếm diện tích */
    div[data-testid="stCustomComponentV1"]:has(iframe) {
        position: absolute !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }

    /* =========================================================================
       3. HAI BOX TRỌNG TÂM DƯỚI: TRÁI & PHẢI (CHUẨN ĐỐI XỨNG TUYỆT ĐỐI - LIQUID GLASS)
       ========================================================================= */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .zone-left-marker),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .zone-right-marker) {
        background: linear-gradient(140deg, rgba(255, 255, 255, 0.04) 0%, rgba(18, 28, 48, 0.45) 40%, rgba(10, 16, 30, 0.70) 100%) !important;
        backdrop-filter: blur(14px) saturate(180%) contrast(105%) !important;
        -webkit-backdrop-filter: blur(14px) saturate(180%) contrast(105%) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
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
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.30), inset 0 1.5px 0.5px rgba(255, 255, 255, 0.30) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .zone-left-marker) > div[data-testid="stVerticalBlock"],
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .zone-right-marker) > div[data-testid="stVerticalBlock"] {
        gap: 6px !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* Sub-box 1 (Trên): Cả trái & phải cao chuẩn 58px - Liquid Glass */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .left-subbox1-marker),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .right-subbox1-marker) {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(14, 22, 38, 0.55) 50%, rgba(8, 14, 26, 0.72) 100%) !important;
        backdrop-filter: blur(12px) saturate(170%) !important;
        -webkit-backdrop-filter: blur(12px) saturate(170%) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        min-height: 58px !important;
        max-height: 58px !important;
        height: 58px !important;
        width: 100% !important;
        box-sizing: border-box !important;
        padding: 6px 8px !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-shrink: 0 !important;
        overflow: hidden !important;
        box-shadow: inset 0 1.2px 0.5px rgba(255, 255, 255, 0.25), 0 4px 16px rgba(0, 0, 0, 0.20) !important;
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

    /* Sub-box 2 (Dưới): Cả trái & phải cao chuẩn 426px - Liquid Glass */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .left-subbox2-marker),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > .right-subbox2-marker) {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(14, 22, 38, 0.55) 50%, rgba(8, 14, 26, 0.72) 100%) !important;
        backdrop-filter: blur(12px) saturate(170%) !important;
        -webkit-backdrop-filter: blur(12px) saturate(170%) !important;
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
        box-shadow: inset 0 1.2px 0.5px rgba(255, 255, 255, 0.20), 0 4px 16px rgba(0, 0, 0, 0.20) !important;
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
       SUB-BOX 1 (TRÁI): 4 NÚT CHỈ SỐ METRICS (CAO 44PX) - LIQUID GLASS HOVER FLARE
       ========================================================================= */
    .metric-box-item button,
    div:has(> div > div > div > .left-subbox1-marker) button,
    div:has(.left-subbox1-marker) .stButton > button {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(15, 23, 42, 0.65) 100%) !important;
        backdrop-filter: blur(14px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(14px) saturate(180%) !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        border-radius: 6px !important;
        color: #F1F5F9 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        height: 44px !important;
        padding: 0 4px !important;
        width: 100% !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        box-shadow: 0 0 6px rgba(56, 189, 248, 0.12), inset 0 1px 0.5px rgba(255, 255, 255, 0.35) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .metric-box-item button:hover,
    div:has(> div > div > div > .left-subbox1-marker) button:hover,
    div:has(.left-subbox1-marker) .stButton > button:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.24) 0%, rgba(56, 189, 248, 0.18) 35%, rgba(20, 32, 54, 0.75) 100%) !important;
        border-color: #38BDF8 !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.45), inset 0 1.8px 1px rgba(255, 255, 255, 0.85), inset 0 0 10px rgba(56, 189, 248, 0.25) !important;
        color: #FFFFFF !important;
    }
    /* Các con số chỉ số: Đậm hơn và có màu xanh cyan (#38BDF8) như màu viền */
    .metric-box-item button strong,
    .metric-box-item button b,
    .metric-box-item button span.stMarkdownColoredText,
    div:has(.left-subbox1-marker) button strong,
    div:has(.left-subbox1-marker) button b {
        color: #38BDF8 !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
    }
    .metric-box-item button:hover strong,
    .metric-box-item button:hover b,
    .metric-box-item button:hover span.stMarkdownColoredText,
    div:has(.left-subbox1-marker) button:hover strong,
    div:has(.left-subbox1-marker) button:hover b {
        color: #38BDF8 !important;
        text-shadow: 0 0 8px rgba(56, 189, 248, 0.6) !important;
    }

    /* =========================================================================
       SUB-BOX 1 (PHẢI): 2 NÚT SWITCHER TƯƠNG ĐƯƠNG 4 METRICS (CAO 44PX) - LIQUID GLASS
       ========================================================================= */
    .switcher-btn-secondary button {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(15, 23, 42, 0.65) 100%) !important;
        backdrop-filter: blur(14px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(14px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 6px !important;
        color: #F1F5F9 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        height: 44px !important;
        padding: 0 4px !important;
        width: 100% !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        box-shadow: inset 0 1px 0.5px rgba(255, 255, 255, 0.3) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .switcher-btn-secondary button:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.20) 0%, rgba(56, 189, 248, 0.15) 35%, rgba(24, 36, 60, 0.75) 100%) !important;
        border-color: rgba(56, 189, 248, 0.55) !important;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.35), inset 0 1.5px 1px rgba(255, 255, 255, 0.75) !important;
        color: #FFFFFF !important;
    }
    .switcher-btn-primary button {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.16) 0%, rgba(56, 189, 248, 0.22) 40%, rgba(15, 28, 48, 0.75) 100%) !important;
        backdrop-filter: blur(14px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(14px) saturate(190%) !important;
        border: 1px solid rgba(56, 189, 248, 0.55) !important;
        border-radius: 6px !important;
        color: #38BDF8 !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        height: 44px !important;
        padding: 0 6px !important;
        width: 100% !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.28), inset 0 1.5px 1px rgba(255, 255, 255, 0.6) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .switcher-btn-primary button:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.24) 0%, rgba(56, 189, 248, 0.28) 40%, rgba(15, 28, 48, 0.80) 100%) !important;
        border-color: #38BDF8 !important;
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.50), inset 0 1.8px 1px rgba(255, 255, 255, 0.85) !important;
        color: #FFFFFF !important;
    }

    /* Thẻ hiển thị quan hệ trong danh sách - Liquid Glass */
    .rel-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(15, 23, 42, 0.55) 100%);
        backdrop-filter: blur(12px) saturate(180%);
        -webkit-backdrop-filter: blur(12px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 8px;
        padding: 5px 8px;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: inset 0 1px 0.5px rgba(255, 255, 255, 0.25);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .rel-card:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.18) 0%, rgba(56, 189, 248, 0.14) 35%, rgba(20, 32, 54, 0.70) 100%);
        border-color: rgba(56, 189, 248, 0.55);
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.35), inset 0 1.5px 1px rgba(255, 255, 255, 0.75);
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

    /* Nút tua video nhỏ gọn - Liquid Glass */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(15, 23, 42, 0.65) 100%) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        border-radius: 6px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        padding: 3px 6px !important;
        color: #F1F5F9 !important;
        box-shadow: inset 0 1px 0.5px rgba(255, 255, 255, 0.25) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.22) 0%, rgba(56, 189, 248, 0.20) 35%, rgba(20, 32, 54, 0.80) 100%) !important;
        border-color: #38BDF8 !important;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.45), inset 0 1.5px 1px rgba(255, 255, 255, 0.80) !important;
        color: #FFFFFF !important;
    }

    /* =========================================================================
       BỘ LỌC VÀ BẢNG CHỌN DROPDOWN (SELECTBOX) - HIỆU ỨNG KÍNH MỜ (GLASSMORPHISM)
       Tương thích toàn diện Streamlit 1.66+ (React-Aria & Emotion) & BaseWeb
       ========================================================================= */

    /* 1. HỘP KÍCH HOẠT (TRIGGER BOX TRÊN GIAO DIỆN) */
    div[data-testid="stSelectbox"] .react-aria-Group,
    div[data-testid="stSelectbox"] .e1oobldh0,
    div[data-testid="stSelectbox"] div[class*="e1oobldh0"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] > div {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(15, 23, 42, 0.65) 100%) !important;
        background-color: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 8px !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        box-shadow: inset 0 1px 0.5px rgba(255, 255, 255, 0.25) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        color: #F8FAFC !important;
    }
    div[data-testid="stSelectbox"] .react-aria-Group:hover,
    div[data-testid="stSelectbox"] .e1oobldh0:hover,
    div[data-testid="stSelectbox"] div[class*="e1oobldh0"]:hover,
    div[data-testid="stSelectbox"] > div:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.16) 0%, rgba(56, 189, 248, 0.12) 35%, rgba(20, 32, 54, 0.75) 100%) !important;
        border-color: rgba(56, 189, 248, 0.65) !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.25), inset 0 1.5px 1px rgba(255, 255, 255, 0.65) !important;
    }

    /* TRIỆT TIÊU HOÀN TOÀN VIỀN HỒNG/ĐỎ KHI FOCUS/EXPANDED -> CHUYỂN SANG VIỀN CYAN PHÁT SÁNG */
    div[data-testid="stSelectbox"] .react-aria-Group[data-focus-within],
    div[data-testid="stSelectbox"] .react-aria-Group:focus-within,
    div[data-testid="stSelectbox"] .e1oobldh0[data-focus-within],
    div[data-testid="stSelectbox"] div[class*="e1oobldh0"][data-focus-within],
    div[data-testid="stSelectbox"] [data-focus-within],
    div[data-testid="stSelectbox"] div[aria-expanded="true"],
    div[data-testid="stSelectbox"] *:focus-within {
        border-color: #38BDF8 !important;
        outline: none !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.50), inset 0 1.5px 1px rgba(255, 255, 255, 0.75) !important;
    }

    /* Input text & icon caret */
    div[data-testid="stSelectbox"] input,
    div[data-testid="stSelectbox"] .react-aria-Input,
    div[data-testid="stSelectbox"] .e1oobldh1,
    div[data-testid="stSelectbox"] div[class*="e1oobldh1"] {
        color: #F8FAFC !important;
        font-weight: 500 !important;
        background: transparent !important;
    }
    div[data-testid="stSelectbox"] .e1oobldh2 svg,
    div[data-testid="stSelectbox"] svg {
        fill: #38BDF8 !important;
        color: #38BDF8 !important;
        transition: transform 0.25s ease !important;
    }
    div[data-testid="stSelectbox"]:has([aria-expanded="true"]) svg,
    div[data-testid="stSelectbox"] div[aria-expanded="true"] svg {
        transform: rotate(180deg) !important;
    }

    /* =========================================================================
       2. BẢNG THẢ XUỐNG (DROPDOWN POPOVER CONTAINER) - HIỆU ỨNG KÍNH MỜ (GLASSMORPHISM)
       Áp dụng cho portal container gắn ở document.body
       ========================================================================= */
    div[data-testid="stSelectboxVirtualDropdown"],
    [data-testid="stSelectboxVirtualDropdown"],
    div.react-aria-Popover,
    .react-aria-Popover[data-trigger="ComboBox"],
    div.e1oobldh4,
    div[class*="e1oobldh4"],
    div[data-baseweb="popover"],
    div[data-baseweb="menu"] {
        background: linear-gradient(145deg, rgba(28, 45, 75, 0.65) 0%, rgba(15, 25, 46, 0.76) 50%, rgba(10, 16, 32, 0.84) 100%) !important;
        background-color: rgba(15, 25, 46, 0.76) !important;
        -webkit-backdrop-filter: blur(20px) saturate(190%) contrast(105%) !important;
        backdrop-filter: blur(20px) saturate(190%) contrast(105%) !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        border-radius: 12px !important;
        box-shadow: 
            0 24px 60px rgba(0, 0, 0, 0.65), 
            0 0 20px rgba(56, 189, 248, 0.20), 
            inset 0 1.8px 1px rgba(255, 255, 255, 0.60), 
            inset 0 -1px 1px rgba(0, 0, 0, 0.40) !important;
        padding: 6px 4px !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
        z-index: 99999999 !important;
    }

    /* 3. DANH SÁCH CUỘN BÊN TRONG (REACT-ARIA LISTBOX HOẶC BASEWEB MENU) */
    div[data-testid="stSelectboxVirtualDropdown"] .react-aria-ListBox,
    div[data-testid="stSelectboxVirtualDropdown"] [role="listbox"],
    div[data-testid="stSelectboxVirtualDropdown"] div.e1oobldh5,
    div[data-testid="stSelectboxVirtualDropdown"] div[class*="e1oobldh5"],
    div[data-testid="stSelectboxVirtualDropdown"] ul,
    .react-aria-Popover .react-aria-ListBox,
    .react-aria-Popover [role="listbox"],
    .react-aria-Popover .e1oobldh5,
    .e1oobldh4 .e1oobldh5,
    .e1oobldh5,
    ul[data-baseweb="menu"],
    div[role="listbox"] {
        background: transparent !important;
        background-color: transparent !important;
        padding: 4px 2px !important;
        box-sizing: border-box !important;
        border: none !important;
    }

    /* 4. TỪNG DÒNG LỰA CHỌN (REACT-ARIA LISTBOXITEM HOẶC BASEWEB OPTION) */
    div[data-testid="stSelectboxVirtualDropdown"] .react-aria-ListBoxItem,
    div[data-testid="stSelectboxVirtualDropdown"] [role="option"],
    div[data-testid="stSelectboxVirtualDropdown"] div.e1oobldh7,
    div[data-testid="stSelectboxVirtualDropdown"] div[class*="e1oobldh7"],
    div[data-testid="stSelectboxVirtualDropdown"] li,
    .react-aria-Popover .react-aria-ListBoxItem,
    .react-aria-Popover [role="option"],
    .react-aria-Popover .e1oobldh7,
    .e1oobldh4 .e1oobldh7,
    .e1oobldh7,
    li[data-baseweb="option"],
    div[role="option"] {
        background: transparent !important;
        background-color: transparent !important;
        color: #E2E8F0 !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        margin: 2px 2px !important;
        padding: 7px 12px !important;
        border: 1px solid transparent !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: pointer !important;
        position: relative !important;
    }

    /* Khối highlight văn bản bên trong */
    div[data-testid="stSelectboxVirtualDropdown"] [data-item-hl],
    .react-aria-Popover [data-item-hl],
    .e1oobldh4 [data-item-hl],
    .e1oobldh7 [data-item-hl],
    .e1oobldh8 {
        background: transparent !important;
        background-color: transparent !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }

    /* 5. KHI RÊ CHUỘT HOẶC DI CHUYỂN PHÍM MŨI TÊN (HOVER & FOCUSED): LÓA SÁNG LIQUID GLASS */
    div[data-testid="stSelectboxVirtualDropdown"] .react-aria-ListBoxItem:hover,
    div[data-testid="stSelectboxVirtualDropdown"] .react-aria-ListBoxItem[data-hovered],
    div[data-testid="stSelectboxVirtualDropdown"] .react-aria-ListBoxItem[data-focused],
    div[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
    div[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-hovered],
    div[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused],
    div[data-testid="stSelectboxVirtualDropdown"] div.e1oobldh7:hover,
    div[data-testid="stSelectboxVirtualDropdown"] div.e1oobldh7[data-hovered],
    div[data-testid="stSelectboxVirtualDropdown"] div.e1oobldh7[data-focused],
    .react-aria-Popover .react-aria-ListBoxItem:hover,
    .react-aria-Popover .react-aria-ListBoxItem[data-hovered],
    .react-aria-Popover .react-aria-ListBoxItem[data-focused],
    .react-aria-Popover [role="option"]:hover,
    .react-aria-Popover [role="option"][data-hovered],
    .react-aria-Popover [role="option"][data-focused],
    .e1oobldh7:hover,
    .e1oobldh7[data-hovered],
    .e1oobldh7[data-focused],
    li[data-baseweb="option"]:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.22) 0%, rgba(56, 189, 248, 0.22) 35%, rgba(20, 36, 62, 0.70) 100%) !important;
        background-color: transparent !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(56, 189, 248, 0.50) !important;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.35), inset 0 1.2px 0.8px rgba(255, 255, 255, 0.70) !important;
    }

    /* Triệt tiêu màu nền xám đen cục bộ của Emotion highlight */
    div[data-testid="stSelectboxVirtualDropdown"] [data-hovered] [data-item-hl],
    div[data-testid="stSelectboxVirtualDropdown"] [data-focused] [data-item-hl],
    div[data-testid="stSelectboxVirtualDropdown"] :hover [data-item-hl],
    .react-aria-Popover [data-hovered] [data-item-hl],
    .react-aria-Popover [data-focused] [data-item-hl],
    .e1oobldh7[data-hovered] [data-item-hl],
    .e1oobldh7[data-focused] [data-item-hl] {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* 6. MỤC ĐANG ĐƯỢC CHỌN (SELECTED ITEM) */
    div[data-testid="stSelectboxVirtualDropdown"] [aria-selected="true"],
    div[data-testid="stSelectboxVirtualDropdown"] [data-selected],
    .react-aria-Popover [aria-selected="true"],
    .react-aria-Popover [data-selected],
    .e1oobldh7[aria-selected="true"],
    .e1oobldh7[data-selected],
    li[data-baseweb="option"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.16) 0%, rgba(56, 189, 248, 0.28) 40%, rgba(15, 28, 48, 0.80) 100%) !important;
        background-color: transparent !important;
        color: #38BDF8 !important;
        font-weight: 700 !important;
        border: 1px solid rgba(56, 189, 248, 0.60) !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.30), inset 0 1px 0.5px rgba(255, 255, 255, 0.60) !important;
    }

    /* 7. THANH CUỘN MẢNH CYAN */
    div[data-testid="stSelectboxVirtualDropdown"] *::-webkit-scrollbar,
    .react-aria-Popover *::-webkit-scrollbar,
    .e1oobldh4 *::-webkit-scrollbar,
    .e1oobldh5::-webkit-scrollbar {
        width: 5px !important;
    }
    div[data-testid="stSelectboxVirtualDropdown"] *::-webkit-scrollbar-track,
    .react-aria-Popover *::-webkit-scrollbar-track,
    .e1oobldh4 *::-webkit-scrollbar-track,
    .e1oobldh5::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.20) !important;
        border-radius: 4px !important;
    }
    div[data-testid="stSelectboxVirtualDropdown"] *::-webkit-scrollbar-thumb,
    .react-aria-Popover *::-webkit-scrollbar-thumb,
    .e1oobldh4 *::-webkit-scrollbar-thumb,
    .e1oobldh5::-webkit-scrollbar-thumb {
        background: rgba(56, 189, 248, 0.35) !important;
        border-radius: 4px !important;
    }
    div[data-testid="stSelectboxVirtualDropdown"] *::-webkit-scrollbar-thumb:hover,
    .react-aria-Popover *::-webkit-scrollbar-thumb:hover,
    .e1oobldh4 *::-webkit-scrollbar-thumb:hover,
    .e1oobldh5::-webkit-scrollbar-thumb:hover {
        background: #38BDF8 !important;
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


@st.cache_data
def get_ontology_filter_vocabularies():
    """Tải và phân loại ngữ nghĩa thực tế từ ontology 60 classes và 26 relations:
    - Chủ thể (Subjects): Thực thể có thể chủ động phát khởi hành vi (Con người, Động vật, Phương tiện tự hành).
    - Hành vi (Actions): Toàn bộ 26 quan hệ VidVRD đóng.
    - Đối tượng (Objects): Toàn bộ các lớp có thể là đối tượng chịu tác động."""
    objects_path = os.path.join("configs", "s_objects.json")
    relations_path = os.path.join("configs", "relations.json")

    raw_objects = []
    if os.path.exists(objects_path):
        with open(objects_path, "r", encoding="utf-8") as f:
            raw_objects = json.load(f)

    raw_relations = []
    if os.path.exists(relations_path):
        with open(relations_path, "r", encoding="utf-8") as f:
            raw_relations = json.load(f)

    # 1. Các lớp có thể đóng vai trò Chủ Thể tác động thực sự (Agents / Subjects):
    agent_subject_set = {
        # Con người (Chủ thể trọng tâm camera CCTV)
        "person", "child",
        # Động vật (Có khả năng hành động chủ động)
        "dog", "cat", "horse", "cattle", "sheep", "pig", "goat",
        "bird", "chicken", "duck", "rabbit", "snake", "turtle", "lizard",
        # Phương tiện di chuyển tự hành (Chủ thể trong va chạm / giao thông VidVRD)
        "car", "bus", "truck", "motorcycle", "bicycle", "scooter", "train", "skateboard", "baby_walker"
    }

    subject_classes = [c for c in raw_objects if c in agent_subject_set]
    sorted_subjects = ["person"] + (["child"] if "child" in subject_classes else []) + sorted(
        [c for c in subject_classes if c not in {"person", "child"}],
        key=lambda x: x.lower()
    )

    # 2. Hành vi (Actions / Predicates): Toàn bộ 26 vị từ VidVRD đóng
    sorted_predicates = sorted(raw_relations, key=lambda x: x.lower())

    # 3. Đối tượng chịu tác động (Patients / Objects): Toàn bộ 60 classes
    sorted_objects = sorted(raw_objects, key=lambda x: x.lower())

    return sorted_subjects, sorted_predicates, sorted_objects


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
            col_head_title, col_head_pop = st.columns([3.75, 1.25], vertical_alignment="center")

            with col_head_title:
                # Box con bên trái: Tiêu đề (Căn chính giữa tuyệt đối)
                st.markdown("""
                <div class="header-subbox-title">
                    <span class="header-title-text">VidVRD RelateAnything</span>
                    <span class="header-dot-sep">•</span>
                    <span class="header-subtitle-text">Phát Hiện Quan Hệ Thị Giác Trong Video</span>
                </div>
                """, unsafe_allow_html=True)

            with col_head_pop:
                # Box con bên phải: Nút Thiết Lập (Tự động kéo sát mép phải đúng chuẩn Streamlit gốc)
                settings_pop = st.popover("☰ THIẾT LẬP NGUỒN VIDEO & PHÂN TÍCH", use_container_width=True)

        # Tắt nhịp thở khi mở popup & đồng bộ style cho Dropdown Portal
        components.html("""
        <script>
        (function() {
            try {
                const pWin = window.parent;
                const pDoc = pWin.document;

                // Tiêm Style trực tiếp vào parent document head để đè toàn bộ Emotion & BaseWeb
                function injectDropdownStyles() {
                    try {
                        if (!pDoc.getElementById('st-liquid-glass-portal-style')) {
                            const styleEl = pDoc.createElement('style');
                            styleEl.id = 'st-liquid-glass-portal-style';
                            styleEl.textContent = `
                                div[data-testid="stSelectboxVirtualDropdown"],
                                [data-testid="stSelectboxVirtualDropdown"],
                                div.react-aria-Popover,
                                .react-aria-Popover[data-trigger="ComboBox"],
                                div.e1oobldh4,
                                div[class*="e1oobldh4"],
                                div[data-baseweb="popover"],
                                div[data-baseweb="menu"] {
                                    z-index: 99999999 !important;
                                    background: linear-gradient(145deg, rgba(28, 45, 75, 0.65) 0%, rgba(15, 25, 46, 0.76) 50%, rgba(10, 16, 32, 0.84) 100%) !important;
                                    background-color: rgba(15, 25, 46, 0.76) !important;
                                    -webkit-backdrop-filter: blur(20px) saturate(190%) contrast(105%) !important;
                                    backdrop-filter: blur(20px) saturate(190%) contrast(105%) !important;
                                    border: 1px solid rgba(56, 189, 248, 0.45) !important;
                                    border-radius: 12px !important;
                                    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.65), 0 0 20px rgba(56, 189, 248, 0.20), inset 0 1.8px 1px rgba(255, 255, 255, 0.60), inset 0 -1px 1px rgba(0, 0, 0, 0.40) !important;
                                }
                                div[data-testid="stSelectboxVirtualDropdown"] .react-aria-ListBox,
                                div[data-testid="stSelectboxVirtualDropdown"] [role="listbox"],
                                div[data-testid="stSelectboxVirtualDropdown"] div.e1oobldh5,
                                .react-aria-Popover .react-aria-ListBox,
                                .react-aria-Popover [role="listbox"],
                                .react-aria-Popover .e1oobldh5,
                                .e1oobldh4 .e1oobldh5,
                                .e1oobldh5,
                                ul[data-baseweb="menu"],
                                div[role="listbox"] {
                                    background: transparent !important;
                                    background-color: transparent !important;
                                }
                                div[data-testid="stSelectboxVirtualDropdown"] .react-aria-ListBoxItem,
                                div[data-testid="stSelectboxVirtualDropdown"] [role="option"],
                                div[data-testid="stSelectboxVirtualDropdown"] div.e1oobldh7,
                                .react-aria-Popover .react-aria-ListBoxItem,
                                .react-aria-Popover [role="option"],
                                .react-aria-Popover .e1oobldh7,
                                .e1oobldh4 .e1oobldh7,
                                .e1oobldh7,
                                li[data-baseweb="option"],
                                div[role="option"] {
                                    background: transparent !important;
                                    background-color: transparent !important;
                                    color: #E2E8F0 !important;
                                }
                                div[data-testid="stSelectboxVirtualDropdown"] [data-item-hl],
                                .react-aria-Popover [data-item-hl],
                                .e1oobldh4 [data-item-hl],
                                .e1oobldh7 [data-item-hl],
                                .e1oobldh8 {
                                    background: transparent !important;
                                    background-color: transparent !important;
                                }
                                div[data-testid="stSelectboxVirtualDropdown"] .react-aria-ListBoxItem:hover,
                                div[data-testid="stSelectboxVirtualDropdown"] .react-aria-ListBoxItem[data-hovered],
                                div[data-testid="stSelectboxVirtualDropdown"] .react-aria-ListBoxItem[data-focused],
                                div[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
                                div[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-hovered],
                                div[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused],
                                .react-aria-Popover .react-aria-ListBoxItem:hover,
                                .react-aria-Popover .react-aria-ListBoxItem[data-hovered],
                                .react-aria-Popover .react-aria-ListBoxItem[data-focused],
                                .e1oobldh7:hover,
                                .e1oobldh7[data-hovered],
                                .e1oobldh7[data-focused],
                                li[data-baseweb="option"]:hover {
                                    background: linear-gradient(135deg, rgba(255, 255, 255, 0.22) 0%, rgba(56, 189, 248, 0.22) 35%, rgba(20, 36, 62, 0.70) 100%) !important;
                                    background-color: transparent !important;
                                    color: #FFFFFF !important;
                                    border: 1px solid rgba(56, 189, 248, 0.50) !important;
                                    box-shadow: 0 0 14px rgba(56, 189, 248, 0.35), inset 0 1.2px 0.8px rgba(255, 255, 255, 0.70) !important;
                                }
                                div[data-testid="stSelectboxVirtualDropdown"] [aria-selected="true"],
                                div[data-testid="stSelectboxVirtualDropdown"] [data-selected],
                                .react-aria-Popover [aria-selected="true"],
                                .react-aria-Popover [data-selected],
                                .e1oobldh7[aria-selected="true"],
                                .e1oobldh7[data-selected],
                                li[data-baseweb="option"][aria-selected="true"] {
                                    background: linear-gradient(135deg, rgba(255, 255, 255, 0.16) 0%, rgba(56, 189, 248, 0.28) 40%, rgba(15, 28, 48, 0.80) 100%) !important;
                                    background-color: transparent !important;
                                    color: #38BDF8 !important;
                                    font-weight: 700 !important;
                                    border: 1px solid rgba(56, 189, 248, 0.60) !important;
                                }
                                div[data-testid="stSelectbox"] .react-aria-Group[data-focus-within],
                                div[data-testid="stSelectbox"] .react-aria-Group:focus-within,
                                div[data-testid="stSelectbox"] .e1oobldh0[data-focus-within],
                                div[data-testid="stSelectbox"] div[class*="e1oobldh0"][data-focus-within],
                                div[data-testid="stSelectbox"] *:focus-within {
                                    border-color: #38BDF8 !important;
                                    outline: none !important;
                                    box-shadow: 0 0 16px rgba(56, 189, 248, 0.50), inset 0 1.5px 1px rgba(255, 255, 255, 0.75) !important;
                                }
                            `;
                            pDoc.head.appendChild(styleEl);
                        }
                    } catch(e) {}
                }

                function checkPulse() {
                    try {
                        const btn = pDoc.querySelector('button[data-testid="stPopoverButton"]');
                        if (!btn) return;

                        if (btn.getAttribute('aria-expanded') === 'true') {
                            pWin._settingsPulseStopped = true;
                        }

                        if (pWin._settingsPulseStopped) {
                            btn.classList.add('pulse-stopped');
                        }

                        if (!btn._clickTrackingAttached) {
                            btn._clickTrackingAttached = true;
                            btn.addEventListener('click', function() {
                                pWin._settingsPulseStopped = true;
                                btn.classList.add('pulse-stopped');
                            });
                        }
                    } catch(e) {}
                }

                function setupHamburger() {
                    try {
                        const btn = pDoc.querySelector('button[data-testid="stPopoverButton"]');
                        if (!btn) return;

                        // 1. Thay thế ký tự ☰ thành icon animated hamburger
                        if (!btn.querySelector('.custom-hamburger')) {
                            const p = btn.querySelector('p') || btn.querySelector('[data-testid="stMarkdownContainer"]');
                            if (p && p.innerHTML.includes('☰')) {
                                p.innerHTML = p.innerHTML.replace('☰', '<span class="custom-hamburger"><span class="hb-line hb-top"></span><span class="hb-line hb-mid"></span><span class="hb-line hb-bot"></span></span>');
                            }
                        }

                        // 2. Ẩn hoàn toàn mũi tên chevron (cả Material Symbols icon lẫn SVG)
                        const extraIcons = btn.querySelectorAll('[data-testid="stIconMaterial"], svg');
                        extraIcons.forEach(icon => {
                            const wrapper = icon.closest('div[aria-hidden="true"]') || icon.parentElement;
                            if (wrapper) {
                                wrapper.style.setProperty('display', 'none', 'important');
                            }
                            icon.style.setProperty('display', 'none', 'important');
                        });
                    } catch(e) {}
                }

                // Giám sát dropdown đang mở trên document.body để triệt tiêu viền đỏ và nền đen tức thì
                function observeDropdowns() {
                    try {
                        const dropdowns = pDoc.querySelectorAll('[data-testid="stSelectboxVirtualDropdown"], .react-aria-Popover, .e1oobldh4, [data-baseweb="popover"]');
                        dropdowns.forEach(drop => {
                            drop.style.setProperty('z-index', '99999999', 'important');
                            drop.style.setProperty('background', 'linear-gradient(145deg, rgba(28, 45, 75, 0.65) 0%, rgba(15, 25, 46, 0.76) 50%, rgba(10, 16, 32, 0.84) 100%)', 'important');
                            drop.style.setProperty('backdrop-filter', 'blur(20px) saturate(190%) contrast(105%)', 'important');
                            drop.style.setProperty('-webkit-backdrop-filter', 'blur(20px) saturate(190%) contrast(105%)', 'important');
                            drop.style.setProperty('border', '1px solid rgba(56, 189, 248, 0.45)', 'important');
                            drop.style.setProperty('border-radius', '12px', 'important');
                        });

                        const popoverBodies = pDoc.querySelectorAll('[data-testid="stPopoverBody"]');
                        popoverBodies.forEach(pop => {
                            pop.style.setProperty('z-index', '1000000', 'important');
                        });

                        const focusedTriggers = pDoc.querySelectorAll('div[data-testid="stSelectbox"] [data-focus-within], div[data-testid="stSelectbox"] .react-aria-Group[data-focus-within]');
                        focusedTriggers.forEach(trig => {
                            trig.style.setProperty('border-color', '#38BDF8', 'important');
                            trig.style.setProperty('box-shadow', '0 0 16px rgba(56, 189, 248, 0.50), inset 0 1.5px 1px rgba(255, 255, 255, 0.75)', 'important');
                        });
                    } catch(e) {}
                }

                injectDropdownStyles();
                checkPulse();
                setupHamburger();
                observeDropdowns();
                setInterval(function() {
                    injectDropdownStyles();
                    checkPulse();
                    setupHamburger();
                    observeDropdowns();
                }, 200);
            } catch(err) {}
        })();
        </script>
        """, height=0, width=0)


        # Popover Settings Dropdown Menu (Overlay Popup)
        selected_video_path = None
        video_name = ""

        with settings_pop:
            st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #F8FAFC; margin-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 4px;'>⚙️ THIẾT LẬP NGUỒN VIDEO & PHÂN TÍCH</div>", unsafe_allow_html=True)
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

                    # Sub-box 1 (Trên): 4 Cột chỉ số Metrics chuẩn kích thước đối xứng
                    with st.container(border=True):
                        st.markdown('<div class="left-subbox1-marker"></div>', unsafe_allow_html=True)
                        c_l1, c_l2, c_l3, c_l4 = st.columns(4, gap="small")

                        with c_l1:
                            st.markdown('<div class="metric-box-item">', unsafe_allow_html=True)
                            st.button(f"THỜI LƯỢNG **{dur:.1f}s**", key="btn_m_dur", use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                        with c_l2:
                            st.markdown('<div class="metric-box-item">', unsafe_allow_html=True)
                            st.button(f"THỰC THỂ: **{len(tracks)}**", key="btn_m_tracks", use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                        with c_l3:
                            st.markdown('<div class="metric-box-item">', unsafe_allow_html=True)
                            st.button(f"QUAN HỆ: **{len(relations)}**", key="btn_m_rels", use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                        with c_l4:
                            st.markdown('<div class="metric-box-item">', unsafe_allow_html=True)
                            st.button(f"TỐC ĐỘ: **{fps:.1f} FPS**", key="btn_m_fps", use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)

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
                    st.markdown("""
                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; min-height: 440px; text-align: center; padding: 24px 20px;">
                        <div style="font-size: 2.6rem; margin-bottom: 12px; filter: drop-shadow(0 4px 12px rgba(0,0,0,0.5));">🔍</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin-bottom: 10px; letter-spacing: -0.01em;">
                            Chưa Có Kết Quả Phân Tích
                        </div>
                        <p style="color: #94A3B8; font-size: 0.88rem; line-height: 1.6; max-width: 440px; margin: 0 auto; text-align: center;">
                            Vui lòng bấm vào nút <b style="color: #38BDF8;">☰ THIẾT LẬP</b> ở góc trên bên phải để chọn video mẫu (hoặc tải video lên), sau đó bấm <b style="color: #F43F5E;">🚀 Bắt Đầu Phân Tích</b>.
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
                        # Build filter sets from full VidVRD Ontology & Semantic Agent Evaluation
                        ontology_subjects, ontology_predicates, ontology_objects = get_ontology_filter_vocabularies()
                        subject_options = ["Tất cả"] + ontology_subjects
                        predicate_options = ["Tất cả"] + ontology_predicates
                        object_options = ["Tất cả"] + ontology_objects

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
                            sub_cls = str(r.get("subject_class", "")).strip().lower()
                            pred_str = str(r.get("predicate", "")).strip().lower()
                            obj_cls = str(r.get("object_class", "")).strip().lower()

                            if sel_sub != "Tất cả" and sub_cls != sel_sub.strip().lower():
                                continue
                            if sel_pred != "Tất cả" and pred_str != sel_pred.strip().lower():
                                continue
                            if sel_obj != "Tất cả" and obj_cls != sel_obj.strip().lower():
                                continue
                            filtered_relations.append(r)

                        st.caption(f"Hiển thị **{len(filtered_relations)}** / {len(relations)} quan hệ phù hợp:")

                        if not filtered_relations:
                            st.markdown("""
                            <div style="background: rgba(15, 23, 42, 0.45); border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 8px; padding: 28px 14px; text-align: center; color: #94A3B8;">
                                <div style="font-size: 1.8rem; margin-bottom: 6px;">🔍</div>
                                <div style="color: #F8FAFC; font-weight: 700; font-size: 1.05rem; margin-bottom: 4px;">None</div>
                                <div style="font-size: 0.82rem;">Không tìm thấy quan hệ nào trong clip khớp với bộ lọc đã chọn.</div>
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

                                        col_card, col_btn = st.columns([3.5, 1.5], vertical_alignment="center")
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
                                            if st.button(f"Show on {int(start_sec)}s", key=f"seek_btn_{idx}", use_container_width=True):
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
                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; min-height: 440px; text-align: center; padding: 24px 20px;">
                        <div style="font-size: 2.4rem; margin-bottom: 12px; opacity: 0.65;">📊</div>
                        <div style="font-size: 1.10rem; font-weight: 600; color: #E2E8F0; margin-bottom: 8px;">
                            Chờ Dữ Liệu Quan Hệ
                        </div>
                        <p style="color: #64748B; font-size: 0.86rem; line-height: 1.6; max-width: 380px; margin: 0 auto; text-align: center;">
                            Dữ liệu danh sách hành vi và quan hệ thị giác giữa các thực thể sẽ xuất hiện tại đây sau khi video được phân tích.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

