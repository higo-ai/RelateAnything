# VidVRD RelateAnything: Hệ Thống Nhận Diện Quan Hệ Thị Giác Video Giám Sát CCTV

Hệ thống cung cấp một giải pháp đầu cuối toàn diện (**End-to-End Pipeline**) cho bài toán **Nhận diện Quan hệ Thị giác trong Video Giám sát Thực tế (Video Visual Relationship Detection - VidVRD)** trên các môi trường camera an ninh (CCTV). Hệ thống tích hợp luồng theo vết tách biệt người và vật thể (**Decoupled Causal Tracking**), thuật toán phân cụm tương tác không gian - thời gian (**STIC v2**), và mô hình nền tảng quan hệ thị giác từ vựng đóng **RelateAnything** (`maelic/relsgg-vits16plus`).

---

## MỤC LỤC
1. [Tổng Quan Kiến Trúc Hệ Thống](#1-tổng-quan-kiến-trúc-hệ-thống)
2. [Kho Thuật Toán Cốt Lõi & Bảng Tham Số Chuẩn Hóa](#2-kho-thuật-toán-cốt-lõi--bảng-tham-số-chuẩn-hóa)
   - [2.1. Chiều Cao Nhân Trắc Học Tham Chiếu ($H_{\text{ref}}$)](#21-chiều-cao-nhân-trắc-học-tham-chiếu-h_textref)
   - [2.2. Thuật Toán Gom Cụm Không Gian - Thời Gian (STIC v2)](#22-thuật-toán-gom-cụm-không-gian---thời-gian-stic-v2)
   - [2.3. Động Cơ Theo Dấu Thực Thể Tách Biệt (Decoupled Causal Tracking)](#23-động-cơ-theo-dấu-thực-thể-tách-biệt-decoupled-causal-tracking)
   - [2.4. Động Cơ Suy Luận Quan Hệ Thị Giác (RelateAnything Engine)](#24-động-cơ-suy-luận-quan-hệ-thị-giác-relateanything-engine)
3. [Cấu Trúc Thư Mục Dự Án](#3-cấu-trúc-thư-mục-dự-án)
4. [Cài Đặt Môi Trường & Yêu Cầu Kỹ Thuật](#4-cài-đặt-môi-trường--yêu-cầu-kỹ-thuật)
5. [Hướng Dẫn Vận Hành Dòng Lệnh (CLI Runner)](#5-hướng-dẫn-vận-hành-dòng-lệnh-cli-runner)
6. [Công Cụ Tái Tạo Video Ngoại Tuyến Từ JSON (Thuần CPU)](#6-công-cụ-tái-tạo-video-ngoại-tuyến-từ-json-thuần-cpu)
7. [Giao Diện Giám Sát Web Đa Tương Tác (Streamlit Dashboard)](#7-giao-diện-giám-sát-web-đa-tương-tác-streamlit-dashboard)
8. [Đóng Gói & Triển Khai Container (Docker Deployment)](#8-đóng-gói--triển-khai-container-docker-deployment)
9. [Cấu Trúc Đặc Tả Dữ Liệu Đầu Ra (JSON Schema)](#9-cấu-trúc-đặc-tả-dữ-liệu-đầu-ra-json-schema)
10. [Bảng Đánh Giá Thực Nghiệm Trên 7 Video Tiêu Biểu (Golden Benchmark)](#10-bảng-đánh-giá-thực-nghiệm-trên-7-video-tiêu-biểu-golden-benchmark)
11. [Báo Cáo Trung Thực Về Phạm Vi & Giới Hạn Hiện Tại](#11-báo-cáo-trung-thực-về-phạm-vi--giới-hạn-hiện-tại)

---

## 1. Tổng Quan Kiến Trúc Hệ Thống

Pipeline xử lý luồng video giám sát từ khung hình thô (raw frames) đến đầu ra dữ liệu quan hệ có cấu trúc và video giám sát trực quan thông qua 4 khối chức năng tuần tự:

```
[Camera Stream / Video CCTV Đầu Vào (MP4)]
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. ĐỘNG CƠ THEO DẤU THỰC THỂ TÁCH BIỆT (DECOUPLED TRACKING) │
│    - Detector YOLOE-26m quét 60 lớp đối tượng VidVRD        │
│    - Stream Người: Standalone ByteTrack + Re-ID OSNet-x0.25 │
│    - Stream Đồ vật: Hungarian Matching tối ưu toàn cục      │
│    - Đánh số định danh nhân quả đơn điệu: [1], [2], [3]...  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PHÂN CỤM TƯƠNG TÁC KHÔNG GIAN - THỜI GIAN (STIC v2)      │
│    - Chuẩn hóa nhân trắc học phi thứ nguyên theo H_ref      │
│    - Căn chỉnh mặt phẳng chân tiếp đất 3D (Ground Plane)    │
│    - Bộ lọc hướng di chuyển khách bộ hành (Passing-by Law)  │
│    - Khóa động học cứng cưỡi xe & khử linh kiện phụ tùng    │
│    - Khóa tọa độ trung vị đồ vật đặt xuống sàn (0.00 pixel) │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. SUY LUẬN QUAN HỆ THỊ GIÁC (RELATEANYTHING FOUNDATION)    │
│    - Cắt khung nhìn vuông Square Guard (1:1) chống méo hình │
│    - Mô hình DINOv3 ViT-S/16+ (53.2M params) & 26 vị từ     │
│    - Khóa ngữ nghĩa VidVRD (Chomsky Selectional Rules)      │
│    - Máy trạng thái FSM Quăng Ném & Tái chuẩn hóa đòn đánh  │
│    - Lọc trôi nhãn thời gian qua EdgeBook Hysteresis        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. KẾT XUẤT DỮ LIỆU ĐA PHƯƠNG THỨC & TRỰC QUAN HÓA         │
│    - File JSON cấu trúc đầy đủ quỹ đạo frame & quan hệ      │
│    - Video MP4 giám sát trực quan chuẩn H.264 (OpenH264)    │
│    - Web Dashboard Streamlit hỗ trợ Filter 3 chiều & Seek   │
│    - Bộ dựng hình ngoại tuyến siêu tốc bằng CPU (2-3s)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Kho Thuật Toán Cốt Lõi & Bảng Tham Số Chuẩn Hóa

Toàn bộ hệ thống được xây dựng trên triết lý **Vật lý học, Hình học xạ ảnh và Nhân trắc học**, loại bỏ $100\%$ việc hardcode tên video, số frame hay số ma thuật pixel (*Zero Magic Numbers*).

### 2.1. Chiều Cao Nhân Trắc Học Tham Chiếu ($H_{\text{ref}}$)
* **Ý nghĩa vật lý:** Trong camera giám sát góc chúc từ trên cao (monocular high-pitch CCTV), cự ly pixel bị co giãn phi tuyến theo trục sâu $Z$. Mọi khoảng cách hình học và ngưỡng vận tốc động học đều được chuẩn hóa phi thứ nguyên theo chiều cao trung vị của người tham gia tương tác:
  $$H_{\text{ref}} = \text{median}(\{h_i \mid h_i = y_{2, i} - y_{1, i} \ge 20\text{px}\})$$
* **Giá trị dự phòng khi khung hình không có người:** $H_{\text{ref}} = 0.25 \cdot H_{\text{img}}$.
* **Vị trí trong mã nguồn:** File `modules/spatial_clustering.py` (hàm `compute_anthropometric_reference_height`, dòng 57–90).

---

### 2.2. Thuật Toán Gom Cụm Không Gian - Thời Gian (STIC v2)

| Hạng Mục Tương Tác | Tham Số & Công Thức | Giá Trị Ngưỡng Thực Tế Trong Code | Ý Nghĩa Kỹ Thuật / Hình Học | Vị Trí Trong Mã Nguồn |
| :--- | :--- | :---: | :--- | :--- |
| **Người – Người** (Hình thành cụm) | $d_{\text{edge}} \le \text{prox\_thresh}$ | **$\le 0.25 \cdot H_{\text{ref}}$** (hoặc $\text{IoU} > 0.0$) | Không gian cá nhân thân mật (Personal Space); phải duy trì liên tục $\ge 0.80$s qua bộ lọc trễ debounce. | `modules/spatial_clustering.py` (dòng 163, 171) |
| **Người – Người** (Giải tán cụm) | $d_{\text{edge}} > \text{disband\_thresh}$ | **$> 0.45 \cdot H_{\text{ref}}$** | Bộ lọc trễ Schmitt Trigger: phải duy trì cách xa liên tục $\ge 0.50$s mới ngắt cụm, chống đứt cụm đột ngột. | `modules/spatial_clustering.py` (dòng 164, 174) |
| **Người – Người** (Căn chỉnh sàn 3D) | $\|y_{2, a} - y_{2, b}\| \le \Delta y_{\text{ground}}$ | **$\le 0.35 \cdot H_{\text{ref}}$** | Bắt buộc 2 người cùng tiếp xúc mặt sàn ở cùng độ sâu 3D; ngăn góc camera chúc làm đè người gần lên người xa. | `modules/spatial_clustering.py` (dòng 169, 190) |
| **Người – Người** (Hướng di chuyển) | $\cos(\theta) = \frac{\vec{v}_a \cdot \vec{v}_b}{\|\vec{v}_a\| \|\vec{v}_b\|}$ | **$\ge -0.20$** | **Passing-by Law:** Veto người đi ngược chiều lướt qua nhau (trừ khi có va chạm thân thể trực tiếp $d \le 0.05 H_{\text{ref}}$). | `modules/spatial_clustering.py` (dòng 207–209) |
| **Người – Vật di chuyển** (Tầm với tay) | $d_{\text{edge}} \le \text{reach\_thresh}$ | **$\le 0.15 \cdot H_{\text{ref}}$** (hoặc $\text{IoU} \ge 0.02$) | Tầm với tay vật lý đối với đồ vật có dịch chuyển tổng thể $disp \ge 0.20 \cdot H_{\text{ref}}$. | `modules/spatial_clustering.py` (dòng 250–253) |
| **Người – Vật di chuyển** (Độ cao tầm với) | $y_{1, o} \le y_{2, p} + 0.1 H \land y_{2, o} \ge y_{1, p} - 0.1 H$ | *Khoảng thân người* | Đồ vật mang theo phải nằm trong không gian chiều dọc cơ thể người. | `modules/spatial_clustering.py` (dòng 270) |
| **Người – Vật tĩnh trên sàn/bàn** | Căn chỉnh độ sâu chân tiếp đất | **$\le \max(0.40 h_o, 0.35 h_p)$** | Ngăn gộp vật thể tĩnh ở hậu cảnh vào người đi ngang qua tiền cảnh. | `modules/spatial_clustering.py` (dòng 297) |
| **Người – Vật tĩnh trên sàn/bàn** | Ràng buộc dừng đỗ (Kinematic Dwell) | **$v_{\text{person}} < 0.005 \cdot H_{\text{ref}}$** | Người phải đứng chậm lại cạnh vật $\ge 1.0$s mới coi là tương tác với đồ vật tĩnh (hoặc tiếp xúc đánh/đá $\ge 0.35$s). | `modules/spatial_clustering.py` (dòng 279–281, 338) |
| **Cưỡi xe máy/xe đạp** | Rigid Kinematic Coupling | $\Delta v \le 0.010 H_{\text{ref}} \land \text{Area}_{\text{inter}} \ge 0.35 \text{Area}_p$ | Người cưỡi xe phải di chuyển đồng tốc tuyệt đối và trùm lên thân xe. Người đi bộ ngang qua xe đỗ bị loại trừ. | `pipeline_relate_anything.py` (dòng 1485–1540) |
| **Khử phụ tùng xe gắn liền** | Sub-part Structural Subordination | $\text{Median}(\frac{\text{Area}(B_o \cap B_v)}{\text{Area}(B_o)}) \ge 0.50$ | Yên xe, giỏ xe bị nhận nhầm thành túi xách độc lập sẽ tự động gộp vào xe máy, loại bỏ đồ vật ma. | `pipeline_relate_anything.py` (dòng 3219–3225) |
| **Bất động vật đặt xuống bàn/sàn** | Spatial Median Clamping | **$0.00$ pixel dịch chuyển** | Khi vật dừng $v < 0.005 H_{\text{ref}}$ và tiếp sàn $\ge 5$ frame, khóa cứng tọa độ trung vị, xóa sạch rung giật. | `pipeline_relate_anything.py` (dòng 2008–2035) |

---

### 2.3. Động Cơ Theo Dấu Thực Thể Tách Biệt (Decoupled Causal Tracking)

| Thành Phần | Tham Số Cấu Hình | Giá Trị Thực Tế | Cơ Chế & Vai Trò | Vị Trí Trong Mã Nguồn |
| :--- | :--- | :---: | :--- | :--- |
| **YOLOE-26m** | Default Confidence (`--conf`) | **`0.20`** (CLI, Runner, Web) | Ngưỡng tin cậy lọc bounding box đầu vào đồng bộ toàn hệ thống. | `pipeline_relate_anything.py` (dòng 100), `run.py` (dòng 31), `app.py` (dòng 570) |
| **YOLOE-26m** | `OBJECT_TRACK_CONF` | **`0.15`** | Ngưỡng chạy `yolo_model.predict` và bể cứu hộ ($D_{\text{low}}$) đồ vật mờ/che khuất. | `pipeline_relate_anything.py` (dòng 80, 2362) |
| **YOLOE-26m** | `OBJECT_SPAWN_CONF` | **`0.40`** | Ngưỡng tin cậy tối thiểu để sinh mới một tracklet đồ vật (chống sinh đồ vật ma). | `pipeline_relate_anything.py` (dòng 79, 2555) |
| **ByteTrack (Người)** | `track_high` / `track_low` | **`0.25`** / **`0.10`** | Hai tầng liên kết IoU cho người theo chuẩn thuật toán ByteTrack. | `pipeline_relate_anything.py` (dòng 2340–2341) |
| **ByteTrack (Người)** | `new_track` / `match_thresh` | **`0.40`** / **`0.70`** | Ngưỡng khởi tạo track người mới và ngưỡng gán liên kết (buffer: 30 frames). | `pipeline_relate_anything.py` (dòng 2342–2344) |
| **Hungarian (Đồ vật)** | Ngưỡng IoU Bipartite Matching | **$\ge 0.15$** | Gán liên kết đồ vật qua ma trận chi phí Hungarian kết hợp Semantic Super-Class Families. | `pipeline_relate_anything.py` (dòng 2517) |
| **OSNet Re-ID** | Vector ngoại hình & Ngưỡng Cosine | 512-dim / **$\ge 0.76$** | Mạng `OSNet-x0.25` (MSMT17) khâu nối vết người bị gián đoạn che khuất. | `pipeline_relate_anything.py` (dòng 2600–2760) |
| **Extraction Quality Gate** | Tỷ lệ khung hình người Re-ID | **$h/w \ge 1.70$** | Chỉ lấy mẫu vector Re-ID khi người hiển thị đầy đủ, không bị mép cửa cắt cụt. | `PROJECT_MASTER_HANDOVER.md` (Mục 4.7) |
| **Biomechanical Dwell Gate** | Ngưỡng thời gian quay đầu tại cửa | **$\Delta t < 0.50$s** | Một người không thể bước qua cửa, hãm quán tính, quay 180° trong $<0.5$s. Nếu đảo chiều $\cos\theta < 0$, cấm ghép vết để tách người áo đen và áo trắng. | `PROJECT_MASTER_HANDOVER.md` (Mục 4.7) |
| **Đánh Số Causal Agnostic ID** | Định danh nhân quả vô ngã | **`[1]`, `[2]`, `[3]`,...** | Đánh số tịnh tiến tăng đơn điệu theo thời điểm xuất hiện thực tế ($t_{\text{birth}}$), độc lập giữa ID và nhãn phân loại class. | `pipeline_relate_anything.py` (dòng 3253–3258) |

---

### 2.4. Động Cơ Suy Luận Quan Hệ Thị Giác (RelateAnything Engine)

| Tham Số | Giá Trị | Cơ Chế & Ý Nghĩa | Vị Trí Trong Mã Nguồn |
| :--- | :---: | :--- | :--- |
| **Foundation Model** | `maelic/relsgg-vits16plus` | 53.2M tham số, visual backbone DINOv3 ViT-S/16+, từ vựng 26 vị từ VidVRD đóng. | `pipeline_relate_anything.py` (dòng 96) |
| **Square Guard (1:1)** | Cắt vuông đẳng hướng tâm | Cắt khung nhìn vuông đưa vào DINOv3 ($448 \times 448$px), triệt tiêu $100\%$ biến dạng méo hình học. | `pipeline_relate_anything.py` (dòng 107) |
| **Ngưỡng Vị Từ Quan Hệ** | `min_rel_conf = 0.15` | Ngưỡng tin cậy tối thiểu để chấp nhận một vị từ quan hệ. | `pipeline_relate_anything.py` (dòng 105) |
| **Tính Nhất Quán Thời Gian** | `min_consistency = 0.25` | Quan hệ phải xuất hiện trong $\ge 25\%$ số frame được đánh giá của cụm tương tác. | `pipeline_relate_anything.py` (dòng 104) |
| **EdgeBook Hysteresis (Kéo dài)** | $on=0.45, off=0.30, \alpha=0.35$ | Áp dụng cho các tư thế kéo dài (`hold`, `carry`, `ride`) để loại bỏ hoàn toàn hiện tượng rung giật nhãn. | `pipeline_relate_anything.py` (dòng 1845–1855) |
| **EdgeBook Hysteresis (Chớp nhoáng)** | $on \approx 0.20-0.35, \alpha=0.85$ | Áp dụng cho hành vi xung lực ngắn (`hit`, `kick`, `throw`, `get_on`, `get_off`) để bắt trọn đòn đánh nhanh. | `pipeline_relate_anything.py` (dòng 1846–1850) |
| **FSM Quăng Ném (Throw Transition)** | Chuỗi 3 pha động học | Phát hiện chuyển tiếp: Cầm giữ ($\text{hold}/\text{carry}$) $\to$ Gia tốc tách rời $\to$ Phân kỳ vận tốc chuyển động tự do. | `pipeline_relate_anything.py` (dòng 2100–2150) |
| **Selectional Restrictions** | Khóa ngữ nghĩa Chomsky 1965 | Khóa cứng $0.0$ cho các vị từ sinh học (`feed`, `kiss`, `bite`, `lick`) đối với đồ vật vô tri (`handbag`, `backpack`...). | `pipeline_relate_anything.py` (dòng 1650–1680) |

---

## 3. Cấu Trúc Thư Mục Dự Án

```
RelateAnything/
├── run.py                          # Điểm chạy đơn giản duy nhất toàn bộ pipeline (A-Z)
├── render_from_json.py             # Công cụ tái tạo video MP4 siêu tốc từ JSON (thuần CPU 2-3s)
├── app.py                          # Giao diện Web Dashboard giám sát CCTV (Streamlit)
├── pipeline_relate_anything.py     # Động cơ pipeline cốt lõi (Tracking, STIC v2, RelateAnything)
├── modules/
│   ├── spatial_clustering.py       # Thuật toán STIC v2, căn chỉnh 3D Ground Plane, Passing-by Law
│   └── appearance_reid.py          # Trích xuất đặc trưng ngoại hình Person Re-ID
├── models/
│   ├── osnet_x0_25.py              # Mã nguồn kiến trúc mạng nơ-ron Re-ID OSNet
│   ├── osnet_x0_25_msmt17.pth      # [Local Weight] Trọng số Re-ID người (MSMT17)
│   └── yoloe-26m-seg.pt            # [Local Weight] Trọng số bộ nhận diện YOLOE-26m
├── configs/
│   ├── s_objects.json              # 60 lớp đối tượng VidVRD chuẩn hóa
│   ├── relations.json              # 26 vị từ quan hệ VidVRD đóng
│   └── custom_bytetrack.yaml       # Cấu hình siêu tham số ByteTrack
├── relsgg/                         # Thư viện mô hình RelateAnything (DINOv3 ViT-S/16+)
├── Dockerfile                      # Cấu hình đóng gói container Docker (NVIDIA CUDA 12.4 + PyTorch)
├── docker-compose.yml              # Cấu hình khởi chạy 1 lệnh với Docker Compose
├── requirements.txt                # Danh mục thư viện Python chính thức
├── DOCKER_GUIDE.md                 # Hướng dẫn chi tiết triển khai Docker
├── data/
│   ├── videos/                     # Kho video giám sát thực tế đầu vào (video1.mp4 ... video18.mp4)
│   └── output/
│       └── full_video/             # Thư mục kết quả chính thức: các file JSON và video MP4
└── README.md                       # Tài liệu hướng dẫn kỹ thuật toàn diện
```

---

## 4. Cài Đặt Môi Trường & Yêu Cầu Kỹ Thuật

### Cấu Hình Kiểm Thử Thực Tế
- **Hệ điều hành:** Windows 10/11 hoặc Linux Ubuntu 22.04 LTS.
- **GPU:** NVIDIA GeForce GTX 1650 Ti (4GB VRAM) hoặc cao hơn (hỗ trợ CUDA 12.4).
- **Mức tiêu thụ tài nguyên thực tế:** Pipeline xử lý cục bộ theo từng cụm, tiêu thụ **$2.5\text{GB} - 3.2\text{GB}$ VRAM**, vận hành mượt mà trên card 4GB VRAM.
- **Hỗ trợ CPU:** Tự động chuyển đổi tính toán trên CPU nếu không có card đồ họa rời qua tham số `--device cpu`.
- **Python:** 3.10 – 3.12 trong môi trường ảo `.venv`.

### Các Bước Cài Đặt
```powershell
# 1. Clone repository
git clone https://github.com/higo-ai/RelateAnything.git
cd RelateAnything

# 2. Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# 3. Cài đặt các gói phụ thuộc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e . --no-deps
```

---

## 5. Hướng Dẫn Vận Hành Dòng Lệnh (CLI Runner)

Hệ thống cung cấp điểm chạy dòng lệnh thống nhất `run.py` hoặc chạy trực tiếp qua `pipeline_relate_anything.py`:

```powershell
# Chạy video1 trên GPU CUDA:
.venv\Scripts\python.exe run.py --video data/videos/video1.mp4 --device cuda

# Hoặc gọi trực tiếp pipeline lõi:
.venv\Scripts\python.exe pipeline_relate_anything.py --video data/videos/video1.mp4 --device cuda --full_video
```

Kết quả tự động xuất ra tại thư mục: `data/output/full_video/<tên_video>/`:
- `<tên_video>_relations.json`: File JSON quan hệ và mảng tọa độ theo frame.
- `<tên_video>_visualizer.mp4`: Video giám sát trực quan chuẩn H.264.

---

## 6. Công Cụ Tái Tạo Video Ngoại Tuyến Từ JSON (Thuần CPU)

Nhờ kiến trúc tách rời dữ liệu (*Decoupled Architecture*), sau khi có file JSON kết quả, có thể tái tạo lại video MP4 chỉ trong **2–3 giây** thuần túy bằng CPU và OpenCV mà không cần nạp lại mô hình AI nặng:

```powershell
.venv\Scripts\python.exe render_from_json.py `
    --video data/videos/video1.mp4 `
    --json data/output/full_video/video1/video1_relations.json `
    --output data/output/full_video/video1/video1_from_json.mp4
```

---

## 7. Giao Diện Giám Sát Web Đa Tương Tác (Streamlit Dashboard)

Giao diện trực quan được thiết kế theo phong cách phòng điều hành giám sát CCTV hiện đại (**Operations Room Dark UI**) vừa khít 1 màn hình máy tính (Zero-Scroll):

```powershell
.venv\Scripts\streamlit.exe run app.py
```
*Truy cập tại:* **`http://localhost:8501`**

### Tính Năng Nổi Bật:
1. **Bộ lọc quan hệ 3 chiều (3-Way Cascade Filter):**
   - Lọc theo **Chủ thể (Subject)**, **Hành vi (Action/Predicate)**, và **Đối tượng (Object)** theo thời gian thực.
2. **Tua video tức thời (Click-to-Seek):**
   - Bấm nút **`▶ Tua {Xs}`** trên từng thẻ quan hệ để trình phát video nhảy ngay lập tức tới đúng giây diễn ra hành vi.
3. **Menu thiết lập linh hoạt (Popover Settings):**
   - Điều chỉnh trực tiếp ngưỡng YOLO Conf, Relation Conf, Consistency Thresh và chọn thiết bị tính toán (Auto / CUDA / CPU).
4. **Tải file JSON trực tiếp:**
   - Bấm nút **`Tải file json 📥`** để tải tệp JSON phân tích tức thì.

---

## 8. Đóng Gói & Triển Khai Container (Docker Deployment)

Dự án đã được container hóa hoàn chỉnh qua Docker và Docker Compose:

```bash
# Khởi chạy toàn bộ hệ thống bằng 1 dòng lệnh:
docker compose up -d --build
```
*Sau khi container khởi động, mở trình duyệt truy cập: `http://localhost:8501`.*  
*Chi tiết hướng dẫn giải thích tham số xem tại: [DOCKER_GUIDE.md](file:///c:/AIThucChien/RelateAnything/DOCKER_GUIDE.md).*

---

## 9. Cấu Trúc Đặc Tả Dữ Liệu Đầu Ra (JSON Schema)

```json
{
  "video_metadata": {
    "video_name": "video1",
    "fps": 25.0,
    "total_frames": 2191,
    "resolution": [1920, 1080],
    "duration_sec": 73.0,
    "mode": "roi_zoom"
  },
  "tracks": {
    "[5]": {
      "track_id": "[5]",
      "type": "person",
      "class": "person",
      "trajectories": {
        "1117": [450, 210, 520, 480]
      }
    }
  },
  "relations": [
    {
      "subject_id": "[5]",
      "subject_class": "person",
      "predicate": "carry",
      "object_id": "[7]",
      "object_class": "handbag",
      "confidence": 0.83,
      "temporal_consistency": 0.57,
      "time_range": {
        "start_sec": 37.4,
        "end_sec": 47.1
      },
      "active_frames": [1117, 1118, 1119]
    }
  ],
  "clusters": [ ... ],
  "singletons": [ "[1]", "[2]", "[3]", "[4]", "[8]", "[9]", "[10]", "[11]" ]
}
```

---

## 10. Bảng Đánh Giá Thực Nghiệm Trên 7 Video Tiêu Biểu (Golden Benchmark)

Dưới đây là kết quả kiểm định thực nghiệm chính thức trên 7 video tiêu biểu đại diện cho các kịch bản giám sát an ninh then chốt (kết xuất trực tiếp từ các file JSON trong `data/output/full_video/`):

| Video | Thời Lượng | Thực Thể Phát Hiện | Phân Cụm STIC v2 & Singletons | Các Quan Hệ Được Xác Nhận (Confirmed Relations) | Trạng Thái Nghiệm Thu |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **`video1`** | **73.0s**<br>(2.191f) | 11 thực thể:<br>10 persons, 1 handbag | **1 Cụm Tương Tác:**<br>`cluster_1`: `[5]`, `[6]`, `[7]`<br><br>*8 Singletons cô lập:* `[1..4]`, `[8..11]` | • `[5] person -carry-> [7] handbag` (conf: 0.83, consist: 57%)<br>• `[6] person -hold-> [5] person` (conf: 0.57, consist: 40%)<br>• `[5] person -touch-> [6] person` (conf: 0.53, consist: 34%)<br>• `[6] person -touch-> [5] person` (conf: 0.54, consist: 28%) | Tách đúng người lạ tại cửa; không gom nhầm bình cứu hỏa trên tường; nhận diện cử chỉ khoác vai. |
| **`video2`** | **94.0s**<br>(1.589f) | 19 thực thể:<br>14 persons, 5 handbags | **4 Cụm Độc Lập:**<br>`cluster_1`: 10 entities<br>`cluster_2`: `[13]`, `[14]`<br>`cluster_3`: `[16]`, `[17]`<br>`cluster_4`: `[18]`, `[19]`<br><br>*3 Singletons:* `[1]`, `[2]`, `[4]` | • `[9] person -carry-> [10] handbag` (conf: 0.80, consist: 100%)<br>• `[3] person -carry-> [5] handbag` (conf: 0.75, consist: 100%)<br>• `[7] person -hold-> [3] person` (conf: 0.51, consist: 79%)<br>• `[11] person -hold-> [8] person` (conf: 0.61, consist: 48%)<br>• `[14] person -hold-> [13] person` (conf: 0.91, consist: 97%)<br>• `[13] person -hold-> [14] person` (conf: 0.90, consist: 97%)<br>• `[16] person -carry-> [17] handbag` (conf: 0.95, consist: 70%)<br>• `[16] person -hold-> [17] handbag` (conf: 0.95, consist: 70%)<br>• `[18] person -carry-> [19] handbag` (conf: 0.77, consist: 98%) | Phân tách 4 cụm độc lập hoàn hảo trong đám đông ga tàu; mỗi người chỉ tương tác với hành lý của mình. |
| **`video3`** | **50.0s**<br>(1.499f) | 4 thực thể:<br>2 persons, 2 handbags | **2 Cụm Độc Lập:**<br>`cluster_1`: `[1] person`, `[2] handbag`<br>`cluster_2`: `[3] handbag`, `[4] person`<br><br>*0 Singletons* | • `[1] person -carry-> [2] handbag` (conf: 0.95, consist: 97%)<br>• `[1] person -hold-> [2] handbag` (conf: 0.96, consist: 91%)<br>• `[4] person -carry-> [3] handbag` (conf: 0.95, consist: 28%)<br>• `[4] person -hold-> [3] handbag` (conf: 0.94, consist: 27%) | Bắt trọn vẹn hành vi xách đồ của từng người độc lập; không có hiện tượng gộp dính chéo. |
| **`video7`** | **53.2s**<br>(1.596f) | 4 thực thể:<br>1 person, 2 handbags, 1 backpack | **1 Cụm Tương Tác:**<br>`cluster_1`: `[2] person`, `[3] handbag`<br><br>*2 Singletons cô lập:*<br>`[1]` (túi tĩnh), `[4]` (balo tĩnh) | • `[2] person -carry-> [3] handbag` (conf: 0.95, consist: 69%)<br>• `[2] person -hold-> [3] handbag` (conf: 0.96, consist: 66%) | Đặt túi xách lên bàn: túi trên bàn đạt độ bất động 0.00px; ngắt nhãn `hold` đúng thời điểm buông tay. |
| **`video10`** | **10.0s**<br>(240f) | 5 thực thể:<br>1 person, 3 motorcycles, 1 bicycle | **1 Cụm Tương Tác:**<br>`cluster_1`: `[3] bicycle`, `[4] motorcycle`, `[5] person`<br><br>*2 Singletons:* `[1]`, `[2]` | • `[5] person -ride-> [4] motorcycle` (conf: 0.96, consist: 100%)<br>• `[5] person -lean_on-> [4] motorcycle` (conf: 0.54, consist: 100%)<br>• `[5] person -get_on-> [3] bicycle` (conf: 0.41, consist: 8%) | Nhận diện chính xác tư thế điều khiển phương tiện `ride (0.96)`; các xe máy đỗ tĩnh được cô lập. |
| **`video11`** | **10.0s**<br>(240f) | 7 thực thể:<br>2 persons, 5 objects | **1 Cụm Tương Tác:**<br>`cluster_1`: `[5] person`, `[6] person`<br><br>*5 Singletons:* `[1..4]`, `[7]` | • `[5] person -hit-> [6] person` (conf: 0.42, consist: 13%) | Bắt chuẩn cú đấm xung lực trong phân cảnh đối kháng bạo lực; các xe máy xung quanh được bảo vệ cô lập. |
| **`video16`** | **10.0s**<br>(240f) | 7 thực thể:<br>1 person, 6 objects | **1 Cụm Tương Tác:**<br>`cluster_1`: `[5] person`, `[6] handbag`<br><br>*5 Singletons:* `[1..4]`, `[7]` | • `[5] person -carry-> [6] handbag` (conf: 0.97, consist: 95%)<br>• `[5] person -hold-> [6] handbag` (conf: 0.97, consist: 95%)<br>• `[5] person -throw-> [6] handbag` (conf: 0.88, consist: 15%) | Máy trạng thái động học FSM bắt trọn cả 3 pha: mang xách, cầm giữ và quăng ném túi xách (`throw 0.88`). |

---

## 11. Báo Cáo Trung Thực Về Phạm Vi & Giới Hạn Hiện Tại

Trên tinh thần trung thực khoa học, hệ thống hiện tại được tối ưu hóa xuất sắc cho các kịch bản giám sát an ninh cốt lõi (mang xách hành lý, di chuyển cùng nhau, lên xuống và điều khiển phương tiện, đặt đồ vật lên bàn/sàn, quăng ném và đối kháng cơ thể). Tuy nhiên, hệ thống vẫn tồn tại một số giới hạn thực tế cần tiếp tục hoàn thiện:

1. **Camera hồng ngoại ban đêm (Thermal IR):** Các video mất hoàn toàn thông tin màu sắc và có độ tương phản cực thấp (như `video5`) khiến bộ nhận diện YOLOE dễ bị giảm độ tự tin, dẫn đến đứt đoạn chuỗi theo vết.
2. **Góc quay bị che khuất sâu và mật độ đám đông quá dày đặc:** Khi nhiều đối tượng đi đè sát lên nhau theo trục sâu quang học trong thời gian dài ($>3$s), việc phân tách ranh giới tiếp xúc cơ thể vẫn gặp thách thức.
3. **Độ nhạy đối với vật thể quá nhỏ:** Các đồ vật kích thước nhỏ (ví, điện thoại cầm tay) ở khoảng cách camera xa ($>15$m) cần được bổ sung thêm dữ liệu huấn luyện tăng cường độ phân giải.

---
*Dự án hoàn thành phục vụ công tác nghiên cứu và thực tập kỹ thuật tại VinFast & VinUni — Tháng 10/2026.*
