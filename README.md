# VidVRD RelateAnything: Hệ Thống Nhận Diện Quan Hệ Thị Giác Trong Video Giám Sát

Hệ thống cung cấp một luồng xử lý toàn diện (End-to-End Pipeline) cho bài toán **Nhận diện Quan hệ Thị giác trong Video (Video Visual Relationship Detection - VidVRD)** trên các bối cảnh camera giám sát an ninh (CCTV). Hệ thống tích hợp theo dõi tách biệt người và vật thể (Decoupled Tracking), thuật toán gom cụm tương tác không gian - thời gian (**STIC v2**), và mô hình nền tảng suy luận quan hệ từ vựng đóng **RelateAnything** (`maelic/relsgg-vits16plus`).

---

## 1. Kiến Trúc Hệ Thống

Pipeline xử lý video giám sát từ khung hình thô (raw frames) đến đầu ra dữ liệu quan hệ có cấu trúc và video trực quan qua 4 giai đoạn tuần tự:

```
[Video Giám Sát Đầu Vào (MP4)]
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Động Cơ Theo Dấu Thực Thể Tách Biệt (Causal Tracking) │
│    - Người: ByteTrack độc lập + Trích xuất Re-ID OSNet   │
│    - Đồ vật: Thuật toán Hungarian dựa trên IoU không gian│
│    - Đánh số định danh nhân quả đơn điệu ([1], [2], ...)│
└──────────────────────────────┬──────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Gom Cụm Tương Tác Không Gian - Thời Gian (STIC v2)   │
│    - Khoảng cách nhân trắc học (Edward T. Hall)         │
│    - Căn chỉnh mặt phẳng chân tiếp đất 3D (Ground Plane)│
│    - Bộ lọc hướng di chuyển của người đi bộ (Passing-by)│
│    - Khóa động học cứng và khử phụ tùng xe gắn liền     │
└──────────────────────────────┬──────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Suy Luận Quan Hệ Thị Giác (RelateAnything)           │
│    - Mô hình nền tảng RelateAnything (DINOv3 ViT-S/16+) │
│    - Bộ từ vựng đóng VidVRD: 60 đối tượng & 26 vị từ    │
│    - Cắt khung nhìn động (Dynamic ROI Zoom)             │
│    - Máy trạng thái thời gian và bộ lọc FSM Quăng Ném   │
└──────────────────────────────┬──────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Xuất Dữ Liệu Đa Phương Thức Độc Lập                  │
│    - File JSON đầy đủ quỹ đạo từng frame và quan hệ     │
│    - Video MP4 giám sát trực quan mượt mà               │
│    - Bộ vẽ ngoại tuyến thuần OpenCV (Không cần GPU/AI)  │
└─────────────────────────────────────────────────────────┘
```

### Các Thành Phần Kỹ Thuật Chính

- **Theo dõi tách biệt người và vật:** Tách riêng luồng theo vết người (ByteTrack) và đồ vật (Hungarian) nhằm triệt tiêu hiện tượng tráo đổi định danh (ID swap) giữa người và xe/túi. Nhận diện người được củng cố khi bị che khuất qua vector ngoại hình trích xuất từ mạng nơ-ron OSNet-x0.25 (huấn luyện trên tập dữ liệu giám sát lớn MSMT17).
- **Gom cụm tương tác không gian - thời gian (STIC):** Thay vì kiểm tra tích chéo tốn kém ($O(N^2)$), STIC gom các thực thể có khả năng tương tác dựa trên tỷ lệ chiều cao người tham chiếu ($H_{\text{ref}}$) và điều kiện tiếp đất cùng độ sâu ($|y_{2, a} - y_{2, b}| \le 0.35 \cdot H_{\text{ref}}$).
- **Cơ chế lọc động học & Máy trạng thái FSM:** Phân định rõ ràng giữa các hành vi chuyển tiếp chớp nhoáng (`get_on` - lên xe, `get_off` - xuống xe, `throw` - quăng ném) và các tư thế kéo dài liên tục (`ride` - lái xe, `hold` - cầm giữ, `carry` - mang vác) dựa trên vận tốc phân kỳ và thời gian tiếp xúc thực tế.
- **Cấu trúc dữ liệu tách biệt hoàn toàn:** Toàn bộ kết quả dự đoán và tọa độ từng frame được tuần tự hóa vào file JSON chuẩn. Nhờ đó, người dùng có thể dùng công cụ vẽ ngoại tuyến để tái tạo lại video kết quả trong 2–3 giây bằng CPU thông thường mà không cần tải lại mô hình AI hay GPU.

---

## 2. Cài Đặt Môi Trường & Yêu Cầu Kỹ Thuật

### Phần Cứng & Hệ Điều Hành
- **Hệ điều hành:** Windows 10/11 hoặc Linux (Ubuntu 20.04/22.04).
- **GPU:** Khuyến nghị card đồ họa NVIDIA có hỗ trợ CUDA 11.8+ (đã kiểm thử trên dòng card RTX với bộ nhớ VRAM từ 8GB trở lên). Hỗ trợ chạy trên CPU thông qua tham số `--device cpu`.
- **Python:** Phiên bản Python 3.10 đến 3.12 (khuyên dùng môi trường ảo `.venv`).

### Các Bước Cài Đặt

1. **Clone repository về máy:**
   ```bash
   git clone https://github.com/higo-ai/RelateAnything.git
   cd RelateAnything
   ```

2. **Kích hoạt môi trường ảo:**
   ```powershell
   # Trên Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # Trên Linux/macOS:
   source .venv/bin/activate
   ```

3. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install ultralytics opencv-python Pillow scipy streamlit pandas pyarrow
   pip install -e ".[hub]"
   ```

4. **Kiểm tra trọng số mô hình:**
   Đảm bảo các file trọng số được đặt đúng trong thư mục `models/`:
   - `models/yoloe-26m-seg.pt` (Trọng số mô hình nhận diện YOLOE)
   - `models/osnet_x0_25_msmt17.pth` (Trọng số mạng Re-ID trích xuất đặc trưng người)

---

## 3. Hướng Dẫn Chạy Dòng Lệnh (CLI)

Hệ thống cung cấp một điểm chạy duy nhất `run.py` để xử lý video từ đầu đến cuối (A-Z).

### Lệnh Chạy Toàn Bộ Pipeline
```powershell
.venv\Scripts\python.exe run.py --video data/videos/video2.mp4 --device cuda
```

### Các Tham Số Dòng Lệnh
| Tham Số | Kiểu Dữ Liệu | Mặc Định | Ý Nghĩa Chi Tiết |
| :--- | :---: | :---: | :--- |
| `--video` | `str` | *Bắt buộc* | Đường dẫn đến file video MP4 cần phân tích (ví dụ: `data/videos/video1.mp4`). |
| `--mode` | `str` | `roi` | Chế độ chạy: `roi` (Gom cụm và zoom động STIC) hoặc `full` (Khung nhìn toàn cảnh). |
| `--conf` | `float` | `0.35` | Ngưỡng tin cậy của bộ nhận diện đối tượng YOLO. |
| `--device` | `str` | `auto` | Thiết bị tính toán: `auto`, `cuda` (GPU), hoặc `cpu`. |
| `--output_subdir` | `str` | `None` | Thư mục con tùy chọn trong `data/output/`. |

Kết quả sau khi chạy được tự động lưu tại thư mục: `data/output/full_video/<tên_video>/`:
- `<tên_video>_relations.json`: File JSON chứa thông tin các track, tọa độ từng frame và toàn bộ các quan hệ phát hiện được.
- `<tên_video>_visualizer.mp4`: Video giám sát trực quan hoàn chỉnh với hộp nhận diện, dây nối tương tác và bảng thông tin hành vi thời gian thực.

---

## 4. Công Cụ Vẽ Video Ngoại Tuyến Từ JSON (Không Cần GPU)

Điểm cải tiến nổi bật của hệ thống là sự tách bạch giữa bước mô hình suy luận và bước hiển thị trực quan. Sau khi file JSON đã được xuất ra, bạn có thể tái tạo lại video MP4 chỉ trong khoảng 2–14 giây trên CPU thuần túy bằng thư viện OpenCV:

```powershell
.venv\Scripts\python.exe render_from_json.py `
    --video data/videos/video2.mp4 `
    --json data/output/full_video/video2/video2_relations.json `
    --output data/output/full_video/video2/video2_tai_tao.mp4
```

---

## 5. Hướng Dẫn Chạy Giao Diện Web Cục Bộ (Streamlit)

Để trình diễn demo cho Mentor hoặc người dùng thao tác bấm chuột trực quan trên máy cá nhân (Localhost):

```powershell
.venv\Scripts\streamlit.exe run app.py
```

*Sau khi gõ lệnh trên, trình duyệt web sẽ tự động mở trang:* **`http://localhost:8501`**

### Các Tính Năng Trên Giao Diện Web
- **Chọn nguồn video:** Chọn nhanh trong danh sách 15 video benchmark có sẵn (`video1` đến `video18`) hoặc tải lên file video MP4 từ máy tính.
- **Trình phát video song song:** Xem đồng thời video giám sát gốc và video kết quả của hệ thống.
- **Bảng dữ liệu quan hệ tương tác:** Xem danh sách quan hệ phát hiện được kèm mức độ tin cậy, thời gian bắt đầu - kết thúc, và số frame diễn ra.
- **Quỹ đạo thực thể:** Tra cứu chi tiết từng cá thể người/vật thể và số frame xuất hiện.
- **Tải file JSON:** Xem trực tiếp cây cấu trúc JSON trên web và bấm nút tải file JSON về máy.
- **Vẽ lại video ngoại tuyến nhanh:** Bấm nút để vẽ lại video từ file JSON chỉ trong vài giây mà không cần gọi lại mô hình AI.

---

## 6. Cấu Trúc Đặc Tả File Đầu Ra JSON (Schema Spec)

File JSON kết quả được thiết kế theo cấu trúc chuẩn, cung cấp đầy đủ dữ liệu cho các ứng dụng downstream:

```json
{
  "video_metadata": {
    "video_name": "video2",
    "fps": 25.0,
    "total_frames": 2818,
    "resolution": [1920, 1080],
    "duration_sec": 112.72,
    "mode": "roi_zoom",
    "description": "VidVRD Spatio-Temporal Interaction Clustering (STIC) with Causal Decoupled Tracking"
  },
  "tracks": {
    "[7]": {
      "track_id": "[7]",
      "type": "person",
      "class": "person",
      "color": [255, 128, 0],
      "total_frames_tracked": 321,
      "trajectories": {
        "1000": [450, 210, 520, 480],
        "1001": [452, 211, 521, 481]
      }
    }
  },
  "relations": [
    {
      "subject_id": "[7]",
      "subject_class": "person",
      "predicate": "carry",
      "object_id": "[9]",
      "object_class": "handbag",
      "confidence": 0.9812,
      "temporal_consistency": 1.0,
      "time_range": {
        "start_sec": 34.6,
        "end_sec": 40.3
      },
      "active_frames": [1000, 1001, 1002]
    }
  ],
  "clusters": [ ... ],
  "singletons": [ "[1]", "[2]", "[3]", "[5]" ]
}
```

### Ý Nghĩa Các Trường Dữ Liệu
- `video_metadata`: Thông số cơ bản của video (tên video, FPS, tổng số frame, độ phân giải và thời lượng).
- `tracks`: Danh bạ toàn bộ các thực thể được theo dõi trong video. Mỗi thực thể lưu mã ID (`[id]`), nhãn phân loại (`class`), loại thực thể (`person` hoặc `object`), màu hiển thị và danh sách tọa độ bounding box theo từng frame (`trajectories: {"frame_idx": [x1, y1, x2, y2]}`).
- `relations`: Danh sách các quan hệ tương tác hợp lệ được hệ thống xác nhận. Mỗi quan hệ ghi rõ chủ thể (`subject_id`), vị từ hành động (`predicate`), đối tượng tiếp nhận (`object_id`), điểm tin cậy trung bình, khoảng thời gian thực tế diễn ra (`time_range`), và danh sách các frame chính xác xuất hiện hành động (`active_frames`).
- `clusters` & `singletons`: Dữ liệu phân nhóm không gian và các thực thể đứng độc lập không tham gia tương tác.

---

## 7. Bảng Tổng Hợp Đánh Giá Thực Nghiệm Trên 15 Video Benchmark

Hệ thống đã được kiểm thử thực tế trên 15 đoạn video giám sát đa dạng ngữ cảnh (trong nhà, hành lang, sân bãi ngoài trời, camera hồng ngoại ban đêm):

| Video | Bối Cảnh Thực Tế & Thách Thức Kỹ Thuật | Số Cụm | Quan Hệ Thực Tế Được Phát Hiện & Xác Nhận | Đánh Giá Kết Quả |
| :--- | :--- | :---: | :--- | :--- |
| **`video1`** | Hành lang tòa nhà (73s). Người đi qua cửa ở 00:24s; hai người trò chuyện; có balo và bình cứu hỏa. | 1 | `[6] carry [8]` (0.96)<br>`[6] hold [8]` (0.87)<br>`[7] hold [6]` (0.53) | Tách đúng người lạ tại cửa; không gom nhầm bình cứu hỏa trên tường. |
| **`video2`** | Lối đi bộ đông đúc (94s). 22 thực thể đồng thời xuất hiện, nhiều người mang túi đi ngang qua nhau. | 6 | `[7] carry [9]` (0.98)<br>`[11] carry [12]` (0.96)<br>`[17] hold [16]` (0.91)<br>`[19] carry [20]` (0.95)<br>`[21] carry [22]` (0.95) | Giữ vững 6 cụm độc lập; căn chỉnh mặt phẳng nền 3D ngăn chặn hiện tượng gộp dính người ở các độ sâu khác nhau; người đi bộ lề phải đứng độc lập. |
| **`video3`** | Sảnh ga tàu (87s). Người đẩy xe em bé, dắt bé, kéo vali bánh lăn. | 3 | `[2] carry [3]` (0.97)<br>`[4] carry [5]` (0.95)<br>`[7] carry [6]` (0.89)<br>`[7] pull [8]` (0.71) | Bắt chính xác hành vi kéo vali lăn (`pull`) và đẩy xe em bé. |
| **`video4`** | Khách bộ hành mang xách đồ (39s). | 1 | `[4] carry [5]` (0.96)<br>`[4] hold [5]` (0.96) | Theo dõi ổn định, không phát sinh quan hệ rác. |
| **`video5`** | Camera hồng ngoại ban đêm (Thermal IR, 110s). Mất màu sắc, độ tương phản thấp. | 6 | `[10] carry [16]` (0.88)<br>`[14] ride [8]` (0.60)<br>`[14] get_off [17]` (0.59) | Vượt qua điều kiện ảnh nhiệt ban đêm; bắt chuẩn hành vi lên xe và điều khiển xe. |
| **`video7`** | Phòng học (160s). Đặt túi xách lên bàn rồi bước ra ngoài. | 1 | `[2] hold [3]` (0.95)<br>`[2] carry [3]` (0.95) | Chiếc túi trên bàn đạt độ dịch chuyển cố định 0.00px; ngắt nhãn `hold` đúng thời điểm buông tay. |
| **`video10`** | Bãi đỗ xe máy (84s). Người dắt xe, leo lên xe chạy, bước xuống xe; có người đi bộ ngang qua. | 2 | `[5] hold [4]` (0.97)<br>`[5] ride [4]` (0.86)<br>`[5] get_off [4]` (0.88)<br>`[7] ride [4]` (0.97) | Phân định rạch ròi người cưỡi xe và người đi bộ lướt qua; không bị nhầm đòn đánh lên xe. |
| **`video11`** | Phân cảnh ẩu đả trong sân (54s). Hai người đấm đá nhau giữa các xe máy đang đỗ. | 1 | `[6] hit [5]` (0.80)<br>`[5] hit [6]` (0.85)<br>`[5] kick [6]` (0.84)<br>`[6] kick [5]` (0.81) | Bắt trọn vẹn cả đòn đấm (`hit`) và đòn đá (`kick`); cô lập hoàn toàn các xe máy xung quanh. |
| **`video12`** | Người đi bộ đơn lẻ qua bãi xe (50s). | 0 | *(Không có quan hệ)* | Đúng thực tế; người đi một mình không bị gom nhầm với xe đỗ tĩnh. |
| **`video13`** | Hai đối tượng ẩu đả, giằng co trong bãi đỗ xe (52s). | 1 | `[5] hit [6]` (0.68)<br>`[6] hit [5]` (0.71)<br>`[5] kick [6]` (0.70) | Nhận diện chính xác xung lực va chạm đối kháng; các xe xung quanh giữ nguyên cô lập. |
| **`video14`** | Người đi bộ đơn lẻ trong sân (48s). | 0 | *(Không có quan hệ)* | Xác định chính xác không có tương tác nào diễn ra. |
| **`video15`** | Cầm túi và ném túi (50s). Chuyển động nhanh gây nhòe mờ (motion blur). | 1 | `[4] hold [5]` (0.98)<br>`[4] carry [5]` (0.95) | Giữ vững vết theo dõi qua các frame nhòe mờ mà không bị đứt đoạn. |
| **`video16`** | Người quăng ném túi xách (51s). | 1 | `[5] hold [6]` (0.94)<br>`[5] carry [6]` (0.93)<br>`[5] throw [6]` (0.88) | Máy trạng thái động học bắt trúng pha tách ly tốc độ cao, gán chuẩn xác `throw`. |
| **`video17`** | Cặp đôi đi dạo nói chuyện trong sân (50s). | 1 | `[4] hold [7]` (0.75)<br>`[7] hold [4]` (0.69) | Bắt cử chỉ đi kề cận thân mật; xe máy đỗ xung quanh được bảo vệ cô lập. |
| **`video18`** | Cặp đôi nắm tay đi dạo qua bãi xe máy (54s). Hộp người đè lên xe máy tiền cảnh. | 1 | `[5] hold [6]` (0.91)<br>`[6] hold [5]` (0.77) | Cặp đôi được gom cụm riêng; căn chỉnh mặt phẳng nền ngăn chặn việc nuốt xe máy đỗ vào cụm. |

---

## 8. Hạn Chế Kỹ Thuật Hiện Tại & Hướng Mở Rộng

Trên tinh thần trung thực khoa học, hệ thống hiện tại còn một số giới hạn thực tế cần lưu ý:

1. **Giả định mặt phẳng sàn trong phép chiếu đơn kính (Monocular 2D Projection):** Công thức căn chỉnh độ sâu chân tiếp xúc ($|y_{2, a} - y_{2, b}|$) giả định mặt đất là một mặt phẳng nằm ngang. Trong các bối cảnh phức tạp có bậc thang, dốc nghiêng hoặc cầu thang nhiều tầng, giả định này có thể gặp sai số về độ sâu.
2. **Độ nhạy nhận diện đồ vật kích thước nhỏ:** Các vật thể cầm tay quá nhỏ (điện thoại, chìa khóa, ví nhỏ) vẫn là thách thức đối với bộ nhận diện YOLOE-26m ở khoảng cách camera xa ($>15$m) hoặc trong video độ phân giải tiêu chuẩn (640x480).
3. **Che khuất lẫn nhau kéo dài (Severe Occlusion):** Trong các đám đông có mật độ di chuyển quá dày đặc, nhiều người ăn mặc tương đồng đi đè lên nhau liên tục trong thời gian dài ($>3$ giây), thuật toán ByteTrack có thể bị suy giảm độ liên tục, thỉnh thoảng dẫn đến việc cấp lại ID mới sau khi tách rời.
4. **Khoảng cách chuyển giao miền dữ liệu (Domain Gap):** Mặc dù 15 video benchmark đã bao quát nhiều môi trường (trong nhà, hành lang, ngoài trời, ảnh nhiệt ban đêm), việc triển khai thực tế trên các điều kiện thời tiết khắc nghiệt mới (mưa lớn, sương mù dày, camera bị lóa đèn pha) sẽ cần tiếp tục tinh chỉnh ngưỡng nhận diện của detector.

---

## 9. Lời Cảm Ơn

Dự án được thực hiện trong khuôn khổ **Chương Trình Đào Tạo AI Trainee - Trung Tâm Thị Giác Máy Tính VinFast**.

- **Người thực hiện:** Bùi Tiến Phát
- **Người hướng dẫn (Mentor):** Lường Mạnh Tú & Trần Minh Thanh
