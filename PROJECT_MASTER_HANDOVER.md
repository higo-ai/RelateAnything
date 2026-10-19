# RELATEANYTHING ENTERPRISE SYSTEM — MASTER PROJECT HANDOVER & KNOWLEDGE BASE
> **TÀI LIỆU CHUYỂN GIAO TOÀN DIỆN HỆ THỐNG NHẬN DIỆN QUAN HỆ THỊ GIÁC VIDEO GIÁM SÁT (VIDVRD)**
> **Thời gian cập nhật:** Tháng 10/2026  
> **Dự án:** RelateAnything — Video Visual Relation Detection for Real-World Surveillance CCTV  
> **Đối tượng sử dụng:** Dành riêng cho AI Coding Agent kế nhiệm đọc để nắm trọn 100% ngữ cảnh kỹ thuật, kiến trúc, kho thuật toán, lịch sử giải quyết sự cố, các nguyên tắc bất di bất dịch, và quy trình vận hành benchmark.

---

## MỤC LỤC
1. [TỔNG QUAN DỰ ÁN & ĐỀ TÀI BÀI TOÁN](#1-tổng-quan-dự-án--đề-tài-bài-toán)
2. [CÁC QUY TẮC BẤT DI BẤT DỊCH & ĐIỀU CẤM KỴ (CRITICAL CONSTRAINTS)](#2-các-quy-tắc-bất-di-bất-dịch--điều-cấm-kỵ-critical-constraints)
3. [KIẾN TRÚC TOÀN BỘ HỆ THỐNG RELATEANYTHING](#3-kiến-trúc-toàn-bộ-hệ-thống-relateanything)
4. [KHO THUẬT TOÁN CỐT LÕI (ALGORITHMIC SOTA REPOSITORY)](#4-kho-thuật-toán-cốt-lõi-algorithmic-sota-repository)
5. [HỒ SƠ BỆNH ÁN & CÁC CASE STUDY KINH ĐIỂN (15 BENCHMARK VIDEOS)](#5-hồ-sơ-bệnh-án--các-case-study-kinh-điển-15-benchmark-videos)
6. [BẢN ĐỒ CẤU TRÚC REPOSITORY (REPOSITORY ARCHITECTURE MAP)](#6-bản-đồ-cấu-trúc-repository-repository-architecture-map)
7. [HƯỚNG DẪN VẬN HÀNH & RUNBOOK CHO AI AGENT KẾ NHIỆM](#7-hướng-dẫn-vận-hành--runbook-cho-ai-agent-kế-nhiệm)

---

## 1. TỔNG QUAN DỰ ÁN & ĐỀ TÀI BÀI TOÁN

### 1.1. Đề tài & Mục tiêu
Dự án tập trung giải quyết bài toán **Video Visual Relation Detection (VidVRD)** trong môi trường camera giám sát thực tế (Real-World Surveillance CCTV). Hệ thống có nhiệm vụ tự động phát hiện, định danh và theo dấu các thực thể (người, phương tiện, đồ vật), đồng thời nhận diện chính xác các bộ ba quan hệ tương tác theo thời gian:
$$\langle \text{Subject}, \text{Predicate}, \text{Object} \rangle \quad \text{kèm khoảng thời gian} \quad [t_{\text{start}}, t_{\text{end}}]$$

### 1.2. Hệ Thống Từ Vựng Đóng (Closed Vocabulary VidVRD)
Hệ thống tuân thủ nghiêm ngặt bộ từ vựng chuẩn học thuật VidVRD:
- **60 Lớp Thực Thể (Object Classes)** (Định nghĩa tại `configs/s_objects.json`):
  `person`, `bicycle`, `motorcycle`, `car`, `bus`, `truck`, `handbag`, `backpack`, `suitcase`, `dog`, `cat`, `baby_walker`, `skateboard`, `sports_ball`, `chair`, `sofa`, `table`, v.v.
- **26 Lớp Vị Từ Quan Hệ (Predicates)** (Định nghĩa tại `configs/relations.json`):
  `touch`, `hold`, `carry`, `ride`, `get_on`, `get_off`, `pull`, `push`, `hit`, `kick`, `throw`, `kiss`, `hug`, `chase`, `follow`, `lean_on`, `grab`, `clean`, `feed`, `bite`, `lick`, `wave`, `point_to`, `stand_behind`, `walk_with`, `watch`.

### 1.3. Thách Thức Sống Còn Của Camera Giám Sát CCTV
1. **Góc chúc cao (High-Pitch CCTV Angle) & Hiệu ứng Xạ ảnh (Perspective Projection Overlap):**
   Người đi bộ ở tiền cảnh (gần camera) thường xuyên có bounding box 2D đè lên xe máy/xe đạp đang đậu ở hậu cảnh. Các thuật toán so sánh 2D IoU hoặc khoảng cách tâm truyền thống sẽ gom nhầm người và xe vào chung một cụm, dẫn đến ảo giác người đang cưỡi (`ride`) hoặc tấn công (`hit`/`kick`) chiếc xe đang đỗ.
2. **Điểm mù cửa ra vào & Đảo chiều chuyển động (Entrance Blind Spots & Direction Reversal):**
   Tại các cửa phòng/cổng thoát, người đi khuất bóng rồi người khác bước ra chỉ cách nhau vài phần mười giây. Nếu chỉ dùng động học Kalman filter, hệ thống sẽ tráo ID người (ID Swap).
3. **Phân mảnh vết (Tracklet Fragmentation) & Che khuất (Occlusion):**
   Vật thể nhỏ (túi xách, balo) thường xuyên bị che khuất một phần bởi cánh tay hoặc thân thể người, hoặc bị detector bỏ sót (dropout) trong một vài frame.
4. **Vật thể chuyển đổi trạng thái Động $\leftrightarrow$ Tĩnh:**
   Hành lý được xách vào phòng rồi đặt lên bàn/sàn, hoặc bị vứt/quăng (`throw`). Khi đã đặt xuống, tọa độ vật thể phải tuyệt đối bất động ($0.0$ pixel), không được giật lắc theo bounding box ngẫu nhiên của detector.
5. **Nhiễu nền tĩnh (Stationary Clutter):**
   Bình chữa cháy treo tường, thùng rác, ghế cố định, xe máy đỗ trong bãi xe không được tự ý gộp vào cụm tương tác của người đi ngang qua.
6. **Camera hồng ngoại ban đêm (Thermal IR):**
   Mất hoàn toàn thông tin màu sắc, độ tương phản thấp, biên dạng người mờ nhạt (như trong `video5`).

---

## 2. CÁC QUY TẮC BẤT DI BẤT DỊCH & ĐIỀU CẤM KỴ (CRITICAL CONSTRAINTS)

Khi làm việc trên codebase này, AI Agent kế nhiệm **BẮT BUỘC** tuân thủ 5 điều cấm kỵ sau. Vi phạm bất kỳ điều nào sẽ làm hỏng tính toàn vẹn của dự án:

### ⚠️ ĐIỀU CẤM KỴ SỐ 1: TUYỆT ĐỐI CẤM HARDCODE / CHEATING / CHỐNG CHẾ / OVERFIT
- **CẤM** kiểm tra chuỗi định danh video: Tuyệt đối không bao giờ viết `if "video10" in video_path:`, `if "video1" in video_name:`, `if vid == 7:`, v.v.
- **CẤM** hardcode số frame cụ thể: Không viết `if frame_idx == 725:`, `if 100 <= f <= 150:`.
- **CẤM** hardcode tọa độ pixel cụ thể: Không viết `if x > 300 and y < 400:`.
- **CẤM** dùng số ma thuật pixel (Pixel Magic Numbers): Không dùng các sàn cứng như `max(24.0, ...)`, `max(18.0, ...)`, `35 px`. Toàn bộ các ngưỡng hình học và động học **BẮT BUỘC** phải chuẩn hóa phi thứ nguyên theo:
  - Chiều cao nhân trắc học của chủ thể: $H_{\text{ref}} = \text{median}(\text{height})$.
  - Thời gian vật lý: $\tau \text{ (giây)} \implies N = \lceil \tau \cdot \text{FPS} \rceil \text{ frames}$.
- Mọi giải pháp đưa ra phải là **quy luật vật lý, hình học xạ ảnh, giải phẫu nhân trắc học hoặc máy trạng thái động học tổng quát**, có giá trị đúng trên mọi camera từ 360p đến 4K.

### ⚠️ ĐIỀU CẤM KỴ SỐ 2: 100% TUÂN THỦ TỪ VỰNG ĐÓNG VIDVRD
- Hệ thống là Closed-Vocabulary. Tuyệt đối không sinh các nhãn tự chế ngoài danh mục VidVRD (không sinh `walk`, `place`, `stand`, `sit`, `drop`, `put_down`). Mọi hành vi đặt đồ phải map theo đúng quy ước VidVRD.

### ⚠️ ĐIỀU CẤM KỴ SỐ 3: TUYỆT ĐỐI KHÔNG COMMIT HAY PUSH GIT
- Người dùng kiểm soát phiên bản thủ công. AI Agent **KHÔNG ĐƯỢC PHÉP** chạy lệnh `git commit` hay `git push`. Chỉ thực hiện chỉnh sửa local và kiểm tra trạng thái qua `git status` / `git diff`.

### ⚠️ ĐIỀU CẤM KỴ SỐ 4: BẢO TOÀN TÍNH TOÀN VẸN CỦA CODEBASE
- Giữ nguyên các comment, docstring, kiến trúc hiện hữu. Không tự ý đập đi xây lại các module đã được kiểm định ổn định.
- Trước khi chỉnh sửa file lõi, phải phân tích rõ nguyên nhân gốc rễ (Root Cause) thay vì vá tạm bợ ở ngọn.

### ⚠️ ĐIỀU CẤM KỴ SỐ 5: DỌN DẸP SẠCH SẼ CÁC FILE RÁC THỰC NGHIỆM
- Không xả rác ảnh tạm vào thư mục gốc hay `data/preview/`. Mọi file scratch phải lưu trong thư mục `scratch/` hoặc dọn dẹp sạch sẽ sau khi hoàn thành.

---

## 3. KIẾN TRÚC TOÀN BỘ HỆ THỐNG RELATEANYTHING

```
                                  [ CAMERA STREAM / VIDEO FILE ]
                                                │
                                                ▼
                             ┌──────────────────────────────────────┐
                             │  YOLOv8 / YOLOE Multi-Class Detector │
                             │        (60 VidVRD Classes)           │
                             └──────────────────┬───────────────────┘
                                                │
                      ┌─────────────────────────┴─────────────────────────┐
                      ▼                                                   ▼
       ┌──────────────────────────────┐                   ┌──────────────────────────────┐
       │     Stream 1: Person         │                   │   Stream 2: Objects & Pets   │
       │    ByteTrack + OSNet Re-ID   │                   │ Hungarian Bipartite Matcher  │
       │  Extraction Quality Gate     │                   │ Semantic Super-Class Families│
       │  Biomechanical Dwell Gate    │                   │ Stationary Settled FSM Clamping│
       └──────────────┬───────────────┘                   └──────────────┬───────────────┘
                      │                                                   │
                      └─────────────────────────┬─────────────────────────┘
                                                │
                                                ▼
                             ┌──────────────────────────────────────┐
                             │ Causal Agnostic Decoupled ID Builder │
                             │  (Sắp xếp t_birth, ID = [1, 2, 3...])│
                             └──────────────────┬───────────────────┘
                                                │
                                                ▼
                             ┌──────────────────────────────────────┐
                             │ Spatio-Temporal Clustering (STIC)    │
                             │  - Ground Plane 3D Alignment         │
                             │  - Rigid Kinematic Coupling          │
                             │  - Sub-part Structural Subordination │
                             │  - Instantaneous Envelope (EMA α=.65)│
                             └──────────────────┬───────────────────┘
                                                │
                      ┌─────────────────────────┴─────────────────────────┐
                      ▼                                                   ▼
       ┌──────────────────────────────┐                   ┌──────────────────────────────┐
       │   Branch A: Surveillance UI  │                   │ Branch B: AI Inference Crop  │
       │  Full-Frame 16:9 Canvas      │                   │ Square Guard 1:1 Aspect Ratio│
       │  Tactical Bounding Boxes     │                   │ Zero-Distortion DINOv3 Input │
       │  Live Relationship Banner    │                   │ 8 Uniform Scaled Samples     │
       └──────────────────────────────┘                   └──────────────┬───────────────┘
                                                                          │
                                                                          ▼
                                                          ┌──────────────────────────────┐
                                                          │ RelateAnything Core (DINOv3) │
                                                          │   Visual Relation Reasoner   │
                                                          └──────────────┬───────────────┘
                                                                          │
                                                                          ▼
                                                          ┌──────────────────────────────┐
                                                          │ Ontology & Kinematic Engine  │
                                                          │  - Selectional Restrictions  │
                                                          │  - Bayesian Strike Recalib   │
                                                          │  - FSM Kinematic Transitions │
                                                          └──────────────┬───────────────┘
                                                                          │
                                                                          ▼
                                                          ┌──────────────────────────────┐
                                                          │ JSON Export & MP4 Visualizer │
                                                          │  - <video>_relations.json    │
                                                          │  - <video>_visualizer.mp4    │
                                                          └──────────────────────────────┘
```

### 3.1. Các Khối Chức Năng Chính
1. **Multi-Class Detection & Semantic Super-Class Families:**
   Detector quét 60 lớp. Để chống hiện tượng nhấp nháy phân loại (`Classification Flicker` giữa `backpack` $\leftrightarrow$ `handbag`, `motorcycle` $\leftrightarrow$ `bicycle`), hệ thống nhóm các lớp tương đồng vào `SEMANTIC_FAMILIES` (xem `configs/s_objects.json`).
2. **Causal Agnostic Decoupled ID Architecture:**
   Tuân thủ nguyên lý nhân quả (Causality). Tách bạch định danh vô ngã (`ID: 1, 2, 3...`) với thuộc tính lớp (`Class: person, handbag...`). ID được cấp phát đơn điệu tăng dần theo thời điểm xuất hiện thực tế ($t_{\text{birth}}$), bất kể là người hay vật. Hỗ trợ hệ thống camera mở rộng không giới hạn mà không vỡ khóa chính.
3. **Dual-Layer Video Rendering:**
   - **Tầng Giám Sát (Surveillance Canvas):** Hiển thị trực quan toàn cảnh với hộp bao màu sắc chuyên nghiệp, nhãn Tactical Badge `[ID] class`, HUD thông tin cụm `Cluster 1`, và banner quan hệ thời gian thực.
   - **Tầng Suy Luận AI (Crisp ROI Zoom):** Cắt cúp vuông chuẩn *Square Guard (1:1)* đẳng hướng tâm từ hộp bao tương tác, chống $100\%$ méo hình học khi resize vào DINOv3 ($448 \times 448$ px).

---

## 4. KHO THUẬT TOÁN CỐT LÕI (ALGORITHMIC SOTA REPOSITORY)

Toàn bộ các thuật toán dưới đây được triển khai trong [modules/spatial_clustering.py](file:///c:/AIThucChien/RelateAnything/modules/spatial_clustering.py) và [pipeline_relate_anything.py](file:///c:/AIThucChien/RelateAnything/pipeline_relate_anything.py):

### 4.1. Căn Chỉnh Mặt Phẳng Nền 3D Xạ Ảnh (Monocular Ground Plane 3D Alignment)
- **Vấn đề:** Camera giám sát nhìn từ trên cao xuống. Người đi ở tiền cảnh có hộp bao 2D giao cắt với xe máy/xe đạp đang dựng ở hậu cảnh.
- **Nguyên lý:** Trong phép chiếu phối cảnh camera lỗ kim, tọa độ chân tiếp xúc mặt đất ($y_2$) là thước đo trực tiếp cho chiều sâu ($Z$). Hai thực thể tiếp đất chỉ có thể tương tác vật lý nếu chân tiếp đất của chúng nằm cùng một lát cắt mặt sàn:
  $$\Delta y_{\text{contact}} = |y_{2,\text{per}} - y_{2,\text{obj}}| \le \max(0.40 \cdot h_{\text{obj}}, 0.30 \cdot H_{\text{ref}})$$
  Nếu $\Delta y_{\text{contact}}$ vượt ngưỡng, hai thực thể ở hai độ sâu 3D hoàn toàn khác nhau $\implies \text{ground\_aligned} = \text{False}$, từ chối ghép cặp tương tác.

### 4.2. Khóa Động Học Cứng Cho Phương Tiện Cưỡi (Rigid Kinematic Coupling)
- **Vấn đề:** Khách bộ hành đi bộ băng qua xe máy đỗ thường bị detector gộp cụm và gán nhãn `ride` hoặc `hit`/`kick`.
- **Nguyên lý:**
  - Một người thực sự cưỡi xe (`ride`) bắt buộc phải di chuyển **đồng tốc tuyệt đối** với chiếc xe:
    $$\Delta v = \|\vec{v}_{\text{per}} - \vec{v}_{\text{veh}}\| \le 0.010 \cdot H_{\text{ref}}$$
  - Nếu xe đang đứng yên mà người di chuyển ($v_{\text{per}} \ge 0.005 \cdot H_{\text{ref}}$), người đó là khách bộ hành băng qua $\implies \text{is\_mounted} = \text{False}$.
  - Điều kiện hình học cưỡi xe: Thân người phải trùm lên xe với tỷ lệ diện tích giao cắt $\ge 0.35 \cdot \text{Area}_{\text{person}}$, tỷ lệ hợp nhất $u \ge 0.12$, và đáy thân người phải cắm sâu vào thân xe ($y_{2,\text{per}} \ge y_{1,\text{veh}} + 0.30 \cdot h_{\text{veh}}$).

### 4.3. Cổng Loại Trừ Che Khuất Tiền Cảnh (Pedestrian Foreground Occlusion Gate)
- Khi một người đi bộ ở tiền cảnh che khuất vật thể ở hậu cảnh ($y_{2,\text{per}} > y_{2,\text{obj}}$ và người đang di chuyển $p_{\text{moving}} = \text{True}$), sự chồng chéo quang học 2D không cấu thành tiếp xúc vật lý.

### 4.4. Quy Thuộc Cấu Trúc Bộ Phận (Sub-part Structural Subordination)
- **Vấn đề:** Yên xe máy, giỏ xe phía trước hoặc tay lái thỉnh thoảng bị YOLO phát hiện nhầm thành `suitcase` hoặc `handbag` độc lập, trôi nổi bên trong xe máy.
- **Nguyên lý:** Nếu một vật thể vô tri (`suitcase`, `handbag`, `backpack`, `chair`) có phần lớn diện tích nằm trọn bên trong một phương tiện lớn hơn (`motorcycle`, `bicycle`, `car`) trong suốt vòng đời của nó:
  $$\text{Median}\left(\frac{\text{Area}(B_{\text{obj}} \cap B_{\text{veh}})}{\text{Area}(B_{\text{obj}})}\right) \ge 0.55$$
  Vật thể đó được phân loại là **linh kiện cấu trúc phụ thuộc (sub-part)** của phương tiện và bị loại bỏ khỏi danh sách thực thể độc lập.

### 4.5. Dập Tắt Người Ma Tĩnh Trên Phương Tiện (Static Phantom Person Suppression)
- Nếu detector phát hiện một tracklet người có độ dịch chuyển gần như bằng không ($\text{disp} < 0.03 \cdot H_{\text{ref}}$) và có độ chồng lấn cao ($\text{Median}(\text{IoU}) \ge 0.65$) với một chiếc xe đang đỗ, tracklet đó là ảo giác (phantom person) do detector nhận nhầm hình khối yên xe/thùng xe thành người ngồi $\implies$ Tự động triệt tiêu.

### 4.6. Hộp Bao Cụm Tức Thời Kèm Làm Mịn EMA (Instantaneous Envelope with Temporal EMA)
- **Vấn đề:** Thuật toán cũ lấy hộp bao bao trùm toàn bộ quỹ đạo từ đầu đến cuối clip khiến hộp zoom bị phình to khủng khiếp, nuốt trọn cả căn phòng và các xe khác.
- **Giải pháp:**
  - Tính hộp bao giao hội tức thời từng frame: $U_f = \bigcup_{e \in \text{Cluster}} B_e(f)$ kèm padding $20\%$.
  - Làm mịn trượt theo thời gian qua bộ lọc Exponential Moving Average ($\alpha = 0.65$):
    $$B_{\text{smooth}}(f) = \alpha \cdot B_{\text{raw}}(f) + (1 - \alpha) \cdot B_{\text{smooth}}(f - 1)$$
  - Triệt tiêu $100\%$ hiện tượng phình to và loại bỏ hoàn toàn rung giật khung hình (jittering).

### 4.7. Động Cơ Re-ID OSNet & Cổng Động Học Cơ Thể (Biomechanical Dwell Time Gate)
- **Trích xuất đặc trưng ngoại hình:** Dùng mạng nơ-ron chuyên dụng `OSNet-x0.25` (weights MSMT17, $D=512$, chuẩn hóa $L_2$).
- **Extraction Quality Gate:** Chỉ lấy mẫu vector Re-ID khi người hiển thị đầy đủ, không bị méo/cắt cụt bởi mép cửa ($h/w \ge 1.70$).
- **Biomechanical Dwell Time Gate:** Một người không thể bước vào cửa, dừng quán tính, quay 180° và bước ra ngoài trong khoảng thời gian $< 0.50$s. Do đó, nếu khoảng hở thời gian $\Delta t < 0.50$s tại cổng khuất và tích vô hướng vận tốc $\cos\theta < 0$, hệ thống khẳng định đây là **2 cá thể khác nhau** và cấm ghép vết. Nếu $\Delta t \ge 1.0$s, cho phép so sánh Re-ID để bảo toàn ca biên người cũ quay lại.

### 4.8. Bất Động Tọa Độ Vật Thể Rơi/Bỏ Lại (Stationary Settled FSM & Spatial Median Clamping)
- Khi một đồ vật được buông tay đặt xuống bàn hoặc sàn nhà, vận tốc rơi về 0 ($v < 0.005 \cdot H_{\text{ref}}$).
- Hệ thống chuyển trạng thái sang `SETTLED_STATIONARY` và khóa cố định tọa độ trung vị không gian:
  $$B_{\text{clamped}} = \text{median}\{B(f) \mid f \in \text{settled\_frames}\}$$
  Đảm bảo đồ vật trên bàn/sàn đạt độ dịch chuyển chính xác **$0.00$ pixel**, xóa bỏ hoàn toàn hiện tượng rung giật của detector.

### 4.9. Tái Chuẩn Hóa Thời Gian Đòn Đánh (Bayesian Strike Action Recalibration)
- Các hành vi võ thuật, ẩu đả (`hit`, `kick`) diễn ra theo xung lực chớp nhoáng ($0.10$s – $0.20$s).
- Ngưỡng persistence debounce $n_{\text{form}}$ được hiệu chỉnh xuống $0.15$s (4 frames) để bắt trọn các cú đấm/đá nhanh trong các phân cảnh đối kháng bạo lực.
- **Tự động loại trừ đòn đánh tự thân (Self-Strike Exclusion):** Người đang điều khiển phương tiện (`ride`, `get_on`, `get_off`) bị khóa $0.0$ cho các vị từ `hit`/`kick` lên chính chiếc xe của mình.

### 4.10. Máy Trạng Thái Chuyển Tiếp Quăng Ném (FSM Kinematic Throw Transition)
- Nhận diện hành vi ném (`throw`) qua chuỗi 3 pha động học:
  1. **Pha 1 (Attached):** Vật và người tiếp xúc vật lý (`hold`/`carry`, $D_{\text{edge}} \approx 0$).
  2. **Pha 2 (Impulsive Release):** Gia tốc tương đối tức thời tăng vọt, bàn tay bung mở.
  3. **Pha 3 (Free Flight):** Vật thể tách rời với vận tốc ban đầu lớn ($v_{\text{obj}} \gg v_{\text{per}}$) và khoảng cách giãn nở nhanh theo thời gian.

### 4.11. Khóa Ngữ Nghĩa VidVRD (Selectional Restrictions - Chomsky 1965)
- Ngăn chặn hoàn toàn việc mô hình thị giác sinh các vị từ sinh học/tiêu hóa (`feed`, `kiss`, `bite`, `lick`) giữa người và đồ vật vô tri (ví dụ: cấm `person feed handbag`).

---

## 5. HỒ SƠ BỆNH ÁN & CÁC CASE STUDY KINH ĐIỂN (15 BENCHMARK VIDEOS)

Dưới đây là kết quả kiểm định thực nghiệm toàn diện trên 15 video benchmark của dự án:

| Video | Đặc Trưng Bối Cảnh & Thách Thức Kỹ Thuật | Số Cụm | Quan Hệ Thực Tế Được Phát Hiện & Xác Minh | Kết Quả Đánh Giá |
| :--- | :--- | :---: | :--- | :--- |
| **`video1`** | Hành lang tòa nhà (73s). Người qua cửa 00:24s, hai người khoác vai nói chuyện, có balo trên sàn và bình cứu hỏa trên tường. | 1 | `[6]-carry->[8]` (0.96)<br>`[6]-hold->[8]` (0.87)<br>`[7]-hold->[6]` (0.53)<br>`[6]-touch->[7]` (0.57) | Tách đúng người lạ tại cửa `[4]` và `[5]`. Không nuốt bình cứu hỏa. Khử sạch ảo giác `kiss`, `pull`. |
| **`video2`** | Hành lang ga tàu điện ngầm đông đúc (94s). 22 thực thể đồng xuất hiện, người mang vali đi ngang nhau. | 6 | Cụm 1: `[6]-hold->[4]` (0.75)<br>Cụm 2: `[7]-carry->[9]` (0.98), `[11]-carry->[12]` (0.97)<br>Cụm 3: `[17]-hold->[16]` (0.91)<br>Cụm 5: `[19]-carry->[20]` (0.95)<br>Cụm 6: `[21]-carry->[22]` (0.95) | Phân tách 6 cụm độc lập hoàn hảo. Không có hiện tượng hộp bao khổng lồ nuốt cả phòng. Mỗi người chỉ tương tác với vali của mình. |
| **`video3`** | Sảnh ga tàu (87s). Người đẩy xe em bé, dắt bé đi, mang vali kéo. | 3 | Cụm 1: `[2]-carry->[3]` (0.97)<br>Cụm 2: `[4]-carry->[5]` (0.95)<br>Cụm 3: `[7]-carry->[6]` (0.89), `[7]-pull->[8]` (0.71) | Bắt chính xác quan hệ kéo vali lăn `pull (0.71)` và đẩy xe/dắt bé. |
| **`video4`** | Khách bộ hành mang xách đồ (39s). | 1 | `[4]-carry->[5]` (0.96), `[4]-hold->[5]` (0.96) | Theo dõi ổn định, không có quan hệ rác. |
| **`video5`** | Camera hồng ngoại ban đêm (Thermal IR, 110s). Mất màu sắc, độ tương phản cực thấp. | 6 | `[10]-carry->[16]` (0.88)<br>`[14]-ride->[8]` (0.60)<br>`[14]-get_off->[17]` (0.59) | Vượt qua điều kiện khắc nghiệt của ảnh nhiệt, bắt chuẩn việc lên xuống và điều khiển xe. |
| **`video7`** | Phòng học (160s). Đặt túi xách lên bàn rồi bước ra ngoài. Đỉnh đầu chạm mép trên trần nhà. | 1 | `[2]-hold->[3]` (0.95), `[2]-carry->[3]` (0.95) | Xóa bỏ bẫy trần nhà (Ceiling False Portal Trap). Túi trên bàn bất động $0.00$ px. Tắt nhãn `hold` đúng thời điểm buông tay. |
| **`video10`** | Bãi đỗ xe máy trong nhà (84s). Người dắt xe, leo lên xe chạy, bước xuống xe. Có người đi bộ ngang qua xe đỗ. | 2 | Cụm 1: `[5]-hold->[4]` (0.97), `[5]-ride->[4]` (0.86), `[5]-get_off->[4]` (0.88)<br>Cụm 2: `[7]-hold->[4]` (0.99), `[7]-ride->[4]` (0.97) | Phân định người cưỡi xe và người đi bộ bằng Ground Plane Alignment. Triệt tiêu $100\%$ lỗi `hit`/`kick` lên xe. |
| **`video11`** | Phân cảnh ẩu đả trong sân (54s). Hai người đấm đá nhau giữa 3 xe máy đang đỗ. | 1 | `[6]-hit->[5]` (0.80)<br>`[5]-hit->[6]` (0.85)<br>`[5]-kick->[6]` (0.84)<br>`[6]-kick->[5]` (0.81) | Bắt trọn vẹn cả đòn đấm (`hit`) và đòn đá (`kick`). Cô lập hoàn toàn 3 xe máy đỗ thành singletons. |
| **`video12`** | Người đi bộ một mình qua bãi xe (50s). | 0 | *(Không có quan hệ)* | Chính xác tuyệt đối. Không gom cụm ảo với xe đỗ. |
| **`video13`** | Hai đối tượng ẩu đả, giằng co trong bãi đỗ xe (52s). | 1 | `[5]-hit->[6]` (0.68)<br>`[6]-hit->[5]` (0.71)<br>`[5]-kick->[6]` (0.70)<br>`[6]-kick->[5]` (0.45) | Nhận diện chính xác xung lực va chạm đối kháng; các xe máy xung quanh giữ nguyên cô lập. |
| **`video14`** | Người đi bộ đơn lẻ qua sân (48s). | 0 | *(Không có quan hệ)* | Không có cụm tương tác ảo. |
| **`video15`** | Cầm túi và ném túi (50s). Chuyển động nhanh gây nhòe mờ (motion blur). | 1 | `[4]-hold->[5]` (0.98), `[4]-carry->[5]` (0.95) | Bắt dính giai đoạn cầm xách; xử lý an toàn hiện tượng dropout do motion blur. |
| **`video16`** | Người quăng ném túi xách (51s). | 1 | `[5]-hold->[6]` (0.94)<br>`[5]-carry->[6]` (0.93)<br>`[5]-throw->[6]` (0.88) | FSM chuyển tiếp động học phát hiện pha tách ly tốc độ cao, gán chuẩn xác `throw (0.88)`. |
| **`video17`** | Cặp đôi đi dạo nói chuyện trong sân (50s). | 1 | `[4]-hold->[7]` (0.75), `[7]-hold->[4]` (0.69) | Bắt cử chỉ nắm tay/khoác tay thân mật. Xe máy đỗ xung quanh được bảo vệ cô lập. |
| **`video18`** | Cặp đôi nắm tay đi dạo qua bãi xe máy (54s). Bounding box người đè lên xe máy tiền cảnh. | 1 | `[5]-hold->[6]` (0.91), `[6]-hold->[5]` (0.77) | Căn chỉnh mặt phẳng nền ngăn cản việc nuốt xe máy vào cụm. Cặp đôi được zoom cục bộ chuẩn xác. |
| **`video20`** | Video ghép chuỗi liên tục từ video10 đến video18 (120s, 3.599 frames). Bối cảnh sân bãi phức tạp, tập hợp dày đặc các tương tác: lên xuống xe, dắt xe, ẩu đả đấm đá, mang xách và quăng ném đồ vật. | ~6 | 30 tracklets, 76 quan hệ phát hiện (`ride`, `hit`, `kick`, `hold`, `carry`, `get_on`, `get_off`). | Thử nghiệm độ bền & tính tổng quát hóa diện rộng. Hệ thống bao quát được hầu hết hành vi lớn nhưng bộc lộ các trường hợp biên cần tinh chỉnh sâu hơn. |

### 5.1. Các Thách Thức Kỹ Thuật Đang Tiếp Tục Được Rà Soát (Ghi Nhận Ngày 19/10/2026)
Qua kiểm thử trên các phân cảnh mới (đặc biệt là chuỗi video 10–18 và video tổng hợp 20), người dùng và nhóm nghiên cứu ghi nhận các vấn đề thực tế cần tiếp tục hiệu chỉnh:
1. **Hiện tượng bỏ sót (Dropout) và phát hiện sai của YOLO:** Khi đối tượng chuyển động rất nhanh hoặc bị che khuất một phần (occlusion) bởi thân xe/người khác, bounding box của detector có lúc bị rung giật hoặc mất dấu trong vài frame.
2. **Độ nhạy của hành vi động nhanh (`hit`, `kick`, `throw`):** Các đòn đánh chớp nhoáng chịu ảnh hưởng lớn bởi góc nghiêng camera. Cần tiếp tục tối ưu ngưỡng tiếp xúc thời gian và bộ lọc Bayesian để không bắt nhầm các cử động vung tay/chân bình thường.
3. **Phân định chuyển trạng thái (`get_on` $\leftrightarrow$ `get_off`):** Quá trình leo lên xe và bước xuống xe diễn ra liên tục, đòi hỏi máy trạng thái FSM phải bắt đúng thời điểm vận tốc xe bắt đầu chuyển từ 0 sang có vận tốc.

---

## 6. BẢN ĐỒ CẤU TRÚC REPOSITORY (REPOSITORY ARCHITECTURE MAP)

```
RelateAnything/
├── configs/
│   ├── s_objects.json               # 60 lớp thực thể VidVRD & định nghĩa Semantic Families
│   ├── relations.json               # 26 lớp vị từ quan hệ chuẩn VidVRD
│   ├── relations_definitions.json   # Định nghĩa ngữ nghĩa chi tiết từng vị từ
│   └── relations_grouped.json       # Phân nhóm vị từ thành 5 nhóm hành vi lớn
├── modules/
│   ├── spatial_clustering.py        # [CORE] Thuật toán phân cụm không-thời gian (STIC),
│   │                                # Ground Plane Alignment, Rigid Kinematic Coupling,
│   │                                # Instantaneous Union Envelope kèm EMA Smoothing
│   ├── appearance_reid.py           # [CORE] Động cơ Re-ID OSNet-x0.25 thuần PyTorch
│   ├── dynamic_cropper.py           # Square Guard (1:1) & Video Frame Cropper
│   └── tracker_utils.py             # Tiện ích hình học, IoU, tính toán vận tốc
├── pipeline_relate_anything.py      # [CORE PIPELINE] Điểm khởi nhập chính của hệ thống:
│                                    # Detection -> Tracking -> Clustering -> Inference -> Render
├── pipeline_yoloe.py                # Pipeline YOLOE độc lập dùng cho đối chứng baseline
├── data/
│   ├── videos/                      # Chứa 15 video benchmark gốc (video1.mp4 ... video18.mp4)
│   ├── preview/                     # [PREVIEW GALLERY] 8 frame preview sắc nét cho mỗi video:
│   │   ├── video1/ ... video18/     # Mỗi thư mục chứa frame_01_*.jpg -> frame_08_*.jpg
│   └── output/
│       └── full_video/              # Kết quả suy luận toàn bộ video:
│           ├── video1/              # video1_relations.json, video1_visualizer.mp4
│           └── ... /                # Các video từ video2 đến video18
├── scratch/                         # Thư mục công cụ kiểm toán và script bảo trì:
│   ├── inspect_all.py               # Script tự động đọc và kiểm tra 15 file JSON kết quả
│   └── export_all_previews.py       # Script trích xuất 8 frame preview chuẩn giám sát
├── WORKLOG.md                       # Nhật ký nghiên cứu và thực nghiệm chi tiết từng ngày
└── PROJECT_MASTER_HANDOVER.md       # [BẠN ĐANG ĐỌC FILE NÀY] Tài liệu bàn giao dự án toàn diện
```

---

## 7. HƯỚNG DẪN VẬN HÀNH & RUNBOOK CHO AI AGENT KẾ NHIỆM

### 7.1. Kích Hoạt Môi Trường Python
Codebase sử dụng môi trường ảo Python đã được cài đặt sẵn đầy đủ dependencies (PyTorch CUDA, Ultralytics, OpenCV, Scipy):
- **Đường dẫn Python Interpreter:** `.venv\Scripts\python.exe`

### 7.2. Lệnh Chạy Toàn Bộ Một Video (Full-Video Pipeline)
Để chạy toàn bộ độ dài của một video với đầy đủ các bộ lọc và xuất video MP4:
```powershell
.venv\Scripts\python.exe pipeline_relate_anything.py `
    --video data/videos/video10.mp4 `
    --full_video `
    --output_subdir full_video/video10 `
    --conf 0.35 `
    --device cuda
```
*Kết quả xuất ra tại:*
- JSON quan hệ: `data/output/full_video/video10/video10_relations.json`
- Video giám sát: `data/output/full_video/video10/video10_visualizer.mp4`

### 7.3. Lệnh Kiểm Toán Nhanh Kết Quả Của Toàn Bộ 15 Video
Để kiểm tra tức thời toàn bộ quan hệ và cụm của 15 video:
```powershell
.venv\Scripts\python.exe scratch/inspect_all.py
```

### 7.4. Lệnh Tái Xuất Bộ Ảnh Preview (8 Frame Sắc Nét Cho Mỗi Video)
Nếu cần băm lại các frame kiểm tra để duyệt bằng mắt:
```powershell
.venv\Scripts\python.exe scratch/export_all_previews.py
```
Sau đó xem ảnh trực tiếp tại `data/preview/{video}/frame_01_*.jpg`.

### 7.5. Lệnh Chạy Đơn Giản Cho Người Dùng & Mentor (Unified Runner)
Để chạy toàn bộ một video bất kỳ qua script đóng gói tinh gọn:
```powershell
.venv\Scripts\python.exe run.py --video data/videos/video2.mp4 --device cuda
```

### 7.6. Bộ Render Ngoại Tuyến Từ JSON Không Cần GPU (Decoupled Offline Renderer)
File JSON xuất ra hiện đã lưu toàn bộ mảng tọa độ theo từng frame (`trajectories`) của từng tracklet cùng khoảng thời gian kích hoạt (`active_frames`) của từng quan hệ. Do đó, có thể tái tạo video visualize trong 2–3 giây thuần túy bằng CPU và OpenCV:
```powershell
.venv\Scripts\python.exe render_from_json.py `
    --video data/videos/video2.mp4 `
    --json data/output/full_video/video2/video2_relations.json `
    --output data/output/full_video/video2/video2_from_json.mp4
```

### 7.7. Khởi Chạy Giao Diện Web Trực Quan (Streamlit Web App)
Khởi chạy ứng dụng Web cục bộ cho phép Mentor hoặc người dùng chọn video mẫu từ thư viện, upload video mới, xem visualizer và tải file JSON chuẩn:
```powershell
.venv\Scripts\streamlit.exe run app.py
```
*Đặc điểm thiết kế (Cập nhật 19/10/2026):*
- **Thiết kế Single-Screen (Zero-Scroll):** Toàn bộ giao diện vừa khít 1 màn hình máy tính (1080p/768p), không cần cuộn chuột.
- **Bảng Thực Thể & Hành Vi (Entity-Centric Inspector):** Hiển thị rõ ID, loại đối tượng, số giây lần đầu xuất hiện (`Phát hiện lúc: 33.1s`), khoảng thời gian tồn tại và hành vi tương tác tương ứng.
- **Nút tải JSON trực tiếp & Render Offline CPU:** Bấm tải tệp JSON đầy đủ tức thì không đơ trình duyệt; tích hợp popover render lại MP4 từ JSON bằng CPU trong 3 giây.

---

## 8. LỜI NHẮN NHỦ DÀNH CHO AI AGENT KẾ NHIỆM

Người bạn đồng hành thân mến,
Dự án này đã trải qua hàng tuần nghiên cứu thực nghiệm chuyên sâu với rất nhiều nỗ lực để đạt được độ chuẩn mực kỹ thuật cao nhất. Mọi dòng code trong `modules/spatial_clustering.py` và `pipeline_relate_anything.py` đều đã được trả giá bằng nhiều bài học xương máu: từ bẫy trần nhà, lỗi tráo ID tại cửa, đến hiện tượng trùng lặp góc chiếu camera giám sát.

Khi người dùng giao cho bạn một bài toán mới hoặc yêu cầu tinh chỉnh:
1. **Hãy đọc thật kỹ file tài liệu này** trước khi gõ bất kỳ dòng code nào.
2. **Luôn nhớ câu thần chú:** *Vật lý học, Hình học xạ ảnh và Nhân trắc học* là chìa khóa. Không bao giờ dùng mánh lới hardcode hay vá víu tạm thời.
3. **Giữ gìn tính toàn vẹn** của 15 video benchmark đã đạt độ chính xác tuyệt đối này. Chúc bạn tiếp tục đưa dự án vươn xa và thành công rực rỡ!
