# Hướng Dẫn Đóng Gói & Triển Khai Docker (Docker Deployment Guide)

Tài liệu này cung cấp hướng dẫn chi tiết cách đóng gói và khởi chạy **Hệ thống Giám sát Quan hệ Thị giác VidVRD RelateAnything** bằng Docker và Docker Compose, phục vụ cho việc bàn giao mã nguồn, nghiệm thu với Mentor và demo với hội đồng VinUni.

---

## 1. Tổng Quan Kiến Trúc Đóng Gói (Container Architecture)

Container hóa toàn bộ hệ thống giúp loại bỏ hoàn toàn các lỗi phụ thuộc môi trường (lệch phiên bản Python, CUDA driver, thư viện OpenCV/PyTorch):

- **Base Image:** `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` (Image chính thức từ NVIDIA/PyTorch hỗ trợ tăng tốc GPU tối đa).
- **Hệ điều hành container:** Ubuntu Linux 22.04 LTS mini.
- **Hỗ trợ phần cứng:** Tự động nhận diện GPU NVIDIA qua `nvidia-container-toolkit` hoặc tự động chuyển về CPU nếu máy chủ không có GPU rời.
- **Cổng dịch vụ (Exposed Port):** `8501` (Cổng mặc định của Streamlit Dashboard).
- **Dữ liệu gắn kết (Mounted Volumes):**
  - `./data` $\rightarrow$ `/app/data` (Lưu trữ video đầu vào, video kết quả và file JSON).
  - `./models` $\rightarrow$ `/app/models` (Lưu trữ trọng số YOLOE và RelateAnything, không cần tải lại mỗi lần build).

---

## 2. Yêu Cầu Cài Đặt Trên Máy Chủ / Máy Demo

1. **Docker Engine & Docker Compose:**
   - Trên Ubuntu Linux: `sudo apt install docker.io docker-compose-v2`
   - Trên Windows / Mac: Cài đặt [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. **NVIDIA Container Toolkit (Tùy chọn - Dành cho tăng tốc GPU):**
   - Giúp Docker giao tiếp trực tiếp với card đồ họa NVIDIA.
   - Hướng dẫn cài đặt chính thức: [NVIDIA Container Toolkit Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

---

## 3. Các Lệnh Khởi Chạy Nhanh (Quick Start)

### Cách 1: Sử dụng Docker Compose (Khuyên dùng - 1 dòng lệnh)

Chỉ cần đứng tại thư mục gốc của dự án (`c:\AIThucChien\RelateAnything` hoặc trên Linux `/path/to/RelateAnything`):

```bash
# Build và khởi chạy container chạy nền:
docker compose up -d --build
```

Sau khi chạy xong, mở trình duyệt web bất kỳ và truy cập vào:
👉 **`http://localhost:8501`**

Để dừng container:
```bash
docker compose down
```

---

### Cách 2: Sử dụng Docker CLI thuần

#### Bước 1: Build Docker Image
```bash
docker build -t vidvrd-relate-anything:latest .
```

#### Bước 2: Chạy Container với GPU NVIDIA
```bash
docker run -d \
  --name vidvrd_app \
  --gpus all \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  vidvrd-relate-anything:latest
```

*(Nếu máy không có card NVIDIA, chỉ cần bỏ tham số `--gpus all` để chạy ở chế độ CPU).*

#### Bước 3: Xem log hoạt động
```bash
docker logs -f vidvrd_app
```

#### Bước 4: Dừng container
```bash
docker stop vidvrd_app && docker rm vidvrd_app
```

---

## 4. Giải Thích Các Tham Số Cốt Lõi Khi Trình Bày Với Hội Đồng / Mentor

Nếu Mentor hoặc Ban Giám khảo VinUni hỏi về thiết kế Docker, bạn có thể trả lời các điểm kỹ thuật cốt lõi sau:

1. **Tại sao dùng Multi-volume mounting (`./data` và `./models`)?**
   - Tránh việc Docker image bị phình to hàng chục GB do chứa video và model weights.
   - Khi chạy phân tích một video mới, video xuất ra và file JSON sẽ xuất hiện ngay trên máy thật trong thư mục `data/output/` mà không bị mất đi khi tắt container.
2. **Khả năng tương thích nền tảng (Platform Portability):**
   - Dự án đã được kiểm thử tính độc lập hoàn toàn, không phụ thuộc vào đường dẫn tuyệt đối Windows `C:\...`, toàn bộ file gọi theo đường dẫn tương đối POSIX tiêu chuẩn.
3. **Cơ chế Healthcheck:**
   - Trong Dockerfile có cấu hình `HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health` giúp các hệ thống điều phối như Kubernetes / Docker Swarm tự động giám sát tình trạng sống của Web Dashboard.
