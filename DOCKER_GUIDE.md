# Hướng Dẫn Đóng Gói & Triển Khai Docker (Docker Deployment Guide)

Tài liệu hướng dẫn chi tiết quy trình đóng gói container và triển khai **Hệ thống Nhận Diện Quan Hệ Thị Giác VidVRD RelateAnything** bằng Docker và Docker Compose trên môi trường máy chủ và thiết bị biên.

---

## 1. Tổng Quan Kiến Trúc Đóng Gói (Container Architecture)

Container hóa toàn bộ hệ thống giúp đảm bảo tính nhất quán của môi trường thực thi (phiên bản Python, CUDA driver, các gói phụ thuộc PyTorch và OpenCV):

- **Base Image:** `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` (Image chính thức từ NVIDIA/PyTorch hỗ trợ tăng tốc GPU tối ưu).
- **Hệ điều hành container:** Ubuntu Linux 22.04 LTS.
- **Hỗ trợ phần cứng:** Tự động kích hoạt GPU NVIDIA qua `nvidia-container-toolkit` hoặc chuyển tiếp về CPU nếu môi trường không có GPU rời.
- **Cổng dịch vụ (Exposed Port):** `8501` (Cổng mặc định của Streamlit Dashboard).
- **Dữ liệu gắn kết (Mounted Volumes):**
  - `./data` $\rightarrow$ `/app/data` (Lưu trữ video đầu vào, video kết quả và file JSON).
  - `./models` $\rightarrow$ `/app/models` (Lưu trữ trọng số YOLOE và RelateAnything, không cần tải lại mỗi lần build).

---

## 2. Yêu Cầu Cài Đặt Môi Trường

1. **Docker Engine & Docker Compose:**
   - Trên Ubuntu Linux: `sudo apt install docker.io docker-compose-v2`
   - Trên Windows / macOS: Cài đặt [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. **NVIDIA Container Toolkit (Tùy chọn - Dành cho tăng tốc GPU):**
   - Cho phép container giao tiếp trực tiếp với phần cứng GPU NVIDIA.
   - Hướng dẫn cài đặt chính thức: [NVIDIA Container Toolkit Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

---

## 3. Hướng Dẫn Khởi Chạy (Quick Start)

### Cách 1: Sử dụng Docker Compose (Khuyên dùng)

Thực thi lệnh tại thư mục gốc của dự án:

```bash
# Build và khởi chạy container ở chế độ nền:
docker compose up -d --build
```

Sau khi khởi chạy thành công, mở trình duyệt web và truy cập địa chỉ:
👉 **`http://localhost:8501`**

Để dừng dịch vụ:
```bash
docker compose down
```

---

### Cách 2: Sử dụng Docker CLI thuần

#### Bước 1: Xây dựng Docker Image
```bash
docker build -t vidvrd-relate-anything:latest .
```

#### Bước 2: Chạy Container với hỗ trợ GPU
```bash
docker run -d \
  --name vidvrd_app \
  --gpus all \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  vidvrd-relate-anything:latest
```

*(Nếu hệ thống không có card đồ họa rời NVIDIA, loại bỏ cờ `--gpus all` để chạy ở chế độ CPU).*

#### Bước 3: Kiểm tra nhật ký hoạt động (Logs)
```bash
docker logs -f vidvrd_app
```

#### Bước 4: Dừng container
```bash
docker stop vidvrd_app && docker rm vidvrd_app
```

---

## 4. Các Tham Số Kỹ Thuật Cốt Lõi Trong Thiết Kế Container

Kiến trúc container hóa của hệ thống được tối ưu theo các tiêu chuẩn kỹ thuật sau:

1. **Cơ chế Multi-volume mounting (`./data` và `./models`):**
   - Phân tách hoàn toàn dữ liệu lớn (video đầu vào/đầu ra và trọng số mạng nơ-ron) khỏi lớp image, giúp kích thước Docker image luôn gọn nhẹ.
   - Kết quả phân tích video và file JSON xuất ra được đồng bộ trực tiếp về máy host tại `data/output/`, bảo toàn dữ liệu ngay cả khi container bị hủy hoặc khởi động lại.
2. **Độc lập nền tảng (Platform Portability):**
   - Toàn bộ đường dẫn tập tin trong mã nguồn sử dụng quy chuẩn tương đối POSIX, đảm bảo hệ thống hoạt động đồng nhất trên Linux, macOS và Windows Subsystem for Linux (WSL).
3. **Cơ chế giám sát sức khỏe (Healthcheck):**
   - Chỉ thị `HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1` tích hợp sẵn trong Dockerfile cho phép các hệ thống điều phối container (Docker Swarm, Kubernetes) tự động theo dõi và khởi động lại dịch vụ nếu xảy ra sự cố.
4. **Phân quyền và tài nguyên GPU:**
   - Cấu hình đặt trước thiết bị NVIDIA trong `docker-compose.yml` đảm bảo container có quyền truy cập đầy đủ vào các năng lực CUDA (`capabilities: [gpu]`) mà không yêu cầu quyền root không an toàn.
