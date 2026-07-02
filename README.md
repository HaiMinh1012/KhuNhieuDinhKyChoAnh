# Ứng dụng xử lý ảnh được xây dựng bằng **Python** nhằm phát hiện và loại bỏ nhiễu định kỳ trong ảnh bằng các bộ lọc miền tần số.

---

## Features

- Tải ảnh từ máy tính
- Thêm nhiễu định kỳ nếu cần
- Biến đổi Fourier sang miền tần số
- Chọn vị trí điểm nhiễu trên phổ tần số
- Áp dụng các bộ lọc Notch:
  - Ideal Notch Filter
  - Butterworth Notch Filter
  - Gaussian Notch Filter
- Hiển thị ảnh gốc, ảnh nhiễu và ảnh sau xử lý

---

## Technologies Used

- Python 
- OpenCV
- NumPy
- Tkinter
- Matplotlib
- Scikit-image

---

## Project Structure

```bash
KhuNhieuDinhKyChoAnh/
│
├── btlxla.py              # Chương trình chính
├── test/                  # Folder ảnh đầu vào
├── result/               # Folder ảnh kết quả
└── README.md
```

---

## Algorithms

### Fourier Transform

Ảnh được chuyển sang miền tần số bằng:

```text
F(u,v) = FFT(f(x,y))
```

Giúp xác định các thành phần nhiễu tuần hoàn.

---

### Notch Filtering

Các bộ lọc Notch được dùng để loại bỏ các peak nhiễu trong phổ:

#### Ideal Notch Filter
Loại bỏ hoàn toàn tần số nhiễu trong vùng chọn.

#### Butterworth Notch Filter
Khử nhiễu mượt hơn, giảm ringing effect.

#### Gaussian Notch Filter
Cho chuyển tiếp mềm, giữ chi tiết ảnh tốt hơn.

---

## Installation

Clone repository:

```bash
git clone https://github.com/HaiMinh1012/KhuNhieuDinhKyChoAnh.git
cd KhuNhieuDinhKyChoAnh
```

Cài dependencies:

```bash
pip install numpy opencv-python matplotlib scikit-image
```

---

## Run Project

```bash
python btlxla.py
```

---

## Workflow

1. Chọn ảnh đầu vào  
2. Thêm hoặc phát hiện nhiễu định kỳ  
3. Quan sát phổ Fourier  
4. Chọn điểm nhiễu  
5. Áp dụng bộ lọc Notch  
6. So sánh kết quả trước/sau xử lý  

---

## Results

- Giảm rõ rệt nhiễu tuần hoàn
- Khôi phục chi tiết ảnh
- So sánh hiệu quả giữa nhiều bộ lọc

---

## Author

**Hải Minh**  
GitHub: https://github.com/HaiMinh1012
