import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tkinter import *
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

def calculate_mse(original, restored):
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)
    mse = np.mean((original - restored) ** 2)
    return mse

def calculate_snr(original, noisy):
    original = original.astype(np.float64)
    noisy = noisy.astype(np.float64)
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - noisy) ** 2)
    if noise_power == 0:
        return float('inf')
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def auto_detect_noise_points(magnitude_spectrum, threshold_ratio=0.7, min_distance=3, center_exclude_radius=30):
    h, w = magnitude_spectrum.shape
    center = (h // 2, w // 2)
    norm_mag = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
    # Dùng footprint nhỏ để phát hiện cực đại cục bộ
    neighborhood = scipy.ndimage.generate_binary_structure(2, 2)
    local_max = scipy.ndimage.maximum_filter(norm_mag, footprint=neighborhood) == norm_mag
    detected_peaks = np.where((local_max) & (norm_mag > threshold_ratio))
    peaks = []
    for y, x in zip(detected_peaks[0], detected_peaks[1]):
        if np.sqrt((y - center[0])**2 + (x - center[1])**2) > center_exclude_radius:
            peaks.append((y, x))
    return peaks


class IdealNotchFilter:
    def apply_filter(self, shape, D0=10, u_k=0, v_k=0):
        M, N = shape
        H = np.ones((M, N), dtype=np.float32)
        for u in range(M):
            for v in range(N):
                D_uv = np.sqrt((u - M/2 + u_k)**2 + (v - N/2 + v_k)**2)
                D_muv = np.sqrt((u - M/2 - u_k)**2 + (v - N/2 - v_k)**2)
                if D_uv <= D0 or D_muv <= D0:
                    H[u, v] = 0
        return H

class ButterworthNotchFilter:
    def apply_filter(self, shape, D0=10, u_k=0, v_k=0, n=2):
        M, N = shape
        H = np.ones((M, N), dtype=np.float32)
        for u in range(M):
            for v in range(N):
                D_uv = np.sqrt((u - M/2 + u_k)**2 + (v - N/2 + v_k)**2)
                D_muv = np.sqrt((u - M/2 - u_k)**2 + (v - N/2 - v_k)**2)
                H[u, v] = (1 / (1 + (D0 / D_muv)**(2 * n))) * (1 / (1 + (D0 / D_uv)**(2 * n)))
        return H

class GaussianNotchFilter:
    def apply_filter(self, shape, D0=10, u_k=0, v_k=0):
        M, N = shape
        H = np.ones((M, N), dtype=np.float32)
        for u in range(M):
            for v in range(N):
                D_uv = np.sqrt((u - M/2 + u_k)**2 + (v - N/2 + v_k)**2)
                D_muv = np.sqrt((u - M/2 - u_k)**2 + (v - N/2 - v_k)**2)
                H[u, v] = (1 - np.exp(-(D_uv**2) / (2 * D0**2))) * (1 - np.exp(-(D_muv**2) / (2 * D0**2)))
        return H

def select_notch_points(fshift):
    magnitude_spectrum = np.log(1 + np.abs(fshift))
    plt.figure(figsize=(6,6))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("Click vào các điểm gây nhiễu (ENTER khi xong)")
    points = plt.ginput(n=-1, timeout=0)
    plt.close()
    return [(int(y), int(x)) for (x, y) in points]

class NotchFilterApp:
    
    def add_noise(self):
        if not hasattr(self, 'current_img'):
            messagebox.showwarning("Lỗi", "Vui lòng chọn ảnh trước!")
            return
        img = self.current_img.copy()
        rows, cols = img.shape
        x = np.arange(cols)
        y = np.arange(rows)
        x_grid, y_grid = np.meshgrid(x, y)
        freq = 20
        sinusoid = 30 * np.sin(2 * np.pi * y_grid * freq / rows)
        noisy_img = img + sinusoid
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        self.current_img = noisy_img  # Cập nhật ảnh hiện tại
        img_pil = Image.fromarray(cv2.resize(noisy_img, (200, 200)))
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.image_label.config(image=self.tk_img)
        messagebox.showinfo("Thông báo", "Đã thêm nhiễu vào ảnh.")

    def __init__(self, master):
        self.master = master
        master.title("Notch Filter GUI")
        master.geometry("800x400")

        self.file_path = ""
        self.points = []
        self.D0 = 10
        self.filter_type = StringVar(value="All")

        main_frame = Frame(master)
        main_frame.pack(fill=BOTH, expand=True)

        left_frame = Frame(main_frame)
        left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=10, pady=10)

        self.image_label = Label(left_frame)
        self.image_label.pack(fill=BOTH, expand=True)

        right_frame = Frame(main_frame)
        right_frame.pack(side=RIGHT, fill=Y, padx=10, pady=10)

        Button(right_frame, text="Chọn ảnh", command=self.choose_image).pack(pady=5)

        Label(right_frame, text="Bán kính D0:").pack()
        self.d0_slider = Scale(right_frame, from_=1, to=50, orient=HORIZONTAL)
        self.d0_slider.set(10)
        self.d0_slider.pack(pady=5)

        Label(right_frame, text="Chọn loại filter:").pack()
        ttk.Combobox(right_frame, textvariable=self.filter_type,
                    values=["Ideal", "Butterworth", "Gaussian", "All"]).pack(pady=5)

        Button(right_frame, text="Tạo nhiễu", command=self.add_noise).pack(pady=5)
        Button(right_frame, text="Chọn điểm nhiễu", command=self.select_points).pack(pady=5)
        Button(right_frame, text="Xử lý ảnh", command=self.apply_filters).pack(pady=5)
        Button(right_frame, text="Xử lý ảnh tự động", command=self.auto_process_image).pack(pady=5)


    def choose_image(self):
        self.file_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.jpg *.png *.bmp")])
        if self.file_path:
            messagebox.showinfo("Thông báo", f"Đã chọn ảnh:\n{self.file_path}")
            img = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
            self.original_img = img.copy()  # Ảnh gốc để so sánh
            self.current_img = img
            img_resized = cv2.resize(img, (200, 200))
            img_pil = Image.fromarray(img_resized)
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.image_label.config(image=self.tk_img)

    def select_points(self):
        if not hasattr(self, 'current_img'):
            messagebox.showwarning("Lỗi", "Vui lòng chọn ảnh trước!")
            return
        img = self.current_img
        f = np.fft.fftshift(np.fft.fft2(img))
        self.points = select_notch_points(f)
        messagebox.showinfo("Thông báo", f"Đã chọn {len(self.points)} điểm.")

    def apply_filters(self):
        if not hasattr(self, 'current_img') or len(self.points) == 0:
            messagebox.showwarning("Lỗi", "Vui lòng chọn ảnh và điểm nhiễu!")
            return

        D0 = self.d0_slider.get()
        filter_choice = self.filter_type.get()

        img = self.current_img
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        shape = img.shape

        ideal = IdealNotchFilter()
        butter = ButterworthNotchFilter()
        gauss = GaussianNotchFilter()

        H_ideal = np.ones(shape, dtype=np.float32)
        H_butter = np.ones(shape, dtype=np.float32)
        H_gauss = np.ones(shape, dtype=np.float32)

        for (u_k, v_k) in self.points:
            u_shift = u_k - shape[0] // 2
            v_shift = v_k - shape[1] // 2
            H_ideal *= ideal.apply_filter(shape, D0=D0, u_k=u_shift, v_k=v_shift)
            H_butter *= butter.apply_filter(shape, D0=D0, u_k=u_shift, v_k=v_shift, n=2)
            H_gauss *= gauss.apply_filter(shape, D0=D0, u_k=u_shift, v_k=v_shift)

        results = {}

        if filter_choice in ["Ideal", "All"]:
            f_ideal = fshift * H_ideal
            img_ideal = np.abs(np.fft.ifft2(np.fft.ifftshift(f_ideal)))
            results["Ideal"] = img_ideal

        if filter_choice in ["Butterworth", "All"]:
            f_butter = fshift * H_butter
            img_butter = np.abs(np.fft.ifft2(np.fft.ifftshift(f_butter)))
            results["Butterworth"] = img_butter

        if filter_choice in ["Gaussian", "All"]:
            f_gauss = fshift * H_gauss
            img_gauss = np.abs(np.fft.ifft2(np.fft.ifftshift(f_gauss)))
            results["Gaussian"] = img_gauss

        plt.figure(figsize=(4 * (len(results) + 1), 5))

        # Ảnh nhiễu (hiện tại)
        plt.subplot(1, len(results) + 1, 1)
        plt.imshow(img, cmap='gray')
        plt.title("Ảnh nhiễu")
        if hasattr(self, 'original_img'):
            mse = calculate_mse(self.original_img, img)
            snr = calculate_snr(self.original_img, img)
            plt.xlabel(f"MSE: {mse:.2f}\nSNR: {snr:.2f}", fontsize=9)

        # Các ảnh sau xử lý
        os.makedirs("result", exist_ok=True)
        for i, (name, result) in enumerate(results.items(), start=2):
            plt.subplot(1, len(results) + 1, i)
            plt.imshow(result, cmap='gray')
            plt.title(name)

            if hasattr(self, 'original_img'):
                mse = calculate_mse(self.original_img, result)
                snr = calculate_snr(self.original_img, result)
                plt.xlabel(f"MSE: {mse:.2f}\nSNR: {snr:.2f}", fontsize=9)
            result_img_uint8 = np.clip(result, 0, 255).astype(np.uint8)
            filename = os.path.basename(self.file_path)
            name_part = os.path.splitext(filename)[0]
            save_path = os.path.join("result", f"{name_part}_{name}.png")
            cv2.imwrite(save_path, result_img_uint8)

        plt.tight_layout()
        plt.show()
    
    def auto_process_image(self):
        if not hasattr(self, 'current_img'):
            messagebox.showwarning("Lỗi", "Vui lòng chọn ảnh trước!")
            return

        D0 = self.d0_slider.get()
        filter_choice = self.filter_type.get()

        img = self.current_img
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        shape = img.shape

        # Phát hiện điểm nhiễu tự động
        magnitude_spectrum = np.log(1 + np.abs(fshift))
        self.points = auto_detect_noise_points(magnitude_spectrum, threshold_ratio=0.7, min_distance=3, center_exclude_radius=30)
        messagebox.showinfo("Thông báo", f"Đã phát hiện {len(self.points)} điểm nhiễu tự động.")

        # Áp dụng filter giống như apply_filters
        ideal = IdealNotchFilter()
        butter = ButterworthNotchFilter()
        gauss = GaussianNotchFilter()

        H_ideal = np.ones(shape, dtype=np.float32)
        H_butter = np.ones(shape, dtype=np.float32)
        H_gauss = np.ones(shape, dtype=np.float32)

        for (u_k, v_k) in self.points:
            u_shift = u_k - shape[0] // 2
            v_shift = v_k - shape[1] // 2
            H_ideal *= ideal.apply_filter(shape, D0=D0, u_k=u_shift, v_k=v_shift)
            H_butter *= butter.apply_filter(shape, D0=D0, u_k=u_shift, v_k=v_shift, n=2)
            H_gauss *= gauss.apply_filter(shape, D0=D0, u_k=u_shift, v_k=v_shift)

        results = {}

        if filter_choice in ["Ideal", "All"]:
            f_ideal = fshift * H_ideal
            img_ideal = np.abs(np.fft.ifft2(np.fft.ifftshift(f_ideal)))
            results["Ideal"] = img_ideal

        if filter_choice in ["Butterworth", "All"]:
            f_butter = fshift * H_butter
            img_butter = np.abs(np.fft.ifft2(np.fft.ifftshift(f_butter)))
            results["Butterworth"] = img_butter

        if filter_choice in ["Gaussian", "All"]:
            f_gauss = fshift * H_gauss
            img_gauss = np.abs(np.fft.ifft2(np.fft.ifftshift(f_gauss)))
            results["Gaussian"] = img_gauss

        plt.figure(figsize=(4 * (len(results) + 1), 5))

        plt.subplot(1, len(results) + 1, 1)
        plt.imshow(img, cmap='gray')
        plt.title("Ảnh nhiễu")
        if hasattr(self, 'original_img'):
            mse = calculate_mse(self.original_img, img)
            snr = calculate_snr(self.original_img, img)
            plt.xlabel(f"MSE: {mse:.2f}\nSNR: {snr:.2f}", fontsize=9)

        os.makedirs("result", exist_ok=True)
        for i, (name, result) in enumerate(results.items(), start=2):
            plt.subplot(1, len(results) + 1, i)
            plt.imshow(result, cmap='gray')
            plt.title(name)

            if hasattr(self, 'original_img'):
                mse = calculate_mse(self.original_img, result)
                snr = calculate_snr(self.original_img, result)
                plt.xlabel(f"MSE: {mse:.2f}\nSNR: {snr:.2f}", fontsize=9)
            result_img_uint8 = np.clip(result, 0, 255).astype(np.uint8)
            filename = os.path.basename(self.file_path)
            name_part = os.path.splitext(filename)[0]
            save_path = os.path.join("result", f"{name_part}_{name}_auto.png")
            cv2.imwrite(save_path, result_img_uint8)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = Tk()
    app = NotchFilterApp(root)
    root.mainloop()
