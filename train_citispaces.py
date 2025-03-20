import numpy as np
import matplotlib.pyplot as plt
from skimage import io, segmentation, feature, future, color, transform
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import os
from tqdm import tqdm
import time

# Đường dẫn ảnh và nhãn
img_path_folder = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\segmetion_detect\train\img"
label_path_folder = r"F:\StudyatCLass\Study\class\Thigiacmaytinh\segmetion_detect\train\label"

# Lấy danh sách file ảnh và nhãn
img_files = sorted(os.listdir(img_path_folder))
label_files = sorted(os.listdir(label_path_folder))

# Khởi tạo danh sách dữ liệu
images, labels = [], []

# Bắt đầu đếm thời gian
start_time = time.time()

# Load dữ liệu với thanh tiến trình
print("Loading images and labels...")
for img_file, label_file in tqdm(zip(img_files, label_files), total=len(img_files)):
    img = io.imread(os.path.join(img_path_folder, img_file))
    label_img = io.imread(os.path.join(label_path_folder, label_file))

    # Resize về kích thước chuẩn (96, 256)
    img = transform.resize(img, (96, 256), anti_aliasing=True)
    label_img = transform.resize(label_img, (96, 256), anti_aliasing=True)

    # Chuyển ảnh nhãn thành grayscale nếu cần
    if len(label_img.shape) == 3:
        label_img = color.rgb2gray(label_img)
        label_img = (label_img * 25).astype(np.uint8)

    images.append(img)
    labels.append(label_img)

# Trích xuất đặc trưng với thanh tiến trình
sigma_min, sigma_max = 1, 16
features_func = partial(
    feature.multiscale_basic_features,
    intensity=True, edges=False, texture=True,
    sigma_min=sigma_min, sigma_max=sigma_max, channel_axis=-1
)

print("Extracting features...")
features_list = []
for img in tqdm(images, total=len(images)):
    features = features_func(img)
    features_list.append(features)

# Chuyển danh sách thành numpy array
X_train = np.concatenate(features_list, axis=0)
y_train = np.concatenate([label.flatten() for label in labels])

# Huấn luyện mô hình với thanh tiến trình
print("Training model...")
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)
clf.fit(X_train, y_train)

# Hiển thị thời gian chạy
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# Dự đoán và hiển thị kết quả
def visualize_prediction(image_idx):
    img = images[image_idx]
    features = features_func(img)
    result = clf.predict(features.reshape(-1, features.shape[-1])).reshape(96, 256)

    fig, ax = plt.subplots(1, 3, figsize=(9, 4))
    ax[0].imshow(segmentation.mark_boundaries(img, result, mode='thick'))
    ax[0].set_title('Ảnh gốc & biên phân đoạn')
    ax[1].imshow(result)
    ax[1].set_title('Kết quả phân đoạn')
    ax[2].imshow(labels[image_idx])
    ax[2].set_title('Nhãn gốc')
    fig.tight_layout()
    plt.show()

# Nhập số thứ tự ảnh để xem kết quả
while True:
    try:
        idx = int(input(f"Nhập số thứ tự ảnh (0-{len(images)-1}) hoặc -1 để thoát: "))
        if idx == -1:
            break
        if 0 <= idx < len(images):
            visualize_prediction(idx)
        else:
            print("Số không hợp lệ, hãy nhập lại!")
    except ValueError:
        print("Hãy nhập một số nguyên hợp lệ!")
