import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Đường dẫn đến thư mục chứa dữ liệu đã được xử lý (ảnh mono)
data_dir = 'mono'

# Định nghĩa các nhãn (mệnh giá tiền)
denominations = ['10k', '50k', '500k']
name_to_label = {name: i for i, name in enumerate(denominations)}
NUM_CLASSES = len(denominations)
IMG_SIZE = 64

X = []
y = []

print("Đang tải dữ liệu...")
if not os.path.exists(data_dir):
    print(f"Lỗi: Không tìm thấy thư mục dữ liệu tại '{data_dir}'.")
    exit()

for denom in denominations:
    denom_path = os.path.join(data_dir, denom)
    label = name_to_label[denom]
    
    if not os.path.exists(denom_path):
        print(f"Lỗi: Không tìm thấy thư mục '{denom_path}'.")
        continue

    for filename in os.listdir(denom_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(denom_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(label)
            else:
                print(f"Cảnh báo: Không thể đọc file ảnh: {img_path}")

print(f"Đã tải được {len(X)} ảnh.")

if len(X) == 0:
    print("Lỗi: Không có ảnh nào được tải. Vui lòng kiểm tra lại đường dẫn và tên thư mục.")
    exit()

# Tiền xử lý dữ liệu cho mô hình
X = np.array(X).reshape((len(X), IMG_SIZE * IMG_SIZE)).astype('float32') / 255.0
y = to_categorical(np.array(y), num_classes=NUM_CLASSES)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình ANN
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(IMG_SIZE * IMG_SIZE,)))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
print("\nBắt đầu huấn luyện mô hình...")
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Lưu mô hình
model_save_path = 'money_recognition_model.h5'
model.save(model_save_path)
print(f"\n✅ Mô hình đã được lưu thành công vào '{model_save_path}'")