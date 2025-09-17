import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import os
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk

model_name = 'money_recognition_model.h5'
if not os.path.exists(model_name):
    print(f"Lỗi: Không tìm thấy file mô hình '{model_name}'. Vui lòng huấn luyện mô hình trước.")
    exit()

loaded_model = load_model(model_name)
print("Đã nạp mô hình thành công.")

denominations = ['10k', '50k', '500k']
label_to_name = {i: name for i, name in enumerate(denominations)}
IMG_SIZE = 64

def preprocess_image(image_path, target_size=IMG_SIZE):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn {image_path}.")
        return None, None

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_resized = cv2.resize(img_gray, (target_size, target_size))
    
    img_processed = img_resized.reshape(1, target_size * target_size).astype('float32') / 255.0
    
    return img_processed, img


def select_image():
    file_path = filedialog.askopenfilename(
        title="Chọn một file ảnh",
        filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
    )
    if file_path:
        predict_and_display(file_path)

# Dự đoán
def predict_and_display(img_path):
    img_processed, original_img = preprocess_image(img_path)
    if img_processed is not None:
        preds = loaded_model.predict(img_processed)
        predicted_label = np.argmax(preds)
        predicted_name = label_to_name[predicted_label]
        confidence = np.max(preds) * 100

        img_pil = Image.open(img_path)
        img_resized = img_pil.resize((300, 300), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_resized)
        
        image_label.config(image=photo)
        image_label.image = photo

        result_text = f"Dự đoán: {predicted_name}\nĐộ tin cậy: {confidence:.2f}%"
        result_label.config(text=result_text)

# giao diện
root = Tk()
root.title("Nhận diện tiền Việt Nam")
root.geometry("400x500")

image_label = Label(root, text="Ảnh sẽ hiển thị ở đây", width=40, height=20, bg="lightgrey")
image_label.pack(pady=20)

result_label = Label(root, text="Chưa có dự đoán", font=("Helvetica", 16))
result_label.pack(pady=10)

button = Button(root, text="Chọn Ảnh", command=select_image)
button.pack(pady=10)

root.mainloop()
