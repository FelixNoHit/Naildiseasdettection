import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Định cấu hình trang web Streamlit
st.set_page_config(page_title="Nail Disease Diagnosis", page_icon="💅", layout="centered")

# Danh sách các nhãn bệnh — PHẢI khớp đúng thứ tự class_indices lúc train
# (lấy từ output train_generator.class_indices trong EffNet.ipynb)
CLASS_NAMES = [
    'Acral_Lentiginous_Melanoma',  # 0
    'Healthy_Nail',                # 1
    'Onychogryphosis',             # 2
    'blue_finger',                 # 3
    'clubbing',                    # 4
    'pitting',                     # 5
]

IMAGE_SIZE = (224, 224)

# Đường dẫn model — dùng đường dẫn TƯƠNG ĐỐI (đặt file .keras cùng thư mục với app.py)
# để chạy được cả local lẫn khi deploy online (Streamlit Cloud / HF Spaces).
MODEL_PATH = "final_efficientnet_model.keras"

# Hàm tải mô hình (dùng cache để không bị load lại mỗi khi chọn ảnh mới)
@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

# Khởi tạo mô hình
with st.spinner("Đang tải mô hình AI, vui lòng đợi..."):
    model = load_my_model()

# --- GIAO DIỆN ỨNG DỤNG ---
st.title("💅 Chẩn Đoán Bệnh Qua Móng Tay (AI)")
st.write("Tải lên hình ảnh móng tay của bạn để mô hình EfficientNetV2B0 phân tích và đưa ra dự đoán.")

uploaded_file = st.file_uploader("Chọn một bức ảnh rõ nét về móng tay...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Đọc và hiển thị ảnh lên giao diện
    image = Image.open(uploaded_file).convert("RGB")  # đảm bảo luôn là RGB (bỏ kênh alpha nếu có)
    st.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)

    st.write("🔄 Đang phân tích...")

    # 2. Tiền xử lý ảnh
    # QUAN TRỌNG: EfficientNetV2 có sẵn lớp chuẩn hoá (Rescaling/Normalization) bên trong
    # kiến trúc model, nên KHÔNG cần rescale (/255) hay trừ mean ImageNet bên ngoài như ResNet50.
    # Chỉ cần resize đúng kích thước lúc train và ép kiểu float32.
    img_resized = image.resize(IMAGE_SIZE)
    img_array = np.array(img_resized).astype("float32")
    img_batch = np.expand_dims(img_array, axis=0)

    # 3. Dự đoán kết quả
    predictions = model.predict(img_batch)[0]

    max_idx = np.argmax(predictions)
    predicted_label = CLASS_NAMES[max_idx]
    confidence = predictions[max_idx] * 100

    # 4. Hiển thị kết quả ra màn hình
    st.subheader("📊 Kết quả dự đoán:")

    if predicted_label == 'Healthy_Nail':
        st.success(f"Móng tay khỏe mạnh ({confidence:.2f}%)")
    else:
        st.error(f"Phát hiện dấu hiệu: **{predicted_label}** ({confidence:.2f}%)")
        st.warning("⚠️ *Lưu ý: Kết quả từ AI chỉ mang tính chất tham khảo, không thay thế cho chẩn đoán y khoa từ bác sĩ.*")

    # Hiển thị chi tiết xác suất dưới dạng thanh tiến trình (Progress bar)
    st.write("---")
    st.write("**Chi tiết tỷ lệ phân tích từ mô hình:**")
    for i, class_name in enumerate(CLASS_NAMES):
        prob = float(predictions[i])
        st.write(f"{class_name}: {prob*100:.2f}%")
        st.progress(prob)
