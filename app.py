import numpy as np
from PIL import Image
import streamlit as st
# Import interpreter từ thư viện ai-edge-litert thay vì tensorflow
from ai_edge_litert.interpreter import Interpreter

# Định cấu hình trang web Streamlit
st.set_page_config(
    page_title="Nail Disease Diagnosis", page_icon="💅", layout="centered"
)

# Danh sách các nhãn bệnh — PHẢI khớp đúng thứ tự class_indices lúc train
CLASS_NAMES = [
    "Acral_Lentiginous_Melanoma",  # 0
    "Healthy_Nail",  # 1
    "Onychogryphosis",  # 2
    "blue_finger",  # 3
    "clubbing",  # 4
    "pitting",  # 5
]

IMAGE_SIZE = (224, 224)
MODEL_PATH = "final_efficientnet_model.tflite"


# Hàm tải mô hình LiteRT (dùng cache để tránh load lại mỗi lượt tương tác)
@st.cache_resource
def load_tflite_model():
  interpreter = Interpreter(model_path=MODEL_PATH)
  interpreter.allocate_tensors()
  return interpreter


# Khởi tạo mô hình
try:
  with st.spinner("Đang tải mô hình AI, vui lòng đợi..."):
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
  st.error(
      f"❌ Không thể tải file mô hình `{MODEL_PATH}`. Vui lòng kiểm tra lại"
      " đường dẫn file trên GitHub."
  )
  st.stop()

# --- GIAO DIỆN ỨNG DỤNG ---
st.title("💅 Chẩn Đoán Bệnh Qua Móng Tay (AI)")
st.write(
    "Tải lên hình ảnh móng tay của bạn để mô hình EfficientNetV2B0 phân tích và"
    " đưa ra dự đoán."
)

uploaded_file = st.file_uploader(
    "Chọn một bức ảnh rõ nét về móng tay...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  # 1. Đọc và hiển thị ảnh
  image = Image.open(uploaded_file).convert("RGB")
  st.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)

  st.write("🔄 Đang phân tích...")

  # 2. Tiền xử lý ảnh
  img_resized = image.resize(IMAGE_SIZE)
  img_array = np.array(img_resized).astype("float32")
  img_batch = np.expand_dims(img_array, axis=0)

  # 3. Dự đoán kết quả bằng LiteRT Interpreter
  interpreter.set_tensor(input_details[0]["index"], img_batch)
  interpreter.invoke()
  predictions = interpreter.get_tensor(output_details[0]["index"])[0]

  max_idx = np.argmax(predictions)
  predicted_label = CLASS_NAMES[max_idx]
  confidence = predictions[max_idx] * 100

  # 4. Hiển thị kết quả
  st.subheader("📊 Kết quả dự đoán:")

  if predicted_label == "Healthy_Nail":
    st.success(f"Móng tay khỏe mạnh ({confidence:.2f}%)")
  else:
    st.error(f"Phát hiện dấu hiệu: **{predicted_label}** ({confidence:.2f}%)")
    st.warning(
        "⚠️ *Lưu ý: Kết quả từ AI chỉ mang tính chất tham khảo, không thay thế"
        " cho chẩn đoán y khoa từ bác sĩ.*"
    )

  # Hiển thị chi tiết xác suất
  st.write("---")
  st.write("**Chi tiết tỷ lệ phân tích từ mô hình:**")
  for i, class_name in enumerate(CLASS_NAMES):
    prob = float(predictions[i])
    st.write(f"{class_name}: {prob*100:.2f}%")
    st.progress(min(prob, 1.0))
