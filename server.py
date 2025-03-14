import uvicorn
import numpy as np
import tensorflow.lite as tflite
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os
import uuid
from fastapi.responses import FileResponse

app = FastAPI()

# Load model TFLite
interpreter = tflite.Interpreter(model_path="mobilenetv2_batik.tflite")
interpreter.allocate_tensors()

# Ambil input dan output model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Nama-nama kelas batik
class_names = ["Bangkalan", "Pamekasan", "Sampang", "Sumenep"] 

# Buat folder untuk menyimpan gambar jika belum ada
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Fungsi prediksi gambar
def predict_image(img):
    img = img.resize((224, 224))  
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Masukkan ke model
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # Ambil hasil prediksi
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    
    return class_names[predicted_class], float(np.max(output_data))

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Gunakan UUID untuk nama unik, tetapi tetap simpan ekstensi asli
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    # Simpan gambar ke folder
    img.save(file_path)

    label, confidence = predict_image(img)

    # Buat URL untuk mengakses gambar
    image_url = f"http://localhost:8000/images/{unique_filename}"

    return {
        "class": label,
        "confidence": confidence,
        "image_url": image_url
    }

# Endpoint untuk mengakses gambar
@app.get("/images/{filename}")
async def get_image(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    return FileResponse(file_path)

# Sajikan folder gambar sebagai static files
app.mount("/images", StaticFiles(directory=UPLOAD_FOLDER), name="images")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)

