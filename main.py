from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import cv2
import numpy as np

app = FastAPI()

@app.get("/")
def home():
    return {"status": "API is running 100% Free!"}

@app.post("/cleanup")
async def cleanup(image_file: UploadFile = File(...), mask_file: UploadFile = File(...)):
    # অ্যাপ থেকে পাঠানো মূল ছবি রিড করা
    img_bytes = await image_file.read()
    img_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    # অ্যাপ থেকে পাঠানো সাদাকালো মাস্ক রিড করা
    mask_bytes = await mask_file.read()
    mask_arr = np.frombuffer(mask_bytes, np.uint8)
    mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE) # মাস্ক সবসময় সাদাকালো হতে হয়

    # OpenCV দিয়ে ম্যাজিক রিমুভ (Inpainting) লজিক
    # 3 হলো ব্লার রেডিয়াস, আপনি চাইলে বাড়াতে/কমাতে পারেন
    result = cv2.inpaint(img, mask, 15, cv2.INPAINT_NS)

    # রেজাল্ট ছবিটিকে PNG তে কনভার্ট করে অ্যাপে ফেরত পাঠানো
    _, encoded_img = cv2.imencode('.png', result)
    return Response(content=encoded_img.tobytes(), media_type="image/png")
