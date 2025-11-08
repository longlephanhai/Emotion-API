from fastapi import FastAPI, HTTPException, File, UploadFile
from transformers import pipeline
import uvicorn
import tempfile
import shutil
import os
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="🎧 Emotion Detection API",
    description="Phân tích cảm xúc từ file âm thanh (.wav) và chuyển thành text",
    version="1.3.1",
)

stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", language="en")

classifier = pipeline("audio-classification", model="superb/hubert-base-superb-er")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/detect_emotion")
async def detect_emotion(audio: UploadFile = File(...)):
    if not audio.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .wav")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_file = tmp.name
            shutil.copyfileobj(audio.file, tmp)

        logging.info(f"Đang xử lý file: {tmp_file}")

        # Speech-to-Text
        stt_result = stt_pipeline(tmp_file)
        user_text = stt_result.get("text", "")

        # Emotion detection
        result = classifier(tmp_file)
        if not result:
            raise HTTPException(status_code=500, detail="Không nhận diện được cảm xúc từ audio")

        emotion = result[0]["label"]
        confidence = round(result[0]["score"], 4)

        return {"text": user_text, "emotion": emotion, "confidence": confidence}

    except Exception as e:
        logging.error(f"Lỗi xử lý audio: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý audio: {e}")

    finally:
        if 'tmp_file' in locals() and os.path.exists(tmp_file):
            os.remove(tmp_file)
            logging.info(f"Đã xóa file tạm: {tmp_file}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
