import os
import uuid
import torch
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from ultralytics import YOLO
from transformers import VitsModel, AutoTokenizer as TtsTokenizer
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO models
model_official = YOLO("yolov8n.pt")
model_custom = YOLO("best.pt")

# Load TTS model
tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)
tts_tokenizer = TtsTokenizer.from_pretrained("facebook/mms-tts-eng")

# FastAPI app
app = FastAPI()

allowed_origins = [
    "http://localhost:5173",          # React dev server
    "https://your-frontend-domain.com" # Replace with your deployed frontend URL
]
# Enable CORS (frontend can call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ restrict in production to your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory for temporary files
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Serve static files
app.mount("/files", StaticFiles(directory=TEMP_DIR), name="files")

# ==== Helper functions ====
object_real_height = {
    "person": 1.7,
    "bicycle": 1.0,
    "car": 1.5,
    "motorcycle": 1.2,
    "bus": 3.0,
    "truck": 3.0,
    "dog": 0.5,
    "cat": 0.3,
    "bench": 1.0,
}
object_real_width = {
    "person": 0.5,
    "bicycle": 0.6,
    "car": 1.8,
    "motorcycle": 0.7,
    "bus": 2.5,
    "truck": 2.5,
    "dog": 0.3,
    "cat": 0.2,
    "bench": 0.5,
}

def get_narration(detected_objects):
    """
    Convert detected object labels to short narrations.
    """
    object_narrations = {
        "person": "There is a person ahead. Please stay alert.",
        "bicycle": "There is a bicycle nearby. Be careful and give way.",
        "car": "There are cars coming on the road. Do not cross the road now.",
        "motorcycle": "There are motorcycles nearby. Stay cautious.",
        "bus": "There is a bus approaching. Please wait safely.",
        "truck": "There is a truck nearby. Keep a safe distance.",
        "traffic light": "There is a traffic light ahead. Wait for the green signal before crossing.",
        "stop sign": "There is a stop sign ahead. Please stop and look around before moving.",
        "zebra crossing": "There is a zebra crossing in front of you. You can cross the road safely here.",
        "fire hydrant": "There is a fire hydrant nearby. Watch your step.",
        "bench": "There is a bench nearby. You may sit if needed.",
        "parking meter": "There is a parking meter close by.",
        "bird": "There are birds ahead. Stay calm and keep safe.",
        "cat": "There is a cat nearby. Please avoid sudden movements.",
        "dog": "There is a dog nearby. Be cautious and quiet.",
        "truck": "There is a truck approaching. Keep a safe distance.",
        "traffic cone": "There is a traffic cone ahead. Take extra caution.",
        "construction barrier": "There is a construction barrier nearby. Avoid this area.",
        "stop sign": "There is a stop sign ahead. Please stop and check before proceeding.",
        "fire extinguisher": "There is a fire extinguisher close by.",
        "mailbox": "There is a mailbox nearby.",
        "potted plant": "There is a potted plant near you. Watch your path.",
        "sidewalk": "You are near the sidewalk. Stay on it for safety.",
        "crosswalk signal": "There is a crosswalk signal ahead. Follow its directions.",
        "road work sign": "There is a road work sign nearby. Be very cautious.",
        "barrier": "There is a barrier ahead. Please avoid walking into it.",
        "wheelchair": "There is a wheelchair nearby. Give way and be respectful.",
        "stroller": "There is a stroller nearby. Be careful around it.",
        "traffic sign": "There is a traffic sign nearby. Follow the traffic instructions.",
    }
    narrations = [object_narrations.get(obj.lower(), None) for obj in detected_objects]
    narrations = [n for n in narrations if n]
    if not narrations:
        return "Detected objects: " + ", ".join(detected_objects) + ". Please be careful."
    return " ".join(narrations)

def tts_to_audiosegment(text):
    """
    Convert text to audio segment using VITS TTS.
    """
    inputs = tts_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = tts_model(**inputs)
    wav_array = output.waveform[0].cpu().numpy()
    sampling_rate = tts_model.config.sampling_rate or 16000
    wav_int16 = (wav_array * 32767).astype(np.int16)
    return AudioSegment(
        wav_int16.tobytes(),
        frame_rate=sampling_rate,
        sample_width=wav_int16.dtype.itemsize,
        channels=1
    )

def draw_boxes_with_distance(frame, results, focal_length=800):
    min_box_size = 10  # Ignore very small boxes (likely noise)
    for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = box.astype(int)
        label = results.names[int(cls)]

        H_image = y2 - y1
        W_image = x2 - x1

        H_real = object_real_height.get(label.lower(), None)
        W_real = object_real_width.get(label.lower(), None)

        distance_h = distance_w = None
        confidence = "approx."

        # Height-based distance
        if H_real and H_image > min_box_size:
            distance_h = (H_real * focal_length) / H_image

        # Width-based distance
        if W_real and W_image > min_box_size:
            distance_w = (W_real * focal_length) / W_image

        # Combine estimates if both available
        if distance_h and distance_w:
            distance_m = (distance_h + distance_w) / 2
            confidence = "high"
        elif distance_h:
            distance_m = distance_h
            confidence = "med"
        elif distance_w:
            distance_m = distance_w
            confidence = "med"
        else:
            distance_m = None

        # Choose color based on distance
        if distance_m:
            if distance_m < 3:
                color = (0, 0, 255)        # Red for close
            elif distance_m < 7:
                color = (0, 255, 0)        # Green for medium
            else:
                color = (255, 0, 0)        # Blue for far
            distance_text = f"{label}: {distance_m:.1f}m ({confidence})"
        else:
            distance_text = label + " (dist. unknown)"
            color = (0, 255, 255)          # Cyan if distance unknown

        # Draw rectangle and text
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
        cv2.putText(frame, distance_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 4)
    return frame

def process_video_with_audio(input_path, frame_skip=5):
    """
    Main pipeline:
    1. Detect objects using YOLO
    2. Generate narration for detected objects
    3. Overlay bounding boxes on video
    4. Merge TTS audio with video
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_video_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_video.mp4")
    out = cv2.VideoWriter(temp_video_path, fourcc, fps / frame_skip, (width, height))

    combined_audio = AudioSegment.silent(duration=0)
    last_narration = ""
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_skip == 0:
            # YOLO detections
            results_official = model_official(frame)[0]
            results_custom = model_custom(frame)[0]

            labels_official = [results_official.names[int(c)] for c in results_official.boxes.cls.cpu().numpy().astype(int)]
            labels_custom = [results_custom.names[int(c)] for c in results_custom.boxes.cls.cpu().numpy().astype(int)]
            all_labels = list(set(labels_official + labels_custom))

            # Draw boxes
            frame = draw_boxes_with_distance(frame, results_official)
            frame = draw_boxes_with_distance(frame, results_custom)

            # Generate TTS narration
            if all_labels:                                                   
                narration = get_narration(all_labels)
                if narration != last_narration:
                    audio_seg = tts_to_audiosegment(narration)
                    combined_audio += audio_seg + AudioSegment.silent(duration=500)
                    last_narration = narration

            # Write same frame multiple times to maintain frame rate
            for _ in range(frame_skip):
                out.write(frame)

        frame_id += 1

    cap.release()
    out.release()

    # Save audio and merge with video
    narration_audio_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_narration.mp3")
    combined_audio.export(narration_audio_path, format="mp3")

    # Merge audio + video with length checks
    final_video_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_with_narration.mp4")
    video_clip = VideoFileClip(temp_video_path)
    audio_clip = AudioFileClip(narration_audio_path)

    video_duration = video_clip.duration
    audio_duration = audio_clip.duration

    # Trim audio if longer than video
    if audio_duration > video_duration:
        audio_clip = audio_clip.subclip(0, video_duration)
    # Optional: pad audio if shorter (not mandatory, depends on your use case)

    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac")
    combined_audio.export(narration_audio_path, format="mp3")
    import time
    time.sleep(0.5)

    video_clip.close()
    audio_clip.close()
    final_clip.close()
    os.remove(temp_video_path)

    return final_video_path, narration_audio_path

# ==== API endpoint ====

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    """
    Upload a video -> detect objects -> generate narration -> return processed video/audio URLs
    """
    # Save uploaded video
    input_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Process video
    output_video, output_audio = process_video_with_audio(input_path, frame_skip=5)

    return JSONResponse({
        "video_url": f"http://localhost:8000/files/{os.path.basename(output_video)}",
        "audio_url": f"http://localhost:8000/files/{os.path.basename(output_audio)}",
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)