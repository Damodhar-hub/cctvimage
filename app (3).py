import streamlit as st
import os
import shutil
import cv2
import numpy as np
import pandas as pd
from datetime import timedelta
from ultralytics import YOLO
from PIL import Image
import easyocr

def main():
    st.set_page_config(page_title="Enhanced CCTV Analysis", layout="wide")
    st.title("🚀 Enhanced CCTV Analysis Tool")

    # Sidebar settings
    st.sidebar.header("Detection Options")
    detect_persons = st.sidebar.checkbox("Detect Persons & Kids", value=True)
    detect_vehicles = st.sidebar.checkbox("Detect Vehicles", value=True)

    st.sidebar.header("Model and Processing")
    model_size = st.sidebar.selectbox(
        "YOLO Model Size", 
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        index=3
    )
    confidence = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, step=0.05)
    iou_thresh = st.sidebar.slider("IoU Threshold", 0.1, 0.9, 0.45, step=0.05)
    frame_interval = st.sidebar.slider("Process every N frames", 1, 15, 5)
    upscale_factor = st.sidebar.selectbox("Upscale Factor", [1,2,3,4], index=1)

    st.sidebar.header("Output Options")
    show_previews = st.sidebar.checkbox("Show Image Previews", value=True)

    # Load models
    @st.cache_resource
    def load_models():
        yolo = YOLO(model_size)
        reader = easyocr.Reader(["en"], gpu=False)
        return yolo, reader

    yolo, reader = load_models()

    uploaded_file = st.file_uploader("Upload CCTV Video", type=["mp4","avi","mov"])
    if not uploaded_file:
        st.info("Please upload a CCTV video to begin analysis.")
        return

    # Save uploaded video to temp file
    temp_video = "input_video" + os.path.splitext(uploaded_file.name)[1]
    with open(temp_video, "wb") as f:
        f.write(uploaded_file.read())

    # Prepare output directories
    output_dir = "cctv_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    categories = ["Persons","Kids","Vehicles","NumberPlates"]
    for cat in categories:
        os.makedirs(os.path.join(output_dir, cat), exist_ok=True)

    # Video capture
    cap = cv2.VideoCapture(temp_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logs = []
    previews = []
    frame_idx = 0
    progress = st.progress(0)
    status_text = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue
        ts = frame_idx / fps
        ts_str = str(timedelta(seconds=int(ts))).replace(":","-")

        # Detection
        results = yolo(frame, conf=confidence, iou=iou_thresh)
        for r in results:
            for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
                name = yolo.model.names[int(cls_id)]
                if name == 'person' and detect_persons:
                    y1,y2 = int(box[1]), int(box[3])
                    cat = 'Kids' if (y2-y1) < 100 else 'Persons'
                elif name in ['car','truck','bus','motorcycle','bicycle'] and detect_vehicles:
                    cat = 'Vehicles'
                else:
                    continue
                x1,y1,x2,y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                up = cv2.resize(crop, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)
                img = Image.fromarray(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))

                fname = f"{ts_str}_{cat}_{frame_idx}.png"
                path = os.path.join(output_dir, cat, fname)
                img.save(path)

                logs.append({
                    "time": ts_str,
                    "category": cat,
                    "filename": fname,
                    "plate_text": ""
                })
                if show_previews and len(previews) < 10:
                    previews.append(up)

        # Plate OCR
        if detect_vehicles:
            for r in results:
                if int(r.boxes.cls[0]) in [2,5,7]:
                    x1,y1,x2,y2 = map(int, r.boxes.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    up = cv2.resize(crop, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)
                    txts = reader.readtext(up)
                    plate_txt = " ".join([t[1] for t in txts])

                    fname = f"{ts_str}_NumberPlate_{frame_idx}.png"
                    path = os.path.join(output_dir, 'NumberPlates', fname)
                    Image.fromarray(cv2.cvtColor(up, cv2.COLOR_BGR2RGB)).save(path)

                    logs.append({
                        "time": ts_str,
                        "category": 'NumberPlates',
                        "filename": fname,
                        "plate_text": plate_txt
                    })
                    if show_previews and len(previews) < 10:
                        previews.append(up)

        progress.progress(min(frame_idx/total_frames, 1.0))
        status_text.text(f"Processed frame {frame_idx}/{total_frames}")

    cap.release()
    st.success("✅ Processing complete!")

    if show_previews and previews:
        st.subheader("Sample Previews")
        st.image(previews, width=150)

    df = pd.DataFrame(logs)
    st.subheader("Detection Summary")
    st.write(df.groupby('category').size().rename('count'))

    csv = df.to_csv(index=False).encode()
    st.download_button("Download Report CSV", data=csv, file_name="report.csv")

    shutil.make_archive('cctv_output', 'zip', output_dir)
    with open('cctv_output.zip', 'rb') as f:
        st.download_button("Download All Outputs", data=f, file_name='cctv_output.zip')

if __name__ == '__main__':
    main()
