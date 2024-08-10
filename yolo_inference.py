from ultralytics import YOLO

model = YOLO('models/best.pt')

video_path = "input_videos/08fd33_4.mp4"

results = model.predict(video_path, save = True)

print(results[0])

print("==================================================")

for box in results[0].boxes:
    print(box)