from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20 
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            confidence = 0.1 # Confidence threshold to filter out weak detections
            batch_detections = self.model.predict(batch_frames, conf = confidence)
            detections += batch_detections
            break # Run on only the first batch for now, remove later
        return detections

    def get_object_tracks(self, frames):
        # Returns the list of object tracks for each frame
        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            class_names = detection.names # {0: person, 1: ball...}
            class_names_inv = {v:k for k, v in class_names.items()} # {person: 0, ball: 1...}
            print(class_names)

            # Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player object 
            for object_id, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_id] = class_names_inv["player"]

            print(detection_supervision)

            break
