from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import sys
sys.path.append('../')
from utils import get_bbox_width, get_center_of_bbox

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
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # Returns the list of object tracks for each frame
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
        
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
            
            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if class_id == class_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == class_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
        
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3]) #Bottom of bounding box
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw the ellipse
        cv2.ellipse(
            frame, 
            center = (x_center, y2), 
            axes = (int(width), int(0.35*width)),
            angle = 0.0,
            startAngle = -45, 
            endAngle = 235,
            color = color,
            thickness = 2,
            lineType = cv2.LINE_4
        )
        
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                pt1 = (int(x1_rect), int(y1_rect)),
                pt2 = (int(x2_rect), int(y2_rect)),
                color = color,
                thickness = cv2.FILLED
            )
            
            x1_text = x1_rect + 12
            
            if track_id > 9 and track_id < 100:
                x1_text -= 5
            
            if track_id > 99:
                x1_text -= 10
            

            cv2.putText(
                frame,
                text = str(track_id),
                org = (int(x1_text), int(y1_rect + 15)),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.6,
                color = (0, 0, 0),
                thickness = 2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1]) #This is the top of the ball
        x_center, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x_center, y],
            [x_center - 10, y - 20],
            [x_center + 10, y - 20]
        ])

        # Draw the triangle
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED) 
        # Draw the border of the triangle
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame
    
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            
            # Draw Players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)
            
            # Draw Referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            output_video_frames.append(frame)

            # Draw Ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

        return output_video_frames