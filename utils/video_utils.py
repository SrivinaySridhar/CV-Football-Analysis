import cv2

def read_video(video_path):
    # Returns the list of frames for the video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    # Saves the list of frames as a video
    width, height = output_video_frames[0].shape[1], output_video_frames[0].shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()