from utils import read_video, save_video
from trackers import Tracker

def main():
    # Read the video
    input_video_path = "input_videos/08fd33_4.mp4"
    video_frames = read_video(input_video_path)

    # Initialize the tracker
    tracker = Tracker(model_path = "models/best.pt")
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path="stubs/track_stubs.pkl")

    # Save the video
    output_video_path = "output_videos/output_video.avi"
    save_video(video_frames, output_video_path)

if __name__ == "__main__":
    main()