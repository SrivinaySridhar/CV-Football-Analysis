# Football Analytics - CV Football Analysis

This project aims to analyze football matches using computer vision techniques. It leverages YOLOv8 for player detection, KMeans clustering for segmentation, and various computer vision methods to track players across frames. Key metrics such as possession rate and distance covered are extracted to provide comprehensive insights into team dynamics and individual 
player contributions.

## Project Demo

<video controls src="output_videos/output_video.mp4" title="Demo Output Video"></video>

## Key Features  
- **YOLOv8 for Object Detection**:  
  Trained a custom YOLOv8 model to detect and track players accurately in various game situations.  
- **KMeans Clustering for Player Segmentation**:  
  Used KMeans to precisely segment players from the background for better identification and analysis.  
- **Player Tracking Across Frames**:  
  Applied optical flow and other computer vision techniques to track player movement throughout the video.  
- **Performance Metrics Extraction**:  
  Calculated key metrics such as possession rate and distance covered to evaluate team performance.