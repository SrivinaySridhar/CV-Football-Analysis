# Football Analytics - CV Football Analysis

This project leverages state-of-the-art computer vision techniques to analyze football team performance from video input. Using YOLO, an advanced object detection model, it identifies players, referees, and footballs in each frame. Custom object detectors are trained to enhance detection accuracy, and tracking algorithms are employed to follow these objects across frames.

Players are assigned to teams based on the color of their jerseys using KMeans clustering for pixel segmentation. Optical flow is applied to measure camera movement, ensuring accurate player motion tracking. Perspective transformation is used to convert player movement from pixels to real-world units, allowing precise calculation of distances covered and speeds achieved during the game.

## Project Demo

https://github.com/user-attachments/assets/d278ae55-0751-4778-bc97-b39d9aea1cd0

## Key Features  
- **YOLOv8 for Object Detection**:  
  Trained a custom YOLOv8 model to detect and track players accurately in various game situations.  
- **KMeans Clustering for Player Segmentation**:  
  Used KMeans to precisely segment players from the background for better identification and analysis.  
- **Player Tracking Across Frames**:  
  Applied optical flow and other computer vision techniques to track player movement throughout the video.  
- **Performance Metrics Extraction**:  
  Calculated key metrics such as possession rate and distance covered to evaluate team performance.

## Key Metrics Extracted
- **Possession Rate**
- **Distance Covered**
- **Player Speed**

This combination of advanced detection, tracking, and transformation techniques provides a comprehensive analysis of player and team performance in real-world metrics.