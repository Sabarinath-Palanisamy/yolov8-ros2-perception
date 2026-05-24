import cv2
from ultralytics import YOLO

# 1. Load your custom-trained model weights
# Ensure this path matches exactly where your 'best.pt' was saved
model_path = "runs/segment/train/weights/best.pt"
model = YOLO(model_path)

# 2. Define the input source
# Use 0 for your laptop's integrated webcam, or replace with a path to a roadwork video file
# video_source = "sample_roadwork.mp4" 
video_source = 0 

print("🚀 Initializing YOLOv8 ByteTrack Pipeline...")

# 3. Execute the tracking loop
# The tracker="bytetrack.yaml" argument tells YOLO to use the advanced tracker
results = model.track(
    source=video_source,
    show=True,              # Opens a window to show the live feed
    tracker="bytetrack.yaml", 
    conf=0.5,               # Only show predictions with >50% confidence
    stream=True             # Stream mode keeps memory usage low for video feeds
)

# Iterate through the frames as they are processed
for frame_result in results:
    # This loop keeps the video window open and processing
    # Press 'q' on your keyboard to safely close the window
    pass
