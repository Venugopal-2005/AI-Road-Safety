# Sample Road Scene Videos

This directory holds test video files for benchmarking and demonstration of the Road Safety and Pedestrian Detection System.

## Generating Synthetic Test Videos
To generate an immediate offline synthetic MP4 video demonstrating lane tracking, pedestrian hazard proximity, and vehicle movement:
```bash
python scripts/generate_samples.py
```
This produces `assets/sample_videos/sample_road_driving.mp4` locally without downloading external video files.

## Adding Custom Driving Video
You can test dashcam footage or competition videos:
- Supported formats: `.mp4`, `.avi`, `.mov`
- Recommended resolution: 1280x720 (720p) or 1920x1080 (1080p)
- CPU Optimization tip: In the Streamlit app sidebar, set **Frame Stride** to 2 or 3 to process at 2x-3x speed on standard laptops.
