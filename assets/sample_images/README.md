# Sample Road Scene Images

This directory holds sample test images for verifying the Road Safety and Pedestrian Detection System.

## Included / Generated Samples
- `highway_lanes.jpg`: High-contrast highway scene with left and right lane markings.
- `urban_pedestrian.jpg`: Urban intersection scene containing pedestrians and vehicles.
- `night_rain_curved.jpg`: Challenging visibility test frame with wet pavement.

## Generating Synthetic Road Samples
You can generate test assets immediately using the provided utility:
```bash
python scripts/generate_samples.py
```

## Adding Your Own Images
Place any `.jpg`, `.jpeg`, or `.png` images into this directory. The Streamlit application will automatically display them in the demo presets dropdown.
