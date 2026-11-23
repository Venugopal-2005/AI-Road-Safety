# Testing & Verification Guide: AI-Road-Safety System

**OpenCV AI Competition 2026 powered by AWS**

---

## 1. Testing Philosophy

The test suite is built on three core engineering principles:
1. **Hermetic & Deterministic**: Basic unit tests do not require external internet access, live camera hardware, or GPU drivers.
2. **Synthetic Visual Fixtures**: Rather than bundling megabytes of proprietary road video in the git repository, unit tests employ programmatic synthetic numpy frames with known mathematical line slopes, horizon lines, and bounding coordinates.
3. **Graceful Degradation Under Faults**: Tests explicitly verify that blank frames, corrupt files, missing weights, and out-of-boundary deviations degrade safely without crashing the host process.

---

## 2. Test Suite Structure

The automated test suite resides in the `tests/` directory and contains 18 unit tests plus 1 end-to-end smoke test script:

```
tests/
├── test_config.py       # Configuration loading, YAML parsing, and hardware device resolution
├── test_lanes.py        # OpenCV lane detection, Hough transforms, slope filtering, and LDW
├── test_safety.py       # Heuristic hazard categorization (SAFE, CAUTION, CRITICAL)
├── test_validation.py   # Input integrity checks for frames, image files, and video containers
└── smoke_test.py        # End-to-end integration test running image and video pipelines
```

### Detailed Test Coverage

| Test Module | Test Case | Target Behavior |
| :--- | :--- | :--- |
| `test_config.py` | `test_default_config_instantiation` | Verifies default threshold boundaries and dataclass structures. |
| `test_config.py` | `test_resolved_device` | Ensures automatic fallback to CPU when CUDA is not present. |
| `test_config.py` | `test_yaml_loading` | Tests custom YAML override parsing via temporary fixture. |
| `test_config.py` | `test_yaml_missing_file_fallback` | Validates fallback to defaults when configuration path is invalid. |
| `test_lanes.py` | `test_lane_detection_on_synthetic_frame` | Verifies left/right line fitting and corridor polygon on synthetic road. |
| `test_lanes.py` | `test_lane_departure_warning_trigger` | Asserts LDW trigger when vehicle drifts right beyond threshold. |
| `test_lanes.py` | `test_graceful_degradation_on_blank_frame` | Checks that unpainted/blank roads return `NO_LANES` without error. |
| `test_lanes.py` | `test_invalid_and_empty_frames` | Validates handling of `None` and zero-dimension input arrays. |
| `test_safety.py` | `test_clear_path_safe` | Confirms empty scene evaluates to `SAFE` with clear path message. |
| `test_safety.py` | `test_pedestrian_critical_in_path` | Asserts pedestrian centered in forward path triggers `CRITICAL_WARNING`. |
| `test_safety.py` | `test_pedestrian_caution_distance` | Checks pedestrian near roadway triggers `CAUTION`. |
| `test_safety.py` | `test_vehicle_critical_proximity` | Confirms lead vehicle with large proximity ratio triggers `CRITICAL`. |
| `test_safety.py` | `test_lane_departure_alert_integration` | Validates integration of lane departure into hazard assessment. |
| `test_safety.py` | `test_traffic_sign_alert_integration` | Asserts stop sign presence adds actionable driver notice. |
| `test_validation.py` | `test_validate_frame_valid` | Validates standard 2D and 3D uint8 image arrays. |
| `test_validation.py` | `test_validate_frame_invalid_inputs` | Asserts rejection of None, wrong types, empty arrays, and tiny shapes. |
| `test_validation.py` | `test_validate_image_file` | Checks detection of nonexistent, unsupported, and empty image files. |
| `test_validation.py` | `test_validate_video_file` | Checks detection of nonexistent, unsupported, and empty video files. |

---

## 3. How to Run the Tests

### Option A: Standard Pytest (Recommended)
From the project root directory:

```powershell
# Windows PowerShell
python -m pytest tests/
```

```bash
# Linux / macOS
pytest tests/
```

**Expected Output**:
```text
============================= test session starts =============================
platform win32 -- Python 3.11.1, pytest-9.1.1, pluggy-1.7.0
rootdir: D:\PROJECTS\OPENCV\AI-Road-Safety
collected 18 items

tests\test_config.py ....                                                [ 22%]
tests\test_lanes.py ....                                                 [ 44%]
tests\test_safety.py ......                                              [ 77%]
tests\test_validation.py ....                                            [100%]

============================= 18 passed in 3.35s ==============================
```

### Option B: End-to-End Smoke Test
To verify the entire vision pipeline, model inference, and MP4 video generation:

```powershell
python tests/smoke_test.py
```

**Smoke Test Operations**:
1. Processes `assets/sample_images/highway_lanes.jpg` and writes `outputs/smoke_test_image.jpg`.
2. Processes `assets/sample_videos/sample_road_driving.mp4` with frame striding and saves `outputs/smoke_test_video.mp4`.
3. Verifies syntax and compilation of the Streamlit application (`app.py`).

---

## 4. Edge Cases Covered

1. **Missing Model Weights**:
   - Tested offline: ObjectDetector enters fallback mode and logs warnings; lane detection and HUD continue functioning.
2. **Zero-Byte / Corrupt Media**:
   - `validate_image_file` and `validate_video_file` catch and reject corrupt media before passing to decoders.
3. **Camera Disconnection**:
   - Live stream tab catches failed `cv2.VideoCapture` read loops and surfaces non-blocking UI warnings.
4. **Lane Markings Invisibility**:
   - Blank asphalt, night scenes, or curved intersections gracefully default to `NO_LANES` status without pipeline halt.
