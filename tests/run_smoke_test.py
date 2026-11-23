"""Quick smoke test for the full pipeline on a real BDD100K image."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.road_safety.config import get_default_config
from src.road_safety.processing.pipeline import RoadSafetyPipeline

print("Loading pipeline...")
config = get_default_config(Path("configs/config.yaml"))
pipeline = RoadSafetyPipeline(config=config)
print(f"Model loaded: {pipeline.detector.is_loaded}")

bdd_dir = Path("assets/real_images/bdd100k")
images = list(bdd_dir.glob("*.jpg"))
if not images:
    print("No BDD100K images found. Run: python scripts/setup_real_data.py")
    sys.exit(1)

img_path = images[0]
print(f"Testing on real BDD100K image: {img_path.name}")
result = pipeline.process_image(img_path)

h, w = result.annotated_frame.shape[:2]
print(f"  Output frame size: {w}x{h}")
print(f"  Total detections: {len(result.detections)}")
for d in result.detections:
    print(f"    -> {d.class_name}  conf={d.confidence:.2f}  depth_proxy={d.depth_proxy:.2f}")

lane_ok = result.lane_result.lane_detected if result.lane_result else False
print(f"  Lane detected: {lane_ok}")

risk_label = result.risk.overall_level.label if result.risk else "N/A"
print(f"  Risk level: {risk_label}")
print(f"  Latency: {result.latency_ms:.1f} ms")

# Save annotated output
out_path = Path("outputs/smoke_test_output.jpg")
out_path.parent.mkdir(exist_ok=True)
import cv2
cv2.imwrite(str(out_path), result.annotated_frame)
print(f"  Annotated output saved to: {out_path}")

print("\n=== SMOKE TEST PASSED ===")
