import pytest
from ultralytics import YOLO

# expected params in millions, ~1% tolerance
CONFIGS = [
    ("configs/yolo11n-seg-c2msca.yaml", 3.65),        
    ("configs/hybrid/yolo11n-seg-L19_23-c2triplet-L15-c2ca.yaml", 3.56),   
    ("configs/yolo11n-seg-c2triplet.yaml", 3.27),
    ("configs/yolo11n-seg-c2ca.yaml", 3.28),
]

@pytest.mark.parametrize("yaml_path,expected_m", CONFIGS)
def test_config_builds_and_param_count_matches(yaml_path, expected_m):
    model = YOLO(yaml_path)
    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    assert abs(params_m - expected_m) / expected_m < 0.01, (
        f"{yaml_path}: expected ~{expected_m}M, got {params_m:.3f}M"
    )