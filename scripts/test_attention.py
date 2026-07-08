import sys
import torch
import traceback

print("\n" + "="*70)
print("ATTENTION MECHANISM VALIDATION TEST SUITE")
print("="*70 + "\n")

# ============================================================
# TEST 1: IMPORTS
# ============================================================
print("="*70)
print("TEST 1: Import Validation")
print("="*70)

try:
    from ultralytics.nn.modules.conv import CBAM
    from ultralytics.nn.modules.eca import ECA
    from ultralytics.nn.modules.c2cbam import C2CBAM
    from ultralytics.nn.modules.eca import  C2ECA
    from ultralytics.nn.modules.msca import C2MSCA
    from ultralytics.nn.modules.ca import C2CA
    from ultralytics.nn.modules.ca import CoordinateAttention
    from ultralytics.nn.modules.gc import C2GC
    from ultralytics.nn.modules.gc import GlobalContext
    from ultralytics.nn.modules.triplet import  TripletAttention
    from ultralytics.nn.modules.triplet import C2Triplet
    from ultralytics.nn.modules.simam import SimAM
    from ultralytics.nn.modules.simam import C2SimAM
    from ultralytics.nn.modules.ema import C2EMA, EMA
    from ultralytics.nn.modules.biformer import BiFormer, C2BiFormer
    from ultralytics.nn.modules import CBAM as CBAM_init, ECA as ECA_init
    
    print(" All imports successful!")
    print("   - CBAM, ECA (from conv.py)")
    print("   - C2CBAM, C2ECA, C2MSCA (from block.py)")
    print("   - Verified __init__.py exports")
    
except ImportError as e:
    print(f" IMPORT FAILED: {e}")
    print("\nACTION REQUIRED:")
    print("1. Check conv.py has ECA class")
    print("2. Check block.py has C2CBAM and C2ECA classes")
    print("3. Check __init__.py has proper exports")
    sys.exit(1)

# ============================================================
# TEST 2: FORWARD PASS
# ============================================================
print("\n" + "="*70)
print("TEST 2: Forward Pass Validation")
print("="*70)

batch_size = 2
channels = 256
height = 32
width = 32
x = torch.randn(batch_size, channels, height, width)

print(f"\nTest input shape: {x.shape}\n")

tests = [
    ("CBAM", lambda: CBAM(c1=channels, kernel_size=7)),
    ("ECA", lambda: ECA(channels=channels, gamma=2, b=1)),
    ("C2CBAM", lambda: C2CBAM(c1=channels, c2=channels, n=1, e=0.5)),
    ("C2ECA", lambda: C2ECA(c1=channels, c2=channels, n=1, e=0.5)),
    ("C2MSCA", lambda: C2MSCA(c1=channels, c2=channels, n=1, e=0.5)),
    ("C2CA", lambda: C2CA(c1=channels, c2=channels, n=1, e=0.5)),
    ("CoordinateAttention", lambda: CoordinateAttention(channels=channels, reduction=32)),
    ("C2GC", lambda: C2GC(c1=channels, c2=channels, n=1, e=0.5)),
    ("GlobalContext", lambda: GlobalContext(channels, reduction=16)),
    ("TripletAttention", lambda: TripletAttention()),
    ("C2Triplet", lambda: C2Triplet(c1=channels, c2=channels, n=1, e=0.5)),
    ("SimAM", lambda: SimAM()),
    ("C2SimAM", lambda: C2SimAM(c1=channels, c2=channels, n=1, e=0.5)),
    ("EMA", lambda: EMA(channels=channels, reduction=16)),
    ("C2EMA", lambda: C2EMA(c1=channels, c2=channels, n=1, e=0.5)),
    ("BiFormer", lambda: BiFormer(dim=channels, num_heads=8)),
    ("C2BiFormer", lambda: C2BiFormer(c1=channels, c2=channels, n=1)),
]

results = {}

for name, create_module in tests:
    print(f"--- Testing {name} ---")
    try:
        module = create_module()
        y = module(x)
        params = sum(p.numel() for p in module.parameters())
        
        assert y.shape == x.shape, f"Shape mismatch! Expected {x.shape}, got {y.shape}"
        
        print(f" {name} passed")
        print(f"   Shape: {x.shape} → {y.shape}")
        print(f"   Parameters: {params:,}\n")
        
        results[name] = {'params': params, 'passed': True}
        
    except Exception as e:
        print(f" {name} FAILED: {e}\n")
        traceback.print_exc()
        results[name] = {'passed': False}
        sys.exit(1)

# ============================================================
# TEST 3: YAML INTEGRATION
# ============================================================
print("="*70)
print("TEST 3: YAML Integration Test")
print("="*70)

try:
    from ultralytics import YOLO
    
    yaml_path = "experiments/yolo11n-seg-c2biformer.yaml"
    print(f"\nLoading model from: {yaml_path}")
    
    model = YOLO(yaml_path)
    print(" Model loaded successfully!")
    
    dummy_input = torch.randn(1, 3, 640, 640)
    model.model.eval()
    
    with torch.no_grad():
        output = model.model(dummy_input)
    
    print(" Model forward pass successful!")
    print(f"   Input: {dummy_input.shape}")
    
except FileNotFoundError:
    print(f"  Test YAML not found: {yaml_path}")
    print("   This is optional - you can create it manually later")
    
except Exception as e:
    print(f" YAML integration failed: {e}")
    traceback.print_exc()
    print("\nACTION REQUIRED:")
    print("1. Check YAML syntax")
    print("2. Verify module names match")
    sys.exit(1)

# ============================================================
# TEST 4: PARAMETER COMPARISON
# ============================================================
print("\n" + "="*70)
print("TEST 4: Parameter Comparison")
print("="*70)

print(f"\nParameter comparison (256 channels):\n")
print(f"{'Mechanism':<20} {'Parameters':<15} {'vs Baseline':<15}")
print("-" * 50)

baseline_params = results['C2MSCA']['params']

for name in ['C2MSCA', 'C2CBAM', 'C2ECA', 'C2CA', 'C2GC', 'C2SimAM', 'C2Triplet', 'C2EMA', 'C2BiFormer']:
    if results[name]['passed']:
        params = results[name]['params']
        diff = ((params / baseline_params) - 1) * 100
        print(f"{name:<20} {params:>12,}   {diff:>+6.2f}%")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

all_passed = all(r['passed'] for r in results.values())

if all_passed:
    print("\n ALL TESTS PASSED! ")
    print("\nNext steps:")
    print(" Run training with train.py")
else:
    print("\n SOME TESTS FAILED!")
    print("\nFix errors before running experiments!\n")
    sys.exit(1)