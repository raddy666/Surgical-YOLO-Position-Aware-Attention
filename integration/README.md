# Integration with Ultralytics

This project extends Ultralytics YOLO (`ultralytics==8.3.185`) with 9 custom
attention mechanisms, following Ultralytics' own documented pattern for adding
custom modules. That pattern requires two small edits to Ultralytics' own
source files. This folder contains the already-modified files — copy them
directly into your installed package rather than re-deriving the edits by hand.

## Steps

1. Install the exact pinned version:
   pip install ultralytics==8.3.185

2. Run the integration script from the repo root — it does the file
   copying below automatically:
   python scripts/apply_integration.py

3. Verify:
   python -c "from ultralytics.nn.modules import C2Triplet; print('ok')"

## What the script does, if you'd rather do it by hand

- Finds your install location: python -c "import ultralytics, os; print(os.path.dirname(ultralytics.__file__))"
- Replaces two files in your installed package with the ones in this folder:
  - integration/nn/tasks.py            -> <install path>/nn/tasks.py
  - modules/attention/__init__.py -> <install path>/nn/modules/__init__.py
- Copies this repo's modules/attention/*.py files FLAT into <install path>/nn/modules/
  (not a subfolder — __init__.py above imports from .msca, .eca, .ca, .triplet,
  .c2cbam, .gc, .simam, .ema, .biformer sitting right next to it)
  
## What changed, exactly

- nn/tasks.py: ~19 names added to the module import statement, 9 classes added
  to base_modules and 9 to repeat_modules inside parse_model(), 9 new elif
  branches for the standalone (non-wrapped) versions of each mechanism.
- nn/modules/__init__.py: import + __all__ entries for all 9 mechanisms and
  their C2f wrappers.

## License note

tasks.py and modules/__init__.py in this folder are modified copies of
Ultralytics' own AGPL-3.0-licensed source, included under those terms (AGPL
header preserved in both). Everything else in this repository — the
attention implementations in modules/attention/, training scripts, and
results — is original work.
