# Integration with Ultralytics

This project extends Ultralytics YOLO (`ultralytics==8.3.185`) with 9 custom
attention mechanisms, following Ultralytics' own documented pattern for adding
custom modules. That pattern requires two small edits to Ultralytics' own
source files. This folder contains the already-modified files — copy them
directly into your installed package rather than re-deriving the edits by hand.

## Steps

1. Install the exact pinned version:
   pip install ultralytics==8.3.185

2. Find where it installed:
   python -c "import ultralytics, os; print(os.path.dirname(ultralytics.__file__))"

3. Replace two files in your installed package with the ones in this folder:
   - integration/nn/tasks.py            -> <install path>/nn/tasks.py
   - integration/nn/modules/__init__.py -> <install path>/nn/modules/__init__.py

4. Copy the actual attention module source into your installed package —
   this repo's modules/attention/*.py files go FLAT into <install path>/nn/modules/
   (not into a subfolder — the __init__.py above imports from .msca, .eca, .ca,
   .triplet, .c2cbam, .gc, .simam, .ema, .biformer sitting right next to it):

   cp modules/attention/*.py <install path>/nn/modules/

5. Verify:
   python -c "from ultralytics.nn.modules import C2Triplet; print('ok')"

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
results — is original work, licensed separately (see the top-level LICENSE).