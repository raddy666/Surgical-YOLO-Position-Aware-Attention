import os
import shutil
import ultralytics

UL_PATH = os.path.dirname(ultralytics.__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

shutil.copy(
    os.path.join(REPO_ROOT, "integration", "nn", "tasks.py"),
    os.path.join(UL_PATH, "nn", "tasks.py"),
)
shutil.copy(
    os.path.join(REPO_ROOT, "modules", "attention", "__init__.py"),
    os.path.join(UL_PATH, "nn", "modules", "__init__.py"),
)

attention_dir = os.path.join(REPO_ROOT, "modules", "attention")
target_dir = os.path.join(UL_PATH, "nn", "modules")
for filename in os.listdir(attention_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        shutil.copy(os.path.join(attention_dir, filename), os.path.join(target_dir, filename))

print(f"Patched Ultralytics install at: {UL_PATH}")
print("Verify with: python -c \"from ultralytics.nn.modules import C2Triplet; print('ok')\"")