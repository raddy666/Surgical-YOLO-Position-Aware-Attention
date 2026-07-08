import os
from pathlib import Path
from collections import defaultdict

label_dirs = {
    'train': 'labels/train',
    'val':   'labels/val'
}
class_names = ['Disc', 'Skeleton', 'Ligament', 'Muscle', 'Nerve', 'Herniation']

for split, path in label_dirs.items():
    counts = defaultdict(int)
    files = 0
    for f in Path(path).glob('*.txt'):
        files += 1
        for line in f.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    print(f"\n{split} ({files} images):")
    for i, name in enumerate(class_names):
        print(f"  {name}: {counts[i]}")
