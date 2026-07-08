import os
import shutil
import json
import random
from pathlib import Path

# Basic settings
source_dir = 'data/'
###################################
#Target directory for YOLO dataset#
###################################
target_root_dir = 'E:/research/train_YOLO/'  
W, H = 1920, 1080  # Image dimensions

# Category mapping
categories = {
    "椎间盘": "IntervertebralDisc",
    "骨骼": "Skeleton",
    "韧带": "Ligament",
    "肌肉": "Muscle",
    "神经": "Nerve",  
    "椎间盘突出核髓": "IntervertebralDiscHerniation",
}


category_mapping = {
    "椎间盘(纤维环)": "椎间盘",
    "椎间盘": "椎间盘",
    "间盘": "椎间盘",
    "骨头": "骨骼",
    "骨骼": "骨骼",
    "上关节突": "骨骼",
    "下关节突(断面)": "骨骼",
    "黄韧带": "韧带",
    "黄韧带(破口)": "其他",
    "韧带": "韧带",
    "下关节突": "骨骼",
    "下关节突及上位椎板下缘": "骨骼",
    "下关节突及上位椎板下缘(断面)": "骨骼",
    "椎间盘(突出的髓核)": "椎间盘突出核髓",
    "肌肉": "肌肉",
    "椎旁肌": "肌肉",
    "神经根或硬膜囊": "神经",
    "神经根": "神经",
    "神经": "神经",
    "B": "椎间盘突出核髓",
    "血管": "其他",
    "出血点": "其他",
    "脂肪": "其他",
    "硬膜外系膜": "其他",
    "椎间盘（纤维环)": "椎间盘",
    "椎间盘(纤维环破口)": "其他",
    "椎间盘(椎间盘内的髓核)": "其他",
    "椎间盘(纤维环破口内的髓核)": "其他",
}

id_mapping = {
    "IntervertebralDisc": 0,
    "Skeleton": 1,
    "Ligament": 2,
    "Muscle": 3,
    "Nerve": 4,
    "IntervertebralDiscHerniation": 5,
}

def normalize_label(label):
    """Standardize label format - completely handle Chinese/English parentheses, spaces and other format issues"""
    import re
    
    # Step 1: Handle mixed spaces inside and outside parentheses
    # Handle spaces before and after parentheses: 'IntervertebralDisc (FibrousRing)' -> 'IntervertebralDisc(FibrousRing)'
    label = re.sub(r'\s*（\s*', '（', label)
    label = re.sub(r'\s*）\s*', '）', label)
    label = re.sub(r'\s*\(\s*', '(', label)
    label = re.sub(r'\s*\)\s*', ')', label)
    label = re.sub(r'\s*【\s*', '【', label)
    label = re.sub(r'\s*】\s*', '】', label)
    label = re.sub(r'\s*\[\s*', '[', label)
    label = re.sub(r'\s*\]\s*', ']', label)
    
    # Step 2: Unify all parenthesis formats (Chinese parentheses to English parentheses)
    label = label.replace('（', '(').replace('）', ')')
    label = label.replace('【', '[').replace('】', ']')
    label = label.replace('｛', '{').replace('｝', '}')
    
    # Step 3: Handle full-width and half-width characters
    label = label.replace('：', ':').replace('，', ',')
    label = label.replace('；', ';').replace('。', '.')
    label = label.replace('！', '!').replace('？', '?')
    
    # Step 4: Remove all types of spaces (including Chinese spaces, tabs, newlines, etc.)
    label = label.replace(' ', '').replace('　', '').replace('\t', '').replace('\n', '').replace('\r', '')
    
    # Step 5: Remove leading and trailing whitespace and punctuation
    label = label.strip(' \t\n\r,:;.!?')
    
    # Step 6: Handle all invisible characters and extra spaces
    label = ''.join(label.split())
    
    # Step 7: Remove consecutive punctuation marks
    label = re.sub(r'[,:;.!?]+', lambda m: m.group()[0], label)
    
    return label

def validate_coordinates(x_norm, y_norm):
    """Validate whether normalized coordinates are within valid range"""
    return 0 <= x_norm <= 1 and 0 <= y_norm <= 1

def validate_polygon(points):
    """Validate whether polygon is valid (at least 3 points)"""
    return len(points) >= 3

def generate_data_yaml():
    """Generate YOLO dataset configuration file"""
    yaml_content = f"""# YOLO Dataset Configuration
path: {target_root_dir}
train: images/train
val: images/val

# Classes
nc: {len(categories)}  # number of classes
names: {list(categories.values())}  # class names
"""
    
    yaml_path = os.path.join(target_root_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"Dataset configuration file generated: {yaml_path}")

# Create normalized category_mapping
normalized_category_mapping = {normalize_label(k): v for k, v in category_mapping.items()}

# Create directory structure
def create_directories():
    """Create necessary directory structure"""
    for split in ['train', 'val']:
        for subdir in ['images', 'labels']:
            Path(target_root_dir, subdir, split).mkdir(parents=True, exist_ok=True)
    print("Directory creation completed")

def get_json_files(source_dir):
    """Generator function that returns JSON files and their corresponding image paths one by one"""
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                img_path = json_path.replace('.json', '.jpg')
                
                # Get relative path
                rel_path = os.path.relpath(root, source_dir)
                # Replace path separators with underscores, remove leading dots and slashes
                prefix = rel_path.strip('./\\').replace('\\', '_').replace('/', '_')
                
                # If prefix is not empty, add underscore
                if prefix:
                    new_file = f"{prefix}_{file}"
                else:
                    new_file = file
                    
                if os.path.exists(img_path):
                    yield json_path, img_path, new_file

def get_split(filename, train_ratio=0.85):
    """Determine dataset split based on random number, ensure reproducibility"""
    random.seed(filename)  # Use filename as random seed
    return 'train' if random.random() < train_ratio else 'val'

def process_single_file(json_path, img_path, filename, train_counts, val_counts):
    """Function to process a single file"""
    split = get_split(filename)
    
    # Set target paths
    target_img_path = os.path.join(target_root_dir, f'images/{split}/{filename.replace(".json", ".jpg")}')
    target_label_path = os.path.join(target_root_dir, f'labels/{split}/{filename.replace(".json", ".txt")}')
    
    # Check if file has been processed
    if os.path.exists(target_img_path) and os.path.exists(target_label_path):
        return None
    
    # Copy image
    shutil.copy(img_path, target_img_path)
    
    # Process annotation file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    category_counts = {'train': train_counts, 'val': val_counts}
    valid_annotations = 0
    
    # Write YOLO format labels
    with open(target_label_path, 'w') as f:
        for shape in data['shapes']:
            label = normalize_label(shape['label'])
            if label in normalized_category_mapping:
                major_category = normalized_category_mapping[label]
                if major_category in categories:
                    points = shape['points']
                    
                    # Validate polygon validity
                    if not validate_polygon(points):
                        print(f"Warning: Label '{label}' in {filename} has fewer than 3 polygon points, skipped")
                        continue
                    
                    class_id = id_mapping[categories[major_category]]
                    
                    # Validate and normalize coordinates
                    normalized_points = []
                    valid_polygon = True
                    
                    for x, y in points:
                        x_norm = x / W
                        y_norm = y / H
                        
                        # Validate coordinate range
                        if not validate_coordinates(x_norm, y_norm):
                            print(f"Warning: Label '{label}' in {filename} has coordinates out of range ({x_norm:.6f}, {y_norm:.6f}), skipped")
                            valid_polygon = False
                            break
                        
                        normalized_points.extend([x_norm, y_norm])
                    
                    if valid_polygon:
                        # Update category count
                        category_counts[split][categories[major_category]] += 1
                        valid_annotations += 1
                        
                        # Write YOLO format annotation
                        f.write(f"{class_id}")
                        for i in range(0, len(normalized_points), 2):
                            f.write(f" {normalized_points[i]:.6f} {normalized_points[i+1]:.6f}")
                        f.write("\n")
    
    # If no valid annotations, create empty file
    if valid_annotations == 0:
        with open(target_label_path, 'w') as f:
            pass  # Create empty file
    
    return split

def generate_dataset_files():
    """Generate dataset file lists"""
    for split in ['train', 'val']:
        with open(os.path.join(target_root_dir, f'{split}.txt'), 'w') as f:
            image_dir = Path(target_root_dir) / 'images' / split
            for img_path in image_dir.glob('*.jpg'):
                f.write(f"./images/{split}/{img_path.name}\n")

def main():
    """Main function"""
    print("Starting dataset processing...")
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_root_dir}")
    print(f"Image dimensions: {W}x{H}")
    
    # Create directories
    create_directories()
    
    # Initialize counters
    train_counts = {cat: 0 for cat in categories.values()}
    val_counts = {cat: 0 for cat in categories.values()}
    processed_files = {'train': 0, 'val': 0}
    
    # Process files
    batch_size = 100
    processed = 0
    
    for json_path, img_path, filename in get_json_files(source_dir):
        split = process_single_file(json_path, img_path, filename, train_counts, val_counts)
        if split:
            processed_files[split] += 1
            processed += 1
            if processed % batch_size == 0:
                print(f"Processed {processed} files...")
    
    # Generate dataset file lists
    generate_dataset_files()
    
    # Generate YOLO dataset configuration file
    generate_data_yaml()
    
    # Output statistics
    print("\n=== Processing Complete! ===")
    print(f"\nDataset directory structure:")
    print(f"  {target_root_dir}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/ ({processed_files['train']} images)")
    print(f"  │   └── val/ ({processed_files['val']} images)")
    print(f"  ├── labels/")
    print(f"  │   ├── train/ ({processed_files['train']} annotation files)")
    print(f"  │   └── val/ ({processed_files['val']} annotation files)")
    print(f"  ├── train.txt")
    print(f"  ├── val.txt")
    print(f"  └── data.yaml")
    
    print("\n=== Category Statistics ===")
    print("Training set:")
    for category, count in train_counts.items():
        print(f"  {category}: {count}")
    
    print("\nValidation set:")
    for category, count in val_counts.items():
        print(f"  {category}: {count}")
    
    # Calculate totals
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    print(f"\n=== Totals ===")
    print(f"Training set: {processed_files['train']} files, {total_train} annotated objects")
    print(f"Validation set: {processed_files['val']} files, {total_val} annotated objects")
    print(f"Total: {processed_files['train'] + processed_files['val']} files, {total_train + total_val} annotated objects")

if __name__ == "__main__":
    main()