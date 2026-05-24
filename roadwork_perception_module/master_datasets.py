import os
import shutil
import yaml
from pathlib import Path

# 1. GLOBAL MASTER SCHEMA DEFINITION
MASTER_CLASSES = [
    "hardhat", "mask", "no-hardhat", "no-mask", "no-vest", 
    "person", "safety-cone", "vehicle", "vest", "road-sign", "road-marking"
]

# 2. CONFIGURATION MAP FOR YOUR 5 LOCAL FOLDERS
DATASET_CONFIGS = [
    {
        "path": "./Construction-site-safety-1",
        "mapping": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
    },
    {
        "path": "./traffic-cone-1",
        "mapping": {0: 6}  # Maps local cone index 0 to master safety-cone index 6
    },
    {
        "path": "./workers-1",
        "mapping": {0: 5}  # Maps local worker index 0 to master person index 5
    },
    {
        "path": "./road-signs-1",
        "mapping": {0: 9}  # Maps local sign index 0 to master road-sign index 9
    },
    {
        "path": "./Road-1",
        "mapping": {0: 10} # Maps local road-marking index 0 to master road-marking index 10
    }
]

MASTER_DIR = Path("./master_dataset")

def consolidate_data(config, ds_id):
    src_root = Path(config["path"])
    class_map = config["mapping"]
    
    if not src_root.exists():
        print(f"⚠️ Warning: Folder {src_root} not found. Skipping...")
        return

    print(f"--> Merging assets from: {src_root.name}")

    # Process train and validation splits (Roboflow uses 'valid' folder layout)
    for src_split, dest_split in [("train", "train"), ("valid", "val")]:
        img_src_dir = src_root / src_split / "images"
        lbl_src_dir = src_root / src_split / "labels"
        
        if not img_src_dir.exists():
            continue
            
        for img_path in img_src_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
                
            # Prepend dataset ID to filenames to prevent overwrite collisions
            unique_filename = f"ds{ds_id}_{img_path.name}"
            lbl_filename = img_path.stem + ".txt"
            lbl_path = lbl_src_dir / lbl_filename
            
            dest_img = MASTER_DIR / "images" / dest_split / unique_filename
            dest_lbl = MASTER_DIR / "labels" / dest_split / f"ds{ds_id}_{lbl_filename}"
            
            # Copy Image
            shutil.copy(img_path, dest_img)
            
            # Read, Remap Class IDs, and Copy Label Coordinates
            if lbl_path.exists():
                with open(lbl_path, "r") as f:
                    lines = f.readlines()
                
                updated_lines = []
                for line in lines:
                    elements = line.strip().split()
                    if not elements:
                        continue
                    
                    local_id = int(elements[0])
                    if local_id in class_map:
                        elements[0] = str(class_map[local_id])
                        updated_lines.append(" ".join(elements) + "\n")
                
                if updated_lines:
                    with open(dest_lbl, "w") as f:
                        f.writelines(updated_lines)

if __name__ == "__main__":
    print("🚀 Starting localized multi-dataset aggregation stack...")
    
    for dataset_identifier, configuration in enumerate(DATASET_CONFIGS):
        consolidate_data(configuration, dataset_identifier)
        
    # Write the master validation YAML file
    master_yaml_config = {
        "path": str(MASTER_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(MASTER_CLASSES)}
    }
    
    with open(MASTER_DIR / "data.yaml", "w") as f:
        yaml.dump(master_yaml_config, f, default_flow_style=False)
        
    print(f"\n✅ Success! Integrated configuration saved to: {MASTER_DIR.resolve()}/data.yaml")
