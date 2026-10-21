"""
Utility script to create anatomically patterned sample Chest X-Rays for local demonstration and unit testing.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def generate_cxr(variant: str = "normal", width: int = 512, height: int = 512) -> Image.Image:
    # Dark thoracic background
    img_array = np.full((height, width), 25, dtype=np.float32)
    
    # Coordinates
    y, x = np.ogrid[:height, :width]
    cx, cy = width // 2, height // 2
    
    # Thoracic cage border
    thorax_mask = ((x - cx)**2) / ((width * 0.42)**2) + ((y - cy*1.05)**2) / ((height * 0.45)**2) <= 1.0
    img_array[thorax_mask] += 40
    
    # Left and Right Lung Fields (radiolucent = darker)
    left_lung_mask = (((x - (cx - width*0.22))**2) / ((width * 0.16)**2) + ((y - (cy - height*0.05))**2) / ((height * 0.32)**2)) <= 1.0
    right_lung_mask = (((x - (cx + width*0.22))**2) / ((width * 0.16)**2) + ((y - (cy - height*0.05))**2) / ((height * 0.32)**2)) <= 1.0
    
    img_array[left_lung_mask] = 18
    img_array[right_lung_mask] = 18
    
    # Mediastinum and Spine (dense = whiter)
    spine_mask = (np.abs(x - cx) <= (width * 0.04)) & (y < height * 0.9)
    img_array[spine_mask] += 95
    
    # Clavicles
    clavicle_left = (np.abs((y - height*0.18) - 0.15*(x - cx)) <= 6) & (x < cx - width*0.05) & (x > cx - width*0.38)
    clavicle_right = (np.abs((y - height*0.18) + 0.15*(x - cx)) <= 6) & (x > cx + width*0.05) & (x < cx + width*0.38)
    img_array[clavicle_left] += 70
    img_array[clavicle_right] += 70
    
    # Rib shadows (horizontal curved lines)
    for r_y in range(int(height * 0.22), int(height * 0.75), 45):
        rib_mask = (np.abs(y - (r_y + 0.0004 * (x - cx)**2)) <= 5) & thorax_mask
        img_array[rib_mask] += 25
        
    # Cardiac Silhouette
    cardiac_radius_x = width * 0.26 if variant == "cardiomegaly" else width * 0.17
    cardiac_radius_y = height * 0.22 if variant == "cardiomegaly" else height * 0.16
    cardiac_center_x = cx + (width * 0.08 if variant == "cardiomegaly" else width * 0.05)
    cardiac_center_y = cy + height * 0.16
    
    heart_mask = (((x - cardiac_center_x)**2) / (cardiac_radius_x**2) + ((y - cardiac_center_y)**2) / (cardiac_radius_y**2)) <= 1.0
    img_array[heart_mask] += 120 if variant == "cardiomegaly" else 90

    # Pathological features
    if variant == "pneumonia":
        # Dense alveolar opacity in right mid/lower zone
        pn_cx = cx + width * 0.20
        pn_cy = cy + height * 0.10
        dist = np.sqrt((x - pn_cx)**2 + (y - pn_cy)**2)
        pn_mask = dist < (width * 0.14)
        noise = np.random.normal(0, 15, (height, width))
        img_array[pn_mask] += (80 * (1 - dist[pn_mask] / (width * 0.14)) + noise[pn_mask])

    # Diaphragm domes
    diaphragm_left = (y >= cy + height * 0.28 - 0.001 * (x - (cx - width*0.22))**2) & (x < cx) & thorax_mask
    diaphragm_right = (y >= cy + height * 0.26 - 0.001 * (x - (cx + width*0.22))**2) & (x >= cx) & thorax_mask
    img_array[diaphragm_left] += 110
    img_array[diaphragm_right] += 110

    # Smooth and clip
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    image = Image.fromarray(img_array).filter(ImageFilter.GaussianBlur(radius=2))
    return image

if __name__ == "__main__":
    out_dir = Path("data/sample_images")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    generate_cxr("normal").save(out_dir / "sample_cxr_normal.png")
    generate_cxr("cardiomegaly").save(out_dir / "sample_cxr_cardiomegaly.png")
    generate_cxr("pneumonia").save(out_dir / "sample_cxr_pneumonia.png")
    print("Created 3 sample CXR images in data/sample_images/")
