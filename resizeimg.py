import cv2
import os
from glob import glob

input_dir = r"D:\code\SkinOCT_Diffusion\exp\mask"
output_dir = r"D:\code\SkinOCT_Diffusion\exp\mask_512"

os.makedirs(output_dir, exist_ok=True)

paths = glob(os.path.join(input_dir, "*.png"))

for path in paths:
    # 讀取 (灰階)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
    print(f"Processing {path}, original shape: {img.shape}")
    # resize
    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)

    print(f"Resized shape: {img_resized.shape}")
    # 存檔
    filename = os.path.basename(path)
    save_path = os.path.join(output_dir, filename)
    
    cv2.imwrite(save_path, img_resized )

print("Done!")