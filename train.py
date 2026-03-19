import model
from glob import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import numpy as np
from model import Unet
from torch import nn
import time
import torchvision.transforms.functional as TF
import random
import torch.nn.functional as F


image_dir = r"D:\\correct\\image"
mask_dir  = r"D:\\correct\\mask"
#Min = 0.0, Max = 65535.0

trans_totensor = transforms.ToTensor()
trans_norm = transforms.Normalize([0.5],[0.5])

def get_dataset_stats(loader):
    dataset_min = float('inf')
    dataset_max = float('-inf')

    print("Calculating dataset statistics...")
    for imgs, _ in loader:
        batch_min = imgs.min()
        batch_max = imgs.max()
        
        if batch_min < dataset_min:
            dataset_min = batch_min.item()
        if batch_max > dataset_max:
            dataset_max = batch_max.item()
            
    return dataset_min, dataset_max

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_transform=None, mask_transform=None):
        self.image_paths = sorted(glob(image_dir + r"\*"))
        self.mask_paths  = sorted(glob(mask_dir  + r"\*"))

        assert len(self.image_paths) == len(self.mask_paths)

        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        mask  = Image.open(self.mask_paths[idx])

        if self.img_transform:
            image = self.img_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

class image_trans:

    def __call__(self, img):
        # PIL → numpy
        img = np.array(img, dtype=np.float32)  # (H, W)
        #img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = img / 65535.0
        img = img * 2 - 1
        # (H, W) → (1, H, W)
        
        img = torch.from_numpy(img).unsqueeze(0)
        
        return img
    
class ImageMaskAugment:
    def __init__(self):
        # Color Jitter for image
        self.jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)

    def __call__(self, img, mask):
        # --- 1. ColorJitter ---
        #img = self.jitter(img)

        # --- 2. Randomly flip ---
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # --- 3. Randomly flip ---
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        # --- 4. Random rotation ---
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)

        return img, mask
       
class image_test_trans:

    def __call__(self, img):

        img = np.array(img, dtype=np.float32)

        img = img / 65535.0
        img = img * 2 - 1

        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        # shape: (1, 1, H, W)

        img = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)

        img = img.squeeze(0)  # → (1, 512, 512)

        return img
    
class mask_test_trans:
    
    def __call__(self, mask):

        # PIL → numpy
        mask = np.array(mask, dtype=np.float32)  # (H, W)
        mask = mask / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

        mask = F.interpolate(mask, size=(512, 512), mode='bilinear', align_corners=False)
        mask = mask.squeeze(0)  
        return mask

class mask_trans:
    def __call__(self, mask):
        # PIL → numpy
        mask = np.array(mask, dtype=np.float32)  # (H, W)
        mask = mask / 255.0

        # (H, W) → (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return mask
    
def get_metrics(outputs, mask):
   # Convert the output to binary (0 or 1)
    preds = (outputs > 0.5).float()
    mask = mask.float()
    
    # 1. Accuracy
    correct = (preds == mask).sum().item()
    total = torch.numel(preds)
    acc = correct / total
    
    # 2. Dice Coefficient
    intersection = (preds * mask).sum().item()
    dice = (2. * intersection) / (preds.sum().item() + mask.sum().item() + 1e-6)
    
    return acc, dice
if __name__ == '__main__':

    img_transform = image_trans()
    mask_transform = mask_trans() 


    dataset = SegmentationDataset(image_dir, mask_dir, img_transform, mask_transform)

    # (90% train, 10% validation)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    #d_min, d_max = get_dataset_stats(train_loader)
    #print(f"Data range: Min = {d_min}, Max = {d_max}")

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,   
    )

    writer = SummaryWriter("loading_0204")
    step = 0
    '''''
    for data in loader:
        img, mask = data
        writer.add_images("loading_mask", mask, global_step=step)
        writer.add_images("loading_img", img, global_step=step)
        #print((mask[0]),mask.shape)
        step += 1
        if step > 3:
            break
    '''''
    unet = Unet()
    unet = unet.cuda()
    loss_fn = nn.BCELoss()
    loss_fn = loss_fn.cuda()
    opt = torch.optim.Adam(unet.parameters(), lr = 1e-4)



    total_train_step = 0
    total_test_step = 0
    start_time = time.time()

    # --- 3. Training and validation loop ---
    epochs = 5

    for epoch in range(epochs):
        # --- training ---
        unet.train()
        train_loss, train_acc, train_dice = 0, 0, 0
        
        for img, mask in train_loader:
            img, mask = img.cuda(), mask.cuda()
            
            outputs = unet(img)
            loss = loss_fn(outputs, mask)
            
            # Indicator calculation
            acc, dice = get_metrics(outputs, mask)
            train_acc += acc
            train_dice += dice
            train_loss += loss.item()
            total_train_step += 1
            writer.add_scalar("loss/train_in_epoch", loss.item()/len(img), total_train_step)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # --- validation ---
        unet.eval()
        val_loss, val_acc, val_dice = 0, 0, 0
        
        with torch.no_grad(): 
            for img, mask in val_loader:
                img, mask = img.cuda(), mask.cuda()
                outputs = unet(img)
                
                loss = loss_fn(outputs, mask)
                acc, dice = get_metrics(outputs, mask)
                
                val_loss += loss.item()
                val_acc += acc
                val_dice += dice

        # --- print results ---
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train - Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc/len(train_loader):.4f}, Dice: {train_dice/len(train_loader):.4f}")
        print(f"Val   - Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc/len(val_loader):.4f}, Dice: {val_dice/len(val_loader):.4f}")
        print("-" * 30)

        # Write to TensorBoard
        writer.add_scalar("Loss/Train", train_loss/len(train_loader), epoch)
        writer.add_scalar("Loss/Val", val_loss/len(val_loader), epoch)
        writer.add_scalar("Dice/Val", val_dice/len(val_loader), epoch)

    writer.close()

    writer.close()
        


