import torch
import numpy as np
from model import Unet_Diffusion
from PIL import Image
from torchvision.utils import make_grid
import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
from train import image_trans, mask_test_trans
from torchvision import transforms
from DDPM import SegmentationDataset
import torch.nn.functional as F
from pytorch_msssim import ssim
import time
import matplotlib.pyplot as plt



#Single test
'''
# read img(gray)
mask = Image.open(r"D:\correct\mask\alan_face1_173645_0100.png")
mask = np.array(mask, dtype=np.float32)  # (H, W)
mask = mask / 255.0
# (H, W) → (1, H, W)
mask = torch.from_numpy(mask).unsqueeze(0)

image = Image.open(r"D:\correct\image\alan_face1_173645_0100.pgm")
image = np.array(image, dtype=np.float32)
image = image / 65535.0
image = image * 2 - 1
image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
image = image.cuda()
print(image.shape)
'''

def denormalize(img):
    img = (img + 1.0) / 2.0
    img = torch.clamp(img, 0, 1)
    return img
    #img = img * 65535.0

def sample_and_show_process(model, mask, T, betas):
    model.eval()
    device = next(model.parameters()).device
    

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), alphas_cumprod[:-1]])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)


    img_shape = (1, 1, mask.shape[-2], mask.shape[-1]) 
    img = torch.randn(img_shape, device=device)
    
    mask_input = mask.unsqueeze(0).to(device) # (1, 1, H, W)
    

    process_images = []


    for i in reversed(range(0, T)):
        t = torch.full((1,), i, device=device, dtype=torch.long)

        predicted_noise = model(img, t, mask_input)

        alpha_t = alphas[i]
        beta_t = betas[i]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[i]
        
        model_mean = sqrt_recip_alphas[i] * (
            img - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise
        )
        
        if i > 0:
            noise = torch.randn_like(img)
            sigma_t = torch.sqrt(posterior_variance[i])
            img = model_mean + sigma_t * noise
        else:
            img = model_mean

        # every 100 step record
        if i % 100 == 0 or i == 0:
            print(f"record the image of the step {i}...")
            # denormalize
            temp_img = denormalize(img)
            process_images.append(temp_img.cpu())

    # Combine all process diagrams into one long image 
    grid = make_grid(torch.cat(process_images, dim=0), nrow=len(process_images))
    #img = denormalize(img)
    return img, grid



def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class SegmentationDataset_test(Dataset):
    def __init__(self, image_dir, mask_dir, gt_dir,img_transform=None, mask_transform=None):
        self.image_paths = sorted(glob(image_dir + r"\*"))
        self.mask_paths  = sorted(glob(mask_dir  + r"\*"))

        self.mask_filenames = [os.path.basename(p) for p in self.mask_paths]
        self.new_mask_paths = [os.path.join(gt_dir, name) for name in self.mask_filenames]
        assert len(self.image_paths) == len(self.mask_paths)

        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        mask  = Image.open(self.new_mask_paths[idx])

        if self.img_transform:
            image = self.img_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask



'''''
with torch.no_grad():
    final_img, process_grid = sample_and_show_process(model, mask, T, betas)



img_np = (final_img.squeeze().cpu().numpy() * 65535).astype(np.uint16)
cv2.imwrite("final_result.pgm", img_np)


#save_image(final_img, "final_result.png")

save_image(process_grid, "sampling_process.png")


writer.add_image("Sampling_Process", process_grid, 0)
print("save！ sampling_process.png")
'''''


#gt_mask_dir = r"H:\\gt_mask"  #resize
def Test_single_parameter(image_dir =r"test\image\0001.pgm", mask_dir  = r"test\mask\0001.png", model_path = r'model/unet_diffusion_lowestloss_0214_14.pth', show_picture = True):
    
    
    mask = Image.open(mask_dir)
    mask = np.array(mask, dtype=np.float32)  # (H, W)
    mask = mask / 255.0
    # (H, W) → (1, H, W)
    mask = torch.from_numpy(mask).unsqueeze(0)

    image = Image.open(image_dir)
    image = np.array(image, dtype=np.float32)
    image = image / 65535.0
    image = image * 2 - 1
    image = torch.from_numpy(image).unsqueeze(0)
    image= torch.where(mask > 0.1, image, -1.0)      
    image = image.unsqueeze(0)
    
    model = Unet_Diffusion() 
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = image.to(device)
    T = 1000
    betas = linear_beta_schedule(T).to(device)
    


    with torch.no_grad():
        final_img, process_grid = sample_and_show_process(model, mask, T, betas)
        mse_loss = F.mse_loss(final_img, image)
        if show_picture:
            gen_img_np = denormalize(final_img.squeeze().cpu())
            gt_img_np = denormalize(image.squeeze().cpu())
            gen_img_np = gen_img_np.numpy()
            gt_img_np = gt_img_np.numpy()
            process_grid.cpu().numpy

            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            plt.figure(figsize=(12,4))
            plt.imshow(process_grid[0], cmap='gray')
            plt.title("Processing")
            plt.axis('off')

            axes[0].imshow(gen_img_np, cmap='gray')
            axes[0].set_title("Generated Image")
            axes[0].axis('off')

            axes[1].imshow(gt_img_np, cmap='gray')
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')

            # Display
            plt.tight_layout()
            print("Opening image window...")
            plt.show()  #      

        ssim_test = ssim(final_img, image, data_range=2.0, size_average=True)
        ssim_loss = 1 - ssim_test
        print("mse loss:", mse_loss)
        print("ssim loss:", ssim_loss)        




def Test_set_parameter(image_dir = r"test\\image", mask_dir  = r"test\\mask", model_path = r'model\\unet_diffusion_lowestloss_0214_14.pth'):
    
    model = Unet_Diffusion() 
    model.load_state_dict(torch.load(model_path))
    model.cuda()
   
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 1000
    betas = linear_beta_schedule(T).to(device)
    img_transform = image_trans()
    mask_transform = mask_test_trans() 
    #gt_mask_dir = r"H:\\gt_mask"
    #test_dataset = SegmentationDataset_test(image_dir, mask_dir, gt_mask_dir, img_transform, mask_transform)
    test_dataset = SegmentationDataset(image_dir, mask_dir, img_transform, mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():

        start_time = time.time()

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        test_ssim_running_loss = 0
        test_mse_running_loss = 0
        n = 0
        print("data size:",len(test_loader))
        for img_test, mask in test_loader:
            now_time = time.time()
            n += 1
            if n % 500 == 0 : print(f"picture number:{n} \n time:{now_time - start_time}s")

            img_test, mask = img_test.cuda(), mask.cuda()
            img_test = torch.where(mask > 0.1, img_test, -1.0)        
            model.eval()
            device = next(model.parameters()).device
            

            # 1. initialization
            img_shape = (1, 1, mask.shape[-2], mask.shape[-1]) 
            pre_img = torch.randn(img_shape, device=device)
            

            # 2. sampling
            for i in reversed(range(0, T)):
                t = torch.full((1,), i, device=device, dtype=torch.long)
                #print(t.shape)
                # predict
                predicted_noise = model(pre_img, t, mask)
                
                # DDPM Backsampling
                alpha_t = alphas[i]
                beta_t = betas[i]
                sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[i]
                
                model_mean = sqrt_recip_alphas[i] * (
                    pre_img - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise
                )
                
                if i > 0:
                    noise = torch.randn_like(pre_img)
                    sigma_t = torch.sqrt(posterior_variance[i])
                    pre_img = model_mean + sigma_t * noise
                else:
                    pre_img = model_mean

            mse_loss = F.mse_loss(pre_img, img_test)
            ssim_test = ssim(pre_img, img_test, data_range=2.0, size_average=True)
            ssim_loss = 1 - ssim_test
            test_mse_running_loss += mse_loss.item()
            test_ssim_running_loss += ssim_loss.item()
            '''
            t_img = torch.randn(img_shape, device=device)
            t_mse = F.mse_loss(image,img_test)
            t_ssim_test = ssim(image, img_test, data_range=2.0, size_average=True)
            t_ssim_loss = 1 - t_ssim_test
            print("mse:", mse_loss)
            print("ssim:", ssim_loss)
            print("t_mse:",t_mse)
            print("t_ssim:",t_ssim_loss)
            '''
        mse_test_loss = test_mse_running_loss / len(test_loader)
        ssim_test_loss = test_ssim_running_loss / len(test_loader)
        print("total test mse:", mse_test_loss)
        print("total test ssim:", ssim_test_loss)
if __name__ == '__main__':
    Test_single_parameter()