import torch.nn.functional as F
import torch
from model import Unet_Diffusion
from train import SegmentationDataset, image_trans, mask_trans, ImageMaskAugment
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import time
from pytorch_msssim import ssim
from datetime import datetime
import json

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# Set total time steps
T = 1000
betas = linear_beta_schedule(T)

# Calculate the coefficients needed in mathematical formulas
device = "cuda" if torch.cuda.is_available() else "cpu"

betas = linear_beta_schedule(T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0).to(device)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)



def q_sample(x_0, t, noise=None):
    """Add noise to the clean image x_0 according to time t."""
    if noise is None:
        noise = torch.randn_like(x_0)
    
    # Extract the corresponding coefficients based on t.
    sqrt_alpha_bar_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    
    return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

def diffusion_train(data_augment = True, 
                    pretrain_model = r'model/unet_diffusion_lowestloss_0214_14.pth', 
                    config_path = r"config/config_quickstart.json"):
    experiment_name = "unet_diffusion"
    run_id = datetime.now().strftime("%m%d_%H")
    unet = Unet_Diffusion().cuda()
    if pretrain_model:
        unet.load_state_dict(torch.load(pretrain_model)) 

    with open(config_path, "r") as f:
        config = json.load(f)
    experiment_name = config["experiment_name"]
    ssim_alpha = config["ssim_alpha"]
    start_lr = config["start_lr"]
    patience = config["patience"]
    factor = config["factor"]
    early_stop = config["early_stop"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]   

    optimizer = torch.optim.Adam(unet.parameters(), lr=start_lr)
    """""
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',       
        factor=factor,       
        patience=patience,      
    )
    """""
    experimental_notes = f"""
    ### Experiment number: {run_id}
    * **Modifications**: quick test.
    * **Augmentation**: {data_augment}
    * **loss**: mse + ssim(alhpa={ssim_alpha})
    * **parameter**: start_lr =  {start_lr},  factor = {factor},  patience = {patience},  early_stop = {early_stop}
    """
    
    image_dir = r"train\\image"
    mask_dir  = r"train\\mask"

    img_transform = image_trans()
    mask_transform = mask_trans() 
    img_aug = ImageMaskAugment()
    dataset = SegmentationDataset(image_dir, mask_dir, img_transform, mask_transform)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    print("train_len:",len(train_dataset))
    print("val_len:",len(val_dataset))

    lowest_loss = float('inf')
    count = 0
    patience_counter = 0
    writer = SummaryWriter(f"DDPM_{run_id}")
    writer.add_text("Experiment_Notes", experimental_notes, 0)
    start_time = time.time()

    for epoch in range(epochs):
        unet.train()
        running_loss = 0.0  
        mse_running_loss = 0.0
        ssim_running_loss = 0.0
        for img, mask in train_loader: 
            img, mask = img.cuda(), mask.cuda()
            img = torch.where(mask > 0.1, img, -1.0)
            img, mask = img_aug(img, mask)
            # 1. Random sampling time steps t for this batch of data 
            t = torch.randint(0, T, (img.shape[0],)).cuda()
            
            # 2. Generate random Gaussian noise
            noise = torch.randn_like(img)
            #print(img.shape, mask.shape, t.shape)
            # 3. Producing noisy images x_t
            x_t = q_sample(img, t, noise)

            # 4. Predict
            predicted_noise = unet(x_t, t, mask)
            
            # 5. Count Loss (Predicted noise vs. actual noise added)
            mse_loss = F.mse_loss(predicted_noise, noise)
            ssim_val = ssim(predicted_noise, noise, data_range=2.0, size_average=True)
            ssim_loss = 1 - ssim_val

            #ssim_alpha = 0.0
            total_loss = (1 - ssim_alpha) * mse_loss + ssim_alpha * ssim_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            mse_running_loss += mse_loss.item()
            ssim_running_loss += ssim_loss.item()
            running_loss += total_loss.item()

        with torch.no_grad():
            check_img = (img[0] + 1.0) / 2.0 
            check_mask = mask[0] 
            
            viz_grid = torch.cat([check_img, check_mask], dim=2)
            
            check_xt = (x_t[0] + 1.0) / 2.0
            print("img max:", img.max(), "img min:", img.min(), "mask max:", mask.max(), "mask min:", mask.min(), "x_t max:", x_t.max(), "x_t min:", x_t.min())
            viz_grid = torch.cat([check_img, check_mask, check_xt], dim=2)
            writer.add_image(f"Check/Img_Mask_XT", viz_grid.clamp(0, 1), epoch)
        epoch_mse_loss = mse_running_loss / len(train_loader)
        epoch_ssim_loss = ssim_running_loss / len(train_loader)
        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar("mse_loss/train_epoch", epoch_mse_loss , epoch)
        print(f"Epoch {epoch}: Train mse Loss  = {epoch_mse_loss :.6f}")
        writer.add_scalar("ssim_loss/train_epoch", epoch_ssim_loss , epoch)
        print(f"Epoch {epoch}: Train ssim Loss  = {epoch_ssim_loss :.6f}")
        writer.add_scalar("total_loss/train_epoch", epoch_loss , epoch)
        print(f"Epoch {epoch}: Train Loss  = {epoch_loss :.6f}")
        print("t:",t)
        unet.eval() 

        val_running_loss = 0.0  
        val_mse_running_loss = 0.0
        val_ssim_running_loss = 0.0

        with torch.no_grad(): 
            for img_val, mask_val in val_loader:
                img_val, mask_val = img_val.cuda(), mask_val.cuda()
                img_val = torch.where(mask_val > 0.1, img_val, -1.0)
                # The validation set also needs to be completely random.
                t_val = torch.randint(0, T, (img_val.shape[0],)).long().cuda()
                noise_val = torch.randn_like(img_val).cuda()
                
                x_t_val = q_sample(img_val, t_val, noise_val)
                val_pred_noise = unet(x_t_val, t_val, mask_val)
                
                val_mse_loss = F.mse_loss(val_pred_noise, noise_val)
                val_ssim_val = ssim(val_pred_noise, noise_val, data_range=2.0, size_average=True)
                val_ssim_loss = 1 - val_ssim_val

                #val_ssim_alpha = 0.0
                val_total_loss = (1 - ssim_alpha) * val_mse_loss + ssim_alpha * val_ssim_loss
                

                val_mse_running_loss += val_mse_loss.item()
                val_ssim_running_loss += val_ssim_loss.item()
                val_running_loss += val_total_loss.item()


        mse_val_loss = val_mse_running_loss / len(val_loader)
        ssim_val_loss = val_ssim_running_loss / len(val_loader)
        val_loss = val_running_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate = {current_lr}")

        if val_loss < lowest_loss:
                lowest_loss = val_loss
                patience_counter = 0
                count = 0
                torch.save(unet.state_dict(), f'model/{experiment_name}_lowestloss_{run_id}.pth')
                print("save!")
        else:
            patience_counter += 1
            count += 1

        if count >= early_stop:
            print("early stop!")
            break

        if patience_counter >= patience:
            print(f"Trigger a decrease in the learning rate! And revert to the optimal model state.")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= factor
            unet.load_state_dict(torch.load(f'model/{experiment_name}_lowestloss_{run_id}.pth', map_location=device))
            patience_counter = 0

        
        writer.add_scalar("mse_loss/val_epoch", mse_val_loss, epoch)
        print(f"Epoch {epoch}: Val mse Loss  = {mse_val_loss :.6f}")
        writer.add_scalar("ssim_loss/val_epoch", ssim_val_loss, epoch)
        print(f"Epoch {epoch}: Val ssim Loss  = {ssim_val_loss :.6f}")
        writer.add_scalar("total_loss/val_epoch", val_loss, epoch)
        print(f"Epoch {epoch}: Val Loss  = {val_loss :.6f}")
        writer.add_scalar("Training/Learning_Rate", current_lr, epoch)
        print("t_val:",t_val)
        now_time = time.time()
        print("time used:", now_time - start_time)

    torch.save(unet.state_dict(), f'model/{experiment_name}_{run_id}.pth')

if __name__ == '__main__':
    #training data
    #import gdown
    #gdown.download_folder("https://drive.google.com/drive/folders/1lozWOGoeAYuMmXX_QKXnb3W2gllPrLDR?usp=drive_link", quiet=False)
    diffusion_train()