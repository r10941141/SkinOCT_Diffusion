import torchvision
import torch.nn as nn
import torch
import math


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)
    
class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1_layer = DoubleConv(1, 16, 0.1) 
        self.p1_layer = nn.MaxPool2d(2)
        self.c2_layer = DoubleConv(16, 32, 0.1) 
        self.p2_layer = nn.MaxPool2d(2)
        self.c3_layer = DoubleConv(32, 64, 0.2) 
        self.p3_layer = nn.MaxPool2d(2)
        self.c4_layer = DoubleConv(64, 128, 0.2) 
        self.p4_layer = nn.MaxPool2d(2)
        self.c5_layer = DoubleConv(128, 256, 0.3)
        self.u6_layer = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c6_layer = DoubleConv(256, 128, 0.2) 
        self.u7_layer = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c7_layer = DoubleConv(128, 64, 0.2) 
        self.u8_layer = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.c8_layer = DoubleConv(64, 32, 0.1) 
        self.u9_layer = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.c9_layer = DoubleConv(32, 16, 0.1) 
        self.output_layer = nn.Conv2d(16, 1, kernel_size=1)
        self.output_layer_sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1 = self.c1_layer(x)
        p1 = self.p1_layer(c1)
        c2 = self.c2_layer(p1)
        p2 = self.p2_layer(c2)
        c3 = self.c3_layer(p2)
        p3 = self.p3_layer(c3)
        c4 = self.c4_layer(p3)
        p4 = self.p4_layer(c4)
        c5 = self.c5_layer(p4)
        u6 = self.u6_layer(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.c6_layer(u6)
        u7 = self.u7_layer(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.c7_layer(u7)
        u8 = self.u8_layer(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.c8_layer(u8)
        u9 = self.u9_layer(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.c9_layer(u9)
        outputs = self.output_layer(c9)
        outputs = self.output_layer_sigmoid(outputs)
        return outputs
    
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t, out_dim):
        # t: (batch_size,)
        half_dim = out_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.mlp(emb)
    

class Unet_Diffusion(nn.Module):
    def __init__(self, time_dim=256):
        super().__init__()
        # 1. MLP for time series
        self.time_mlp = TimeEmbedding(time_dim)
        
        # 2. channel: image(1) + Mask(1) = 2 
        self.c1_layer = DoubleConv(2, 16, 0.1) 
        self.p1_layer = nn.MaxPool2d(2)
        
        # Define the time linear projection for each layer (mapping time_dim to the feature dimension of that layer)
        self.t1_proj = nn.Linear(time_dim, 16)
        self.t2_proj = nn.Linear(time_dim, 32)
        self.t3_proj = nn.Linear(time_dim, 64)
        self.t4_proj = nn.Linear(time_dim, 128)
        self.t5_proj = nn.Linear(time_dim, 256)

        # (The intermediate layers and decoding layers c2~c9 should remain similar to original Unet, but the corresponding time projection layers need to be added)
        #self.c1_layer = DoubleConv(1, 16, 0.1) 
        self.p1_layer = nn.MaxPool2d(2)
        self.c2_layer = DoubleConv(16, 32, 0.1) 
        self.p2_layer = nn.MaxPool2d(2)
        self.c3_layer = DoubleConv(32, 64, 0.2) 
        self.p3_layer = nn.MaxPool2d(2)
        self.c4_layer = DoubleConv(64, 128, 0.2) 
        self.p4_layer = nn.MaxPool2d(2)
        self.c5_layer = DoubleConv(128, 256, 0.3)
        self.u6_layer = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c6_layer = DoubleConv(256, 128, 0.2) 
        self.u7_layer = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c7_layer = DoubleConv(128, 64, 0.2) 
        self.u8_layer = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.c8_layer = DoubleConv(64, 32, 0.1) 
        self.u9_layer = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.c9_layer = DoubleConv(32, 16, 0.1) 

        # The diffusion model predicts a Gaussian function, so do not use sigmoid.
        self.output_layer = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x, t, mask):
        # x: image with noise (B, 1, H, W)
        # t: time step (B,)
        # mask: (B, 1, H, W)
        
        # 1. splicing
        x = torch.cat([x, mask], dim=1) # (B, 2, H, W)
        
        # 2. MLP for time series
        t_emb = self.time_mlp(t, 256) # set time_dim=256
        
        # 3. encode
        c1 = self.c1_layer(x)
        c1 = c1 + self.t1_proj(t_emb)[:, :, None, None] # Broadcast   
        p1 = self.p1_layer(c1)

        c2 = self.c2_layer(p1)
        c2 = c2 + self.t2_proj(t_emb)[:, :, None, None]
        p2 = self.p2_layer(c2)

        c3 = self.c3_layer(p2)
        c3 = c3 + self.t3_proj(t_emb)[:, :, None, None]
        p3 = self.p3_layer(c3)

        c4 = self.c4_layer(p3)
        c4 = c4 + self.t4_proj(t_emb)[:, :, None, None]
        p4 = self.p4_layer(c4)

        c5 = self.c5_layer(p4)
        c5 = c5 + self.t5_proj(t_emb)[:, :, None, None]

        # 4. decode
        u6 = self.u6_layer(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.c6_layer(u6)
        u7 = self.u7_layer(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.c7_layer(u7)
        u8 = self.u8_layer(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.c8_layer(u8)
        u9 = self.u9_layer(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.c9_layer(u9)
        outputs = self.output_layer(c9)
        #outputs = self.output_layer_sigmoid(outputs)
        return outputs        
    
if __name__ == '__main__':
    unet = Unet()
    input = torch.ones(64, 1, 512, 512)
    output = unet(input)
    print(output.shape)