import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import numpy as np
import os, sys

root_path = os.path.abspath('.')
sys.path.append(root_path)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super(SEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        dim = (2, 3)

        if 'Half' in x.type():
            x0 = flow.mean(x.float(), dim=dim, keepdim=True).half()
        else:
            x0 = flow.mean(x, dim=dim, keepdim=True)

        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = F.sigmoid(x0)
        x = flow.mul(x, x0)
        return x
    
    def forward_mean(self, x, x0):
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = F.sigmoid(x0)
        x = flow.mul(x, x0)
        return x

class UNetConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, se):
        super(UNetConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True)
        )

        if se:
            self.seblock = SEBlock(out_channels, 8, True)
        else:
            self.seblock = None
    
    def forward(self, x):
        z = self.conv(x)
        if self.seblock is not None:
            z = self.seblock(z)
        return z

class UNet1(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z
    
    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2
    
    def forward_b(self, x1, x2):
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

class UNet1x3(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1x3, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 5, 3, 2)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x1, x2):
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet2, self).__init__()

        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)

        x3 = self.conv2_down(x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)

        x2 = F.pad(x2, (-4, -4, -4, -4))
        x4 = self.conv4(x2 + x3)
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)

        x1 = F.pad(x1, (-16, -16, -16, -16))
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)

        z = self.conv_bottom(x5)
        return z

    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1, x2

    def forward_b(self, x2):
        x3 = self.conv2_down(x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3.conv(x3)
        return x3

    def forward_c(self, x2, x3):
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)

        x2 = F.pad(x2, (-4, -4, -4, -4))
        x4 = self.conv4.conv(x2 + x3)
        return x4

    def forward_d(self, x1, x4):
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)

        x1 = F.pad(x1, (-16, -16, -16, -16))
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)

        z = self.conv_bottom(x5)
        return z

class UpCunet(nn.Module):
    def __init__(self, scale_factor, in_channels=3, out_channels=3):
        super(UpCunet, self).__init__()
        self.scalef = scale_factor

        self.unet1 = UNet1(in_channels, out_channels if scale_factor == 2 else 64, deconv=True)
        self.unet2 = UNet2(
            in_channels if scale_factor == 2 else 64,
            out_channels if scale_factor == 2 else 64,
            deconv=False
        )

        if scale_factor == 4:
            self.ps = nn.PixelShuffle(2)
            self.conv_final = nn.Conv2d(64, 12, 3, 1, padding=0, bias=True)
    
    def forward(self, x, tile_mode):
        n, c, h0, w0 = x.shape
        x00 = x

        if tile_mode == 0:
            ph = ((h0 - 1) // 2 + 1) * 2
            pw = ((w0 - 1) // 2 + 1) * 2
            mx = 18 if self.scalef == 2 else 19
            x = F.pad(x, (mx, mx + pw - w0, mx, mx + ph - h0), 'reflect')
            x = self.unet1.forward(x)
            x0 = self.unet2.forward(x)
            x1 = F.pad(x, (-20, -20, -20, -20))
            x = flow.add(x0, x1)

            if self.scalef == 4:
                x = self.conv_final(x)
                x = F.pad(x, (-1, -1, -1, -1))
                x = self.ps(x)
            
            if (w0 != pw or h0 != ph):
                x = x[:, :, :h0 * self.scalef, :w0 * self.scalef]
            
            if self.scalef == 4:
                x += F.interpolate(x00, scale_factor=4, mode='nearest')
            
            return x
        elif (tile_mode == 1):
            if (w0 >= h0):	
                crop_size_w = ((w0 - 1) // 4 * 4 + 4) // 2
                crop_size_h = (h0 - 1) // 2 * 2 + 2
            else:	
                crop_size_h = ((h0 - 1) // 4 * 4 + 4) // 2
                crop_size_w = (w0 - 1) // 2 * 2 + 2
            crop_size = (crop_size_h, crop_size_w)
        elif (tile_mode == 2):
            crop_size = (((h0 - 1) // 4 * 4 + 4) // 2, ((w0 - 1) // 4 * 4 + 4) // 2)
        elif (tile_mode == 3):
            crop_size = (((h0 - 1) // 6 * 6 + 6) // 3, ((w0 - 1) // 6 * 6 + 6) // 3)
        elif (tile_mode == 4):
            crop_size = (((h0 - 1) // 8 * 8 + 8) // 4, ((w0 - 1) // 8 * 8 + 8) // 4)

        ph = ((h0 - 1) // crop_size[0] + 1) * crop_size[0]	
        pw = ((w0 - 1) // crop_size[1] + 1) * crop_size[1]
        mx = 18 if self.scalef == 2 else 19
        x = F.pad(x, (mx, mx + pw - w0, mx, mx + ph - h0), 'reflect')
        n, c, h, w = x.shape

        se_mean0 = flow.zeros((n, 64, 1, 1)).to(x.device)
        n_patch = 0
        tmp_dict = {}
        opt_res_dict = {}
        scale_range = 36 if self.scalef == 2 else 38

        for i in range(0, h - scale_range, crop_size[0]):
            tmp_dict[i] = {}
            for j in range(0, w - scale_range, crop_size[1]):
                x_crop = x[:, :, i:i + crop_size[0] + scale_range, j:j + crop_size[1] + scale_range]
                n, c1, h1, w1 = x_crop.shape
                tmp0, x_crop = self.unet1.forward_a(x_crop)
                tmp_se_mean = flow.mean(x_crop, dim=(2, 3), keepdim=True)
                se_mean0 += tmp_se_mean
                n_patch += 1
                tmp_dict[i][j] = (tmp0, x_crop)
        
        se_mean0 /= n_patch
        se_mean1 = flow.zeros((n, 128, 1, 1)).to(x.device)

        for i in range(0, h - scale_range, crop_size[0]):
            for j in range(0, w - scale_range, crop_size[1]):
                tmp0, x_crop = tmp_dict[i][j]
                x_crop = self.unet1.conv2.seblock.forward_mean(x_crop, se_mean0)
                opt_unet1 = self.unet1.forward_b(tmp0, x_crop)
                tmp_x1, tmp_x2 = self.unet2.forward_a(opt_unet1)
                tmp_se_mean = flow.mean(tmp_x2, dim=(2, 3), keepdim=True)
                se_mean1 += tmp_se_mean
                tmp_dict[i][j] = (opt_unet1, tmp_x1, tmp_x2)
        
        se_mean1 /= n_patch
        se_mean0 = flow.zeros((n, 128, 1, 1)).to(x.device)

        for i in range(0, h - scale_range, crop_size[0]):
            for j in range(0, w - scale_range, crop_size[1]):
                opt_unet1, tmp_x1, tmp_x2 = tmp_dict[i][j]
                tmp_x2 = self.unet2.conv2.seblock.forward_mean(tmp_x2, se_mean1)
                tmp_x3 = self.unet2.forward_b(tmp_x2)
                tmp_se_mean = flow.mean(tmp_x3, dim=(2, 3), keepdim=True)
                se_mean0 += tmp_se_mean
                tmp_dict[i][j] = (opt_unet1, tmp_x1, tmp_x2, tmp_x3)
        
        se_mean0 /= n_patch
        se_mean1 = flow.zeros((n, 64, 1, 1)).to(x.device)

        for i in range(0, h - scale_range, crop_size[0]):
            for j in range(0, w - scale_range, crop_size[1]):
                opt_unet1, tmp_x1, tmp_x2, tmp_x3 = tmp_dict[i][j]
                tmp_x3 = self.unet2.conv3.seblock.forward_mean(tmp_x3, se_mean0)
                tmp_x4 = self.unet2.forward_c(tmp_x2, tmp_x3)
                tmp_se_mean = flow.mean(tmp_x4, dim=(2, 3), keepdim=True)
                se_mean1 += tmp_se_mean
                tmp_dict[i][j] = (opt_unet1, tmp_x1, tmp_x4)
        
        se_mean1 /= n_patch

        for i in range(0, h - scale_range, crop_size[0]):
            opt_res_dict[i] = {}
            for j in range(0, w - scale_range, crop_size[1]):
                opt_unet1, tmp_x1, tmp_x4 = tmp_dict[i][j]
                tmp_x4 = self.unet2.conv4.seblock.forward_mean(tmp_x4, se_mean1)
                x0 = self.unet2.forward_d(tmp_x1, tmp_x4)
                x1 = F.pad(opt_unet1, (-20, -20, -20, -20))
                x_crop = flow.add(x0, x1)
                opt_res_dict[i][j] = x_crop
        
        del tmp_dict
        flow.cuda.empty_cache()

        cal = 72 if self.scalef == 2 else 152
        res = flow.zeros((n, c, h * self.scalef - cal, w * self.scalef - cal)).to(x.device)

        for i in range(0, h - scale_range, crop_size[0]):
            for j in range(0, w - scale_range, crop_size[1]):
                res[:, :, i * self.scalef:i * self.scalef + h1 * self.scalef - cal, j * self.scalef:j * self.scalef + w1 * self.scalef - cal] = opt_res_dict[i][j]
        
        del opt_res_dict
        flow.cuda.empty_cache()

        if (w0 != pw or h0 != ph):
            res = res[:, :, :h0 * self.scalef, :w0 * self.scalef]
        
        if self.scalef == 4:
            res += F.interpolate(x00, scale_factor=4, mode='nearest')
        
        return res

class RealWaifuUpScaler(object):
    def __init__(self, scalef, weight_path, half, device):
        weight = flow.load(weight_path)

        self.scalef = scalef
        self.model = eval('UpCunet')(scale_factor=scalef)

        if half:
            self.model = self.model.half().to(device)
        else:
            self.model = self.model.to(device)
        
        self.model.load_state_dict(weight)
        self.model.eval()
        self.half = half
        self.device = device
    
    def np2tensor(self, frame):
        if not self.half:
            return flow.from_numpy(np.transpose(frame, (2, 0, 1))).unsqueeze(0).to(self.device).float() / 255
        else:
            return flow.from_numpy(np.transpose(frame, (2, 0, 1))).unsqueeze(0).to(self.device).half() / 255
    
    def tensor2np(self, tensor):
        if not self.half:
            return (np.transpose((tensor.data.squeeze() * 255.0).round().clamp_(0, 255).to(flow.uint8).cpu().numpy(), (1, 2, 0)))
        else:
            return (np.transpose((tensor.data.squeeze().float() * 255.0).round().clamp_(0, 255).byte().cpu().numpy(), (1, 2, 0)))
    
    def __call__(self, frame, tile):
        with flow.no_grad():
            tensor = self.np2tensor(frame)
            result = self.tensor2np(self.model(tensor, tile))

        return result