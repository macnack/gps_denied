import torch
import torch.nn as nn
from core.inverse import InverseBatch
from models.gradient import GradientBatch
from models.feature_extractors import vgg16Conv
from core.geometry import param_to_H, warp_hmg, meshgrid


class DeepLK(nn.Module):
    def __init__(self, conv_net):
        super().__init__()
        self.img_gradient_func = GradientBatch()
        self.conv_func = conv_net
        self.inv_func = InverseBatch.apply  # Use .apply for torch.autograd.Function

    def forward(self, img, temp, init_param=None, tol=1e-3, max_itr=500, conv_flag=0, ret_itr=False):
        device = img.device

        # Feature extraction (or raw pixels)
        if conv_flag:
            # start = time.time()
            Ft = self.conv_func(temp)
            # stop = time.time()
            Fi = self.conv_func(img)
            # print(f"Feature extraction time: {stop - start:.4f} seconds")
        else:
            Fi = img
            Ft = temp

        batch_size, k, h, w = Ft.size()

        Ftgrad_x, Ftgrad_y = self.img_gradient_func(Ft)
        dIdp = self.compute_dIdp(Ftgrad_x, Ftgrad_y, device)
        dIdp_t = dIdp.transpose(1, 2)

        invH = self.inv_func(dIdp_t.bmm(dIdp))
        invH_dIdp = invH.bmm(dIdp_t)

        # Initialize p
        if init_param is None:
            p = torch.zeros(batch_size, 8, 1, device=device)
        else:
            p = init_param.to(device)

        # Make sure dp is large enough initially to enter loop
        dp = torch.ones(batch_size, 8, 1, device=device)

        itr = 1
        while (dp.norm(p=2, dim=1, keepdim=True).max() > tol or itr == 1) and (itr <= max_itr):
            Fi_warp, mask = warp_hmg(Fi, p)

            mask = mask.unsqueeze(1).repeat(1, k, 1, 1)
            Ft_mask = Ft * mask

            r = (Fi_warp - Ft_mask).view(batch_size, k * h * w, 1)
            dp_new = invH_dIdp.bmm(r)
            dp_new[:, 6:8, 0] = 0  # zero out projective components

            # Mask updates based on convergence threshold
            update_mask = (dp.norm(p=2, dim=1, keepdim=True) > tol).float().to(device)
            dp = update_mask * dp_new
            p = p - dp
            itr += 1

        # print('finished at iteration', itr)

        H = param_to_H(p)
        return (p, H, itr) if ret_itr else (p, H)

    def compute_dIdp(self, Ftgrad_x, Ftgrad_y, device):
        batch_size, k, h, w = Ftgrad_x.size()
        x = torch.arange(w, device=device)
        y = torch.arange(h, device=device)

        X, Y = meshgrid(x, y)  # [H, W]
        X = X.flatten().unsqueeze(1)  # [H*W, 1]
        Y = Y.flatten().unsqueeze(1)

        X = X.repeat(batch_size, k, 1)
        Y = Y.repeat(batch_size, k, 1)

        Ftgrad_x = Ftgrad_x.view(batch_size, k * h * w, 1)
        Ftgrad_y = Ftgrad_y.view(batch_size, k * h * w, 1)

        dIdp = torch.cat((
            X * Ftgrad_x,
            Y * Ftgrad_x,
            Ftgrad_x,
            X * Ftgrad_y,
            Y * Ftgrad_y,
            Ftgrad_y,
            -X * X * Ftgrad_x - X * Y * Ftgrad_y,
            -X * Y * Ftgrad_x - Y * Y * Ftgrad_y
        ), dim=2)

        return dIdp