import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
from PIL import Image
import sys
from math import sin, cos, pi
import DeepLKBatch as old
import time
from torchvision import models

def normalize_img_batch(img: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of images to zero mean and unit variance per channel.

    Args:
        img: Tensor of shape [N, C, H, W]

    Returns:
        Normalized image batch of same shape
    """
    N, C, H, W = img.shape

    # Flatten spatial dimensions
    img_vec = img.view(N, C, -1)

    # Compute per-image per-channel mean and std
    mean = img_vec.mean(dim=2, keepdim=True)
    std = img_vec.std(dim=2, keepdim=True)

    # Avoid division by zero
    std = std + 1e-8

    # Normalize
    img_norm = (img_vec - mean) / std

    return img_norm.view(N, C, H, W)

def H_to_param(H: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 3x3 homographies to 8D parameter vectors by subtracting identity
    and flattening the top 8 elements (ignores scale).

    Args:
        H (Tensor): [N, 3, 3] batch of homography matrices

    Returns:
        Tensor: [N, 8, 1] parameter vectors
    """
    batch_size = H.shape[0]
    device = H.device

    # Identity matrix, repeated for batch
    I = torch.eye(3, device=device).expand(batch_size, 3, 3)

    # Subtract identity to get delta from identity
    p = H - I

    # Flatten and keep first 8 parameters (drop H[2,2])
    p = p.reshape(batch_size, 9, 1)
    p = p[:, 0:8, :]

    return p

def param_to_H(p: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 8-parameter vectors to 3x3 homography matrices by adding identity.

    Args:
        p: Tensor of shape [N, 8, 1]

    Returns:
        H: Tensor of shape [N, 3, 3]
    """
    batch_size = p.size(0)
    device = p.device

    # Add last (scale) element to get 9 params
    z = torch.zeros(batch_size, 1, 1, device=device)
    # print('p size: ', p.size())
    # print('z size: ', z.size())
    p_ = torch.cat((p, z), dim=1)  # shape: [N, 9, 1]

    # Optional debug prints
    # print('p size: ', p.size())
    # print('z size: ', z.size())
    # print('p_ size: ', p_.size())

    # Identity matrix
    I = torch.eye(3, device=device).expand(batch_size, 3, 3)

    # Reshape to [N, 3, 3] and add identity
    H = p_.view(batch_size, 3, 3) + I

    return H

def meshgrid(x: torch.Tensor, y: torch.Tensor):
    """
    Create a centered meshgrid from vectors x and y.
    Args:
        x: Tensor of shape [W]
        y: Tensor of shape [H]
    Returns:
        X, Y: Meshgrid tensors of shape [H, W]
    """
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)

    x = x - x.max() / 2
    y = y - y.max() / 2

    X = x.unsqueeze(0).repeat(y.size(0), 1)  # shape [H, W]
    Y = y.unsqueeze(1).repeat(1, x.size(0))  # shape [H, W]

    return X, Y


def grid_bilinear_sampling(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    """
    Bilinear sampling of A at (x, y) locations using grid_sample.

    Args:
        A: [N, C, H, W] input feature map
        x: [N, H, W] x-coordinates to sample
        y: [N, H, W] y-coordinates to sample

    Returns:
        Q: [N, C, H, W] sampled output
        in_view_mask: [N, H, W] binary mask of valid samples
    """
    batch_size, C, H, W = A.size()
    device = A.device

    # Normalize coordinates to [-1, 1]
    x_norm = x / ((W - 1) / 2) - 1
    y_norm = y / ((H - 1) / 2) - 1

    # Stack into grid: shape [N, H, W, 2]
    grid = torch.stack((x_norm, y_norm), dim=-1)

    # Perform bilinear sampling
    Q = F.grid_sample(A, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # Compute in-view mask (inside [-1+ε, 1−ε])
    eps_w = 2 / W
    eps_h = 2 / H
    in_view_mask = (
        (x_norm > -1 + eps_w) & (x_norm < 1 - eps_w) &
        (y_norm > -1 + eps_h) & (y_norm < 1 - eps_h)
    ).to(dtype=A.dtype)

    return Q, in_view_mask

class vgg16Conv(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()

        if model_path:
            print('Loading VGG16 from:', model_path)
            vgg16 = torch.load(model_path)
        else:
            print('Loading pretrained VGG16 from torchvision...', end='')
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            print('done')

        # Extract layers up to ReLU after 3rd conv block (index 15)
        self.features = nn.Sequential(*list(vgg16.features.children())[:15])

        # Freeze early layers (e.g., conv1 and conv2)
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.Conv2d) and layer.out_channels < 256:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.features(x)
    
def warp_hmg(img: torch.Tensor, p: torch.Tensor):
    """
    Warp a batch of images using homography parameters.

    Args:
        img: [N, C, H, W] image batch
        p:   [N, 8, 1] homography parameter batch

    Returns:
        X_warp, Y_warp: warped coordinate grids [N, H, W]
    """
    device = img.device
    batch_size, C, H, W = img.size()

    # Create coordinate grid
    x = torch.arange(W, device=device)
    y = torch.arange(H, device=device)
    X, Y = meshgrid(x, y)  # both [H, W]
    
    # Homogeneous grid: [3, H*W]
    ones = torch.ones(1, X.numel(), device=device)
    xy = torch.cat([
        X.view(1, -1),
        Y.view(1, -1),
        ones
    ], dim=0)  # [3, H*W]

    # Expand to batch: [N, 3, H*W]
    xy = xy.unsqueeze(0).repeat(batch_size, 1, 1)

    # Convert p to homography matrix
    H_mat = param_to_H(p)  # returns [N, 3, 3]

    # Warp: [N, 3, H*W]
    xy_warp = H_mat.bmm(xy)

    # Normalize homogeneous coordinates
    X_warp = xy_warp[:, 0, :] / xy_warp[:, 2, :]
    Y_warp = xy_warp[:, 1, :] / xy_warp[:, 2, :]

    # Reshape and shift grid to original image frame
    X_warp = X_warp.view(batch_size, H, W) + (W - 1) / 2
    Y_warp = Y_warp.view(batch_size, H, W) + (H - 1) / 2

    img_warp, mask = grid_bilinear_sampling(img, X_warp, Y_warp)

    return img_warp, mask

class GradientBatch(nn.Module):
    def __init__(self):
        super().__init__()
        # Define Sobel-like gradient kernels
        wx = torch.tensor([[-0.5, 0.0, 0.5]], dtype=torch.float32).view(1, 1, 1, 3)
        wy = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float32).view(1, 1, 3, 1)
        
        # Register as buffers (so they move with the model to GPU if needed)
        self.register_buffer('wx', wx)
        self.register_buffer('wy', wy)

        # Replication padding for proper border handling
        self.padx_func = nn.ReplicationPad2d((1, 1, 0, 0))
        self.pady_func = nn.ReplicationPad2d((0, 0, 1, 1))

    def forward(self, img):
        batch_size, k, h, w = img.size()

        # Flatten channel dimension to apply the same kernel to each channel
        img_reshaped = img.view(batch_size * k, 1, h, w)

        # Pad and convolve in x direction
        img_padx = self.padx_func(img_reshaped)
        img_dx = F.conv2d(img_padx, self.wx).view(batch_size, k, h, w)

        # Pad and convolve in y direction
        img_pady = self.pady_func(img_reshaped)
        img_dy = F.conv2d(img_pady, self.wy).view(batch_size, k, h, w)

        return img_dx, img_dy
    

class GradientBatch(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel-like filters for x and y gradients
        wx = torch.tensor([[-0.5, 0, 0.5]], dtype=torch.float32).view(1, 1, 1, 3)
        wy = torch.tensor([[-0.5], [0], [0.5]], dtype=torch.float32).view(1, 1, 3, 1)
        self.register_buffer('wx', wx)
        self.register_buffer('wy', wy)
        self.padx = nn.ReplicationPad2d((1, 1, 0, 0))
        self.pady = nn.ReplicationPad2d((0, 0, 1, 1))

    def forward(self, img):
        batch_size, channels, height, width = img.size()
        img = img.view(batch_size * channels, 1, height, width)

        dx = F.conv2d(self.padx(img), self.wx).view(batch_size, channels, height, width)
        dy = F.conv2d(self.pady(img), self.wy).view(batch_size, channels, height, width)

        return dx, dy


class InverseBatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        assert input.size(1) == input.size(2), "Input must be square matrices"
        inv_input = torch.inverse(input)
        ctx.save_for_backward(inv_input)
        return inv_input

    @staticmethod
    def backward(ctx, grad_output):
        inv_input, = ctx.saved_tensors  # [N, h, h]
        grad_input = -inv_input.transpose(1, 2).bmm(grad_output).bmm(inv_input)
        return grad_input

def InverseBatchFun(input: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a batch of square matrices.

    Args:
        input: Tensor of shape [N, h, h]

    Returns:
        H: Tensor of shape [N, h, h], batch of inverses
    """
    assert input.size(1) == input.size(2), "Input must be a batch of square matrices"
    return torch.linalg.inv(input)  # modern, batched, differentiable

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

def corner_loss(p: torch.Tensor, p_gt: torch.Tensor, training_sz_pad: float) -> torch.Tensor:
    """
    Compute corner-based geometric loss between two sets of homography parameters.

    Args:
        p (Tensor):      [N, 8, 1] predicted warp parameters
        p_gt (Tensor):   [N, 8, 1] ground truth warp parameters
        training_sz_pad (float): Side length of the padded image region

    Returns:
        loss (Tensor): Scalar loss (sum over batch)
    """
    device = p.device
    batch_size = p.size(0)

    # Convert to homography matrices
    H_p = param_to_H(p)
    H_gt = param_to_H(p_gt)

    # Define 4 corners of square: [-pad/2, pad/2] range
    corners = torch.tensor([
        [-0.5,  0.5,  0.5, -0.5],  # x
        [-0.5, -0.5,  0.5,  0.5],  # y
        [ 1.0,  1.0,  1.0,  1.0]   # homogeneous
    ], dtype=torch.float32, device=device) * training_sz_pad  # scale to training size

    # Repeat for batch
    corners = corners.unsqueeze(0).repeat(batch_size, 1, 1)  # [N, 3, 4]

    # Warp corners with predicted and GT homographies
    warped_p = H_p.bmm(corners)    # [N, 3, 4]
    warped_gt = H_gt.bmm(corners)  # [N, 3, 4]

    # Convert from homogeneous to 2D
    warped_p = warped_p[:, :2, :] / warped_p[:, 2:3, :]
    warped_gt = warped_gt[:, :2, :] / warped_gt[:, 2:3, :]

    # Compute squared corner loss
    loss = ((warped_p - warped_gt) ** 2).sum()

    return loss

if __name__ == "__main__":
    path = "/home/user/work/sat_data/woodbridge/images/m_4007430_ne_18_1_20100829.png"
    sz = 200
    xy = [0, 0]
    sm_factor = 8
    sz_sm = sz // sm_factor
    preprocess = transforms.Compose([
    transforms.ToTensor(),
    ])
    img1 = Image.open(path).crop((xy[0], xy[1], xy[0] + sz, xy[1] + sz))
    img1_coarse = preprocess(img1.resize((sz_sm, sz_sm)))
    img1 = preprocess(img1)
    print(img1.shape, img1_coarse.shape)
    # transforms.ToPILImage()(img1).save("img1.png")
    
    scale = 1.6
    angle = 30
    projective_x = 0
    projective_y = 0
    translation_x = 0
    translation_y = 0

    rad_ang = angle / 180 * pi

    # Create transformation parameter tensor
    p = torch.tensor([
        scale + cos(rad_ang) - 2,
        -sin(rad_ang),
        translation_x,
        sin(rad_ang),
        scale + cos(rad_ang) - 2,
        translation_y,
        projective_x,
        projective_y
    ], dtype=torch.float32, requires_grad=True)  # Add requires_grad if used in optimization

    # Reshape and repeat
    p = p.view(8, 1)
    pt = p.repeat(5, 1, 1)
    print("p shape:", p.shape)
    print("pt shape:", pt.shape)
    img1 = img1.repeat(5, 1, 1, 1)  # Repeat image for batch size of 5
    print("img1 shape:", img1.shape)
    #compare normalize img batch
    img1_norm = normalize_img_batch(img1)
    img1_norm_ = old.normalize_img_batch(img1)
    print("img1_norm shape:", img1_norm.shape)
    print("img1_norm_ shape:", img1_norm_.shape)
    assert torch.allclose(img1_norm, img1_norm_, atol=1e-6), "Normalization functions do not match!"
    H = torch.eye(3).unsqueeze(0).repeat(4, 1, 1)
    from torch.autograd import Variable

    H[:, 0, 2] += 0.1  # x translation
    H[:, 1, 2] += 0.2  # y translation
    H[:, 0, 0] += 0.05 # small scale
    H_old = Variable(H.clone())
    
    p_old = old.H_to_param(H_old)
    p_new = H_to_param(H)
    assert torch.allclose(p_old, p_new, atol=1e-6), "Parameter extraction does not match!"
    print("Parameter extraction matches!")
    H_test = param_to_H(p_new)
    H_test_old = old.param_to_H(p_old)
    assert torch.allclose(H_test, H_test_old, atol=1e-6), "Homography reconstruction does not match!"
    assert torch.allclose(H, H_test, atol=1e-6), "Homography reconstruction does not match original!"
    print("Homography reconstruction matches!")
    
    VGG_MODEL_PATH = "../models/vgg16_model.pth"
    dlk = old.DeepLK(old.vgg16Conv(VGG_MODEL_PATH))
    x_old= dlk.inv_func.apply(old.param_to_H(pt))
    print("pt: ", pt.size())
    print("param_to_H(pt): ", old.param_to_H(pt).size())
    print('x_old size: ', x_old.size())
    
    x_new = InverseBatch.apply(param_to_H(pt))
    print('x_new size: ', x_new.size())
    assert torch.allclose(x_old, x_new, atol=1e-6), "Inverse batch computation does not match!"
    print("Inverse batch computation matches!")
    
    batch_of_wart_params = H_to_param(x_new)
    batch_of_wart_params_old = old.H_to_param(x_old)
    assert torch.allclose(batch_of_wart_params, batch_of_wart_params_old, atol=1e-6), "Parameter extraction from inverse does not match!"    
    print("Parameter extraction from inverse matches!")
    #test warp
    img_warp_old, mask_old = old.warp_hmg(img1, batch_of_wart_params_old)
    img_warp, mask = warp_hmg(img1, batch_of_wart_params)
    print("img_warp_old shape:", img_warp_old.shape)
    from torchvision.transforms.functional import to_pil_image

    # print("img_warp shape:", img_warp.shape)
    # for i in range(img_warp.size(0)):
    #     img_i = mask_old[i].cpu().clamp(0, 1)
    #     img_pil = to_pil_image(img_i)
    #     img_pil.save(f"warped_output_{i}.png")
        
    # # for i in range(img_warp_old.size(0)):
    # #     img_i = img_warp[i].cpu().clamp(0, 1)
    # #     img_pil = to_pil_image(img_i)
    # #     img_pil.save(f"warped_output_{i}.png")
    VGG_MODEL_PATH = "../models/vgg16_model.pth"
    dlk = DeepLK(vgg16Conv(VGG_MODEL_PATH))
    # copy img1 to img2
    img2 = img1.clone()
    img1_coarse = img1_coarse.repeat(5, 1, 1, 1)  # Repeat for batch size of 5
    img2_coarse = img1_coarse.clone()
    
    wimg2, _ = warp_hmg(img2, H_to_param(dlk.inv_func(param_to_H(pt))))
    wimg2_n = normalize_img_batch(wimg2)
    wimg2_coarse, _ = warp_hmg(img2_coarse, H_to_param(dlk.inv_func(param_to_H(pt))))
    
    img1_coarse_n = normalize_img_batch(img1_coarse)
    wimg2_coarse_n = normalize_img_batch(wimg2_coarse)
    
    print("start conv...")
    start = time.time()
    p_lk_conv, H_conv = dlk(wimg2_n, img1_norm, tol=1e-4, max_itr=200, conv_flag=1)
    stop = time.time()
    print(f"Conv time: {stop - start:.4f} seconds")
    start = time.time()
    print("start raw...")
    p_lk, H = dlk(wimg2_coarse_n, img1_coarse_n, tol=1e-4, max_itr=200, conv_flag=0)
    stop = time.time()
    print(f"Raw time: {stop - start:.4f} seconds")
    
    
    print((p_lk_conv[0,:,:]-pt[0,:,:]).norm())
    print((p_lk[0,:,:]-pt[0,:,:]).norm())
    print(H_conv)
    print(H)
    warp_pad = 0.4

    # normalized size of all training pairs
    training_sz = 175
    training_sz_pad = round(training_sz + training_sz * 2 * warp_pad)
    conv_loss = corner_loss(p_lk_conv, pt, training_sz_pad)
    raw_loss = corner_loss(p_lk, pt, training_sz_pad)
    print("Conv loss:", conv_loss.item())
    print("Raw loss:", raw_loss.item())
    # show device
    print("Device:", img1.device)
    print("conv shape", conv_loss.shape)
    print("raw shape", raw_loss.shape)
    x = torch.rand(10, 8, 8, requires_grad=True)
    y = InverseBatch.apply(x)
    loss = y.sum()
    loss.backward()
    print(x.grad.shape)  # Should be [10, 8, 8]
