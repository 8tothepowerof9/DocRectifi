import torch
import torch.nn.functional as F


def bilinear_unwarping(warped_img, point_positions, img_size):
    """
    Utility function that unwarps an image.
    Unwarp warped_img based on the 2D grid point_positions with a size img_size.
    Args:
        warped_img  :       torch.Tensor of shape BxCxHxW (dtype float)
        point_positions:    torch.Tensor of shape Bx2xGhxGw (dtype float)
        img_size:           tuple of int [w, h]
    """
    upsampled_grid = F.interpolate(
        point_positions,
        size=(img_size[1], img_size[0]),
        mode="bilinear",
        align_corners=True,
    )
    unwarped_img = F.grid_sample(
        warped_img, upsampled_grid.transpose(1, 2).transpose(2, 3), align_corners=True
    )

    return unwarped_img
