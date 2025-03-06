import sys
import os
import torch
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from rectifi.utils import bilinear_unwarping
from rectifi.model import UVDocnet

if __name__ == "__main__":
    # Read image path with os
    img_path = "tests/rectifi/test.jpg"

    model = UVDocnet(num_filter=32, kernel_size=5)
    checkpoint_path = f"checkpoints/best/uvdocnet.pkl"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to("cuda")
    model.eval()

    # Load image
    img_size = [488, 712]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    inp = torch.from_numpy(cv2.resize(img, img_size).transpose(2, 0, 1)).unsqueeze(0)

    # Make pred
    inp = inp.to("cuda")
    point_positions2D, _ = model(inp)

    # Unwarp
    size = img.shape[:2][::-1]
    unwarped = bilinear_unwarping(
        warped_img=torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to("cuda"),
        point_positions=torch.unsqueeze(point_positions2D[0], dim=0),
        img_size=tuple(size),
    )
    unwarped = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(
        np.uint8
    )

    # Display result
    unwarped_BGR = cv2.cvtColor(unwarped, cv2.COLOR_RGB2BGR)
    cv2.imshow("Unwarped", unwarped_BGR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()