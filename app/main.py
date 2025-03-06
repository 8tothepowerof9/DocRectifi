import os
import sys
import json
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import torch.nn.functional as F
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from illum.model import DRNet, GCNet
from rectifi.model import UVDocnet
from rectifi.utils import bilinear_unwarping
from illum.utils import pad_to_stride


# Ensure we find the config and model
@st.cache_resource
def load_model():
    """Load UVDocNet, DRNet & GCNet models"""
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Moves up from "app/"
    cfg_path = os.path.join(base_dir, "configs", "illum", "drnet.json")
    dr_model_path = os.path.join(base_dir, "checkpoints", "illum", "drnet.pt")
    gc_model_path = os.path.join(base_dir, "checkpoints", "illum", "gcnet.pt")
    uvdoc_model_path = os.path.join(base_dir, "checkpoints", "best", "uvdocnet.pkl")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    
    if not os.path.exists(dr_model_path):
        raise FileNotFoundError(f"Model file not found: {dr_model_path}")
    
    if not os.path.exists(uvdoc_model_path):
        raise FileNotFoundError(f"UVDocNet model file not found: {uvdoc_model_path}")

    # Load config
    with open(cfg_path, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load DRNet model
    dr = DRNet(config).to(device)
    dr_checkpoint = torch.load(dr_model_path, weights_only=True)
    dr.load_state_dict(dr_checkpoint)
    dr.eval()

    # Load GCNet model
    gc = GCNet(config).to(device)
    gc_checkpoint = torch.load(gc_model_path, weights_only=True)
    gc.load_state_dict(gc_checkpoint)
    gc.eval()

    # Load UVDocNet model
    uvdoc = UVDocnet(num_filter=32, kernel_size=5).to(device)
    uvdoc_checkpoint = torch.load(uvdoc_model_path, weights_only=True)
    uvdoc.load_state_dict(uvdoc_checkpoint["model_state"])
    uvdoc.to("cuda")
    uvdoc.eval()

    return uvdoc, dr, gc, device

# Load the models
uvdoc_model, dr_model, gc_model, device = load_model()


def enhance_image(image):
    """Extract document using UVDocNet, then enhance using DRNet & GCNet"""

    img = np.array(image.convert("RGB")).astype(np.float32) / 255  # Ensure float32

    # Convert to tensor for UVDocNet
    img_size = [488, 712]  # Required input size
    inp = torch.from_numpy(cv2.resize(img, tuple(img_size)).transpose(2, 0, 1)).unsqueeze(0).to(device)

    # Run UVDocNet to extract document
    with torch.no_grad():
        point_positions2D, _ = uvdoc_model(inp)

    # Unwarp document
    size = img.shape[:2][::-1]  # (width, height)
    unwarped = bilinear_unwarping(
        warped_img=torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device),
        point_positions=torch.unsqueeze(point_positions2D[0], dim=0),
        img_size=tuple(size),
    )

    # Pad to stride 32 for GCDRNet
    gc_input = unwarped[0].cpu().numpy().transpose(1, 2, 0)
    gc_input, _, _ = pad_to_stride(gc_input, stride=32)
    
    # Save original size for upscaling
    original_size = gc_input.shape[:2]

    # Convert to tensor
    gc_input = torch.from_numpy(gc_input.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        # Reshape to 512x512 for GCNet
        gc_input_down = F.interpolate(gc_input, size=(512, 512), mode="bilinear")
        pred_shadow_map = gc_model(gc_input_down)
        # Upsample to original size
        pred_shadow_map = F.interpolate(pred_shadow_map, size=original_size, mode="bilinear")
    
    i_gc = torch.clamp(gc_input / pred_shadow_map, 0, 1)
    dr_input = torch.cat((gc_input, i_gc), dim=1)

    with torch.no_grad():
        _, _, _, enhanced = dr_model(dr_input)
    
    # Convert tensors to Image
    unwarped = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    unwarped = Image.fromarray(unwarped)

    enhanced = (enhanced[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    enhanced = Image.fromarray(enhanced)

    i_gc = (i_gc[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    i_gc = Image.fromarray(i_gc)

    return unwarped, enhanced, i_gc

st.title("📄 Document Rectification App")
st.write("Upload a raw document photo to extract and enhance its quality.")

# Upload image
uploaded_file = st.file_uploader("Choose a document image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2, col3 = st.columns(3)

    # Show original image
    image = Image.open(uploaded_file)
    col1.header("Original Image")
    col1.image(image, use_container_width=True)

    # Process image
    with st.spinner("Extracting & Enhancing document..."):
        extracted_image, enhanced_image, i_gc = enhance_image(image)

    # Show extracted document (UVDocNet output)
    col2.header("Extracted Document")
    col2.image(extracted_image, use_container_width=True)

    # Show enhanced image (DRNet output)
    col3.header("Enhanced Document")
    col3.image(enhanced_image, use_container_width=True)

    # Provide download button for enhanced image
    st.download_button(
        label="Download Enhanced Image",
        data=enhanced_image.tobytes(),
        file_name="enhanced_document.png",
        mime="image/png",
    )
