# DocRectify

## Overview
DocRectify is a document image processing pipeline that consists of two primary functionalities:

1. **Document Rectification** - Corrects the perspective distortion of a document within an image.
2. **Illumination Correction** - Enhances the document image by correcting uneven lighting conditions.

To demonstrate these functionalities, a Streamlit-based web application has been developed.

## Chosen Models
After reviewing multiple models designed for these tasks, the following models were selected based on their efficiency and suitability for limited computational resources (RTX 4050 GPU):

- **UVDocNet** for document rectification ([Paper](https://arxiv.org/abs/2302.02887))
- **GCDRNet** for illumination correction ([Paper](https://ieeexplore.ieee.org/document/10268585))

UVDocNet was chosen because the full implementation was publicly available, while GCDRNet had to be implemented from scratch due to the lack of publicly available training code.

## Implementation Details
### GCDRNet Implementation
Since the full training code for GCDRNet was not available, the implementation was done from scratch. The model structure was straightforward as the architecture was provided in the paper, but training posed significant challenges:

1. **Dataset Handling**:
   - The dataset contained images of varying sizes, while the model required full-sized images.
   - To accommodate batch training, a custom batch sampler was implemented to handle different image sizes efficiently.

2. **Optimizing Training for Limited Resources**:
   - The paper used full-sized images, but due to GPU memory constraints, the longest image dimension was restricted to **2000 pixels** while maintaining the aspect ratio.
   - A batch size of **1** was used to fit within memory constraints.
   - All preprocessing was performed within the dataset pipeline instead of in the training loop to reduce memory usage.

3. **Training Results**:
   - Achieved an average **PSNR of 22** and **SSIM of 0.91** on the test set.

### UVDocNet Training and Testing
While attempting to train UVDocNet, it was found that the dataset (120k images) exceeded the computational resources available. As a result, the best pre-trained model provided in the repository was used instead.

Upon testing, it was observed that **GCDRNet performs better when provided with rectified document images**. This behavior is likely due to the dataset used for illumination correction, which primarily consisted of already rectified images.

## Summary
- Successfully re-implemented **GCDRNet** from scratch and trained it with resource-efficient modifications.
- Used the pre-trained **UVDocNet** model due to dataset size limitations.
- Discovered that **GCDRNet works best on rectified document images**, making **UVDocNet a necessary pre-processing step** for optimal illumination correction.

## Future Improvements
- Explore alternative training strategies for UVDocNet with a subset of data to fine-tune the model.
- Experiment with further optimizations for training efficiency, such as mixed precision training.
- Improve the Streamlit app for better user experience and real-time inference.

## How to Run the Streamlit App
1. Clone the repository:
   ```bash
   cd DocRectify
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```