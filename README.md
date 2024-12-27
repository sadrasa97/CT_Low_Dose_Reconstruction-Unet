
---

# CT Low Dose Reconstruction with U-Net

This project aims to reconstruct high-quality CT images from low-dose CT scans using a U-Net-based architecture. The implementation uses TensorFlow and evaluates the reconstruction quality using metrics like Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR).

## Features

- **Data Handling**: Automatically downloads and preprocesses the dataset.
- **Deep Learning Model**: Implements a U-Net architecture for image reconstruction.
- **Custom Loss Function**: Uses SSIM as the loss function to ensure perceptual quality.
- **Evaluation Metrics**: Calculates SSIM and PSNR to measure reconstruction performance.
- **Visualization**: Displays original and reconstructed images for qualitative evaluation.

## Dataset

The dataset is sourced from [Kaggle: CT Low Dose Reconstruction](https://www.kaggle.com/datasets/andrewmvd/ct-low-dose-reconstruction). The notebook downloads and extracts the dataset automatically.

## Installation

Ensure you have the following installed:
- Python 3.x
- Jupyter Notebook
- Required Python libraries (`TensorFlow`, `OpenCV`, `NumPy`, `Matplotlib`, etc.)

Install the dependencies:
```bash
pip install tensorflow numpy matplotlib opencv-python lpips torch-fidelity kaggle
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ct-low-dose-reconstruction.git
    cd ct-low-dose-reconstruction
    ```

2. Open the Jupyter Notebook:
    ```bash
    jupyter notebook CT_Low_Dose_Reconstruction.ipynb
    ```

3. Follow the steps in the notebook to:
    - Download and preprocess the dataset.
    - Train the U-Net model.
    - Evaluate the model using SSIM and PSNR.
    - Visualize the results.

## Model Architecture

The U-Net architecture consists of:
- **Encoder**: Extracts hierarchical features.
- **Bottleneck**: Encodes the latent representation.
- **Decoder**: Reconstructs the high-quality CT image.

![U-Net Diagram](https://raw.githubusercontent.com/yourusername/ct-low-dose-reconstruction/main/unet_diagram.png)

## Evaluation

The reconstruction quality is evaluated using:
- **SSIM**: Structural Similarity Index
- **PSNR**: Peak Signal-to-Noise Ratio

## Results

| Metric | Value   |
|--------|---------|
| SSIM   | 0.9986  |
| PSNR   | 40.83 dB |

Example reconstructed images are shown in the notebook.

## Acknowledgments

- [Kaggle Dataset: CT Low Dose Reconstruction](https://www.kaggle.com/datasets/andrewmvd/ct-low-dose-reconstruction)
- TensorFlow for the deep learning framework.

---
