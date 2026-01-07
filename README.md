# Vegetable Quality Inspection using Autoencoder

This project uses a **Convolutional Autoencoder** to detect the quality of vegetables. The idea is simple: train the autoencoder on images of fresh vegetables, then use reconstruction error to determine if new vegetables are fresh or not usable.  

## Project Overview

- **Training**: The autoencoder learns to compress and reconstruct fresh vegetable images.  
- **Threshold Calculation**: Compute a reconstruction error threshold on validation images to identify anomalies.  
- **Inference**: Compare reconstruction error of test images to the threshold to classify vegetables as *Fresh* or *Not Usable*.  

This approach is useful for automated quality inspection in agriculture or food processing.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/mztpaswan/vegetable-quality-inspection.git
cd vegetable-quality-inspection

pip install -r requirements.txt

```

## How to Use
- **Train the Autoencoder**: This trains the model on your training images and saves it as models/autoencoder_fresh_veggies.pth.
```bash
  python src/train.py
```
 - **Compute Anomaly Threshold**: This calculates the mean reconstruction error from validation images and saves the threshold to threshold.npy.
```bash
  python src/threshold.py
```
- **Run Inference**: The script predicts whether each test image is Fresh or Not Usable, based on the threshold.
```bash
  python src/inference.py
```
## Author

**Manjeet Kumar Paswan**  

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

