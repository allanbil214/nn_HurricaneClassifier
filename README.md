# Hurricane Damage Image Classifier

A Convolutional Neural Network (CNN) project for classifying satellite images of buildings affected by hurricanes. The model distinguishes between images of **damaged** and **non-damaged** structures, aiding disaster response teams in quickly identifying areas that need urgent attention.

## Project Overview

This project leverages deep learning techniques to build a binary classifier that processes satellite images and identifies whether a building has sustained damage post-hurricane. The model achieves high accuracy by utilizing convolutional layers to extract essential features and a final sigmoid-activated neuron for binary classification.

## Dataset Information

- **Source**: [IEEE DataPort](https://ieee-dataport.org/open-access/detecting-damaged-buildings-post-hurricane-satellite-imagery-based-customized)
- **Description**: The dataset comprises satellite images captured over Texas in the aftermath of Hurricane Harvey. The images are categorized into two classes:
  - **damage**: Buildings that have sustained visible hurricane damage.
  - **no_damage**: Buildings that appear unaffected.

## Model Architecture

The CNN model includes the following layers:

- **Input Layer**: Accepts images resized to **128x128x3**.
- **Convolutional Layers**: Four convolutional layers with ReLU activation, paired with max-pooling layers for feature extraction.
- **Dense Layers**: Fully connected layers with dropout for regularization.
- **Output Layer**: A single neuron with sigmoid activation to classify images into either "damage" or "no_damage".

### Key Requirements:

- Input images must be resized to **128x128** pixels.
- The final layer must have **1 neuron** with **sigmoid activation** for binary classification.
- Achieve a validation accuracy of approximately **95%** or higher.

## Data Preprocessing

- Images are normalized by scaling pixel values between 0 and 1.
- The dataset is divided into training and validation sets, ensuring proper model evaluation.

## How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/allanbil214/nn_HurricaneClassifier.git
   cd nn_HurricaneClassifier
   ```

2. **Install Dependencies**:
   Ensure you have TensorFlow installed:
   ```bash
   pip install tensorflow
   ```

3. **Run the Model**:
   Execute the script to download the dataset, train the model, and save the results:
   ```bash
   python nn_HurricaneClassifier.py
   ```

4. **Model Output**:
   The trained model will be saved as `model.h5`.

## Results

The model achieves a validation accuracy of over **95%**, demonstrating strong performance in differentiating between damaged and undamaged buildings from satellite imagery.

## License

This project is open-source and available under the [MIT License](LICENSE).
