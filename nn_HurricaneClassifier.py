import urllib
import zipfile

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Function to download and extract dataset from a given URL
def download_and_extract_data():
    # URL to the zip file containing the dataset
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/satellitehurricaneimages.zip'
    # Download the dataset zip file to the local directory
    urllib.request.urlretrieve(url, 'satellitehurricaneimages.zip')
    # Extract the contents of the zip file to the current directory
    with zipfile.ZipFile('satellitehurricaneimages.zip', 'r') as zip_ref:
        zip_ref.extractall()

# Function to preprocess images before feeding into the model
def preprocess(image, label):
    # Normalize the image pixel values to be between 0 and 1
    image = image / 255.0
    return image, label

# Main function to define, train, and save the model
def solution_model():
    # Download and extract the dataset
    download_and_extract_data()

    # Define image size and batch size
    IMG_SIZE = (128, 128)  # Resizing images to 128x128
    BATCH_SIZE = 64  # Define the batch size for training

    # Load training dataset from the 'train' directory, with image resizing
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='train',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Load validation dataset from the 'validation' directory, with image resizing
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='validation',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Preprocess the training dataset using the defined preprocess function
    train_ds = train_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
        tf.data.experimental.AUTOTUNE)
    
    # Preprocess the validation dataset
    val_ds = val_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Define the architecture of the model
    model = tf.keras.models.Sequential([
        # First Convolutional Layer: 32 filters, 3x3 kernel, ReLU activation
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),  # MaxPooling layer to reduce spatial dimensions
        
        # Second Convolutional Layer: 64 filters, 3x3 kernel, ReLU activation
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),  # MaxPooling layer to reduce spatial dimensions
        
        # Third Convolutional Layer: 64 filters, 3x3 kernel, ReLU activation
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),  # MaxPooling layer to reduce spatial dimensions
        
        # Fourth Convolutional Layer: 128 filters, 3x3 kernel, ReLU activation
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),  # MaxPooling layer to reduce spatial dimensions
        
        Flatten(),  # Flatten the 3D data to 1D for fully connected layers
        Dense(512, activation='relu'),  # Dense layer with 512 units and ReLU activation
        Dropout(0.5),  # Dropout layer to prevent overfitting
        Dense(128, activation='relu'),  # Dense layer with 128 units and ReLU activation
        Dropout(0.3),  # Dropout layer to further prevent overfitting

        # Output layer: Single unit with sigmoid activation for binary classification
        Dense(1, activation='sigmoid')
    ])

    # Compile the model with Adam optimizer, binary crossentropy loss, and accuracy metric
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model for 15 epochs using the training and validation datasets
    model.fit(
        train_ds,
        epochs=15,
        validation_data=val_ds,
        verbose=1
    )
    
    return model

# Main execution: if the script is run directly, train and save the model
if __name__ == '__main__':
    # Train the model and save it to a file
    model = solution_model()
    model.save("model.h5")
