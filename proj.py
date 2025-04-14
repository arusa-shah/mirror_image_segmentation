import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Hyperparameters
H, W = 256, 256
batch_size = 8  # Adjusted for GPU memory
num_epochs = 10
learning_rate = 1e-4

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
# Paths to dataset
image_paths = sorted(glob('/kaggle/input/isic2018-challenge-task1-data-segmentation/ISIC2018_Task1-2_Training_Input/*.jpg'))
mask_paths = sorted(glob('/kaggle/input/isic2018-challenge-task1-data-segmentation/ISIC2018_Task1_Training_GroundTruth/*.png'))

# Verify dataset loading
print(f"Total images: {len(image_paths)}, Total masks: {len(mask_paths)}")
# Preprocess images
def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (W, H))
    image = image / 255.0  # Normalize to [0, 1]
    return image.astype(np.float32)

# Preprocess masks
def read_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (W, H))
    mask = mask / 255.0  # Normalize to [0, 1]
    mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
    return mask.astype(np.float32)

# TensorFlow dataset parser
def tf_parse(image_path, mask_path):
    def _parse(image_path, mask_path):
        image = read_image(image_path.decode())
        mask = read_mask(mask_path.decode())
        return image, mask
    image, mask = tf.numpy_function(_parse, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, 1])
    return image, mask

# Create TensorFlow dataset
def tf_dataset(images, masks, batch):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Prepare dataset
dataset = tf_dataset(image_paths, mask_paths, batch_size)
print("Dataset ready.")

def build_unet_with_hooks(input_shape):
    inputs = tf.keras.layers.Input(input_shape)

    """Encoder"""
    s1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv1_1")(inputs)
    s1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv1_2")(s1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(s1)

    s2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv2_1")(p1)
    s2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv2_2")(s2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(s2)

    s3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="conv3_1")(p2)
    s3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="conv3_2")(s3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(s3)

    s4 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="conv4_1")(p3)
    s4 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="conv4_2")(s4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(s4)

    """Bridge"""
    b1 = tf.keras.layers.Conv2D(1024, (3, 3), activation="relu", padding="same", name="bridge1")(p4)
    b1 = tf.keras.layers.Conv2D(1024, (3, 3), activation="relu", padding="same", name="bridge2")(b1)

    """Decoder"""
    d1 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same", name="deconv1")(b1)
    d1 = tf.keras.layers.Concatenate()([d1, s4])
    d1 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="deconv1_1")(d1)

    d2 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same", name="deconv2")(d1)
    d2 = tf.keras.layers.Concatenate()([d2, s3])
    d2 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="deconv2_1")(d2)

    d3 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same", name="deconv3")(d2)
    d3 = tf.keras.layers.Concatenate()([d3, s2])
    d3 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="deconv3_1")(d3)

    d4 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same", name="deconv4")(d3)
    d4 = tf.keras.layers.Concatenate()([d4, s1])
    d4 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="deconv4_1")(d4)

    """Outputs"""
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", name="output")(d4)

    model = tf.keras.Model(inputs, outputs)
    return model

model = build_unet_with_hooks((H, W, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss="binary_crossentropy", metrics=["accuracy"])
# Build and compile the model
model = build_unet((H, W, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"])

# Display model summary
model.summary()

# Train the model
history = model.fit(dataset, epochs=num_epochs, steps_per_epoch=len(image_paths) // batch_size)
for layer in model.layers:
    print(layer.name)

def compute_cam(model, image, layer_name="conv2d_28"):  # Update with the correct layer name
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / tf.reduce_max(cam)
    return cam.numpy()


# Visualize CAM
def visualize_cam(image_path, model):
    image = read_image(image_path)
    cam = compute_cam(model, image)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(cam, cmap="jet", alpha=0.5)
    plt.title("CAM Overlay")
    plt.axis("off")

    plt.show()

# Example Usage
visualize_cam(image_paths[0], model)
visualize_cam(image_paths[0], model)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Function to calculate relevance maps
def calculate_lrp(model, image, layer_name="conv2d_26"):
    """
    Compute Layer-wise Relevance Propagation (LRP) for the model.
    Args:
        model: Trained U-Net model.
        image: Preprocessed input image (H, W, 3).
        layer_name: Name of the convolutional layer to use for LRP.
    Returns:
        relevance_map: Relevance heatmap.
    """
    grad_model = tf.keras.models.Model(
        [model.input],
        [model.get_layer(layer_name).output, model.output]
    )

    # Calculate gradients
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(np.expand_dims(image, axis=0))
        target_class = tf.reduce_max(predictions)  # Focus on the maximum prediction
        grads = tape.gradient(target_class, conv_output)
    
    # Compute relevance
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    relevance_map = tf.reduce_sum(pooled_grads * conv_output, axis=-1)

    # Normalize for visualization
    relevance_map = tf.maximum(relevance_map, 0)
    relevance_map = relevance_map / tf.reduce_max(relevance_map)
    return relevance_map.numpy()

# Function to visualize relevance maps inline
def visualize_lrp_inline(image_paths, model, num_images=5):
    """
    Display relevance maps for multiple input images inline.
    Args:
        image_paths: List of paths to input images.
        model: Trained U-Net model.
        num_images: Number of images to process and display.
    """
    for i, image_path in enumerate(image_paths[:num_images]):
        # Preprocess the input image
        image = read_image(image_path)

        # Calculate relevance map
        relevance_map = calculate_lrp(model, image)

        # Plot original image and relevance heatmap
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Original Image {i+1}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(relevance_map, cmap="hot")
        plt.title(f"Relevance Map {i+1}")
        plt.axis("off")

        plt.show()

# Example usage
visualize_lrp_inline(image_paths, model, num_images=5)
import cv2
import numpy as np
from sklearn.metrics import classification_report, f1_score

# Image dimensions
H, W = 256, 256

# Function to preprocess images
def preprocess_image(image_path):
    """
    Preprocess the input image.
    Args:
        image_path: Path to the input image file.
    Returns:
        Preprocessed image as a numpy array.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (W, H))
    img = img / 255.0  # Normalize pixel values
    return img.astype(np.float32)

# Function to preprocess masks
def preprocess_mask(mask_path):
    """
    Preprocess the input mask.
    Args:
        mask_path: Path to the mask file.
    Returns:
        Preprocessed mask as a numpy array.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (W, H))
    mask = mask / 255.0  # Normalize to 0 and 1
    return (mask > 0.5).astype(np.uint8)  # Binarize mask

# Function to calculate F1 score and generate a report
def calculate_f1_and_report(model, image_paths, mask_paths, num_samples=10, threshold=0.5):
    """
    Calculate F1 score and generate a classification report for the model predictions.

    Args:
        model: Trained U-Net model.
        image_paths: List of image file paths.
        mask_paths: List of mask file paths.
        num_samples: Number of samples to evaluate.
        threshold: Threshold for converting predicted probabilities to binary.

    Returns:
        f1: F1 score.
        report: Classification report as a string.
    """
    y_true = []
    y_pred = []

    for i, (image_path, mask_path) in enumerate(zip(image_paths[:num_samples], mask_paths[:num_samples])):
        # Preprocess image and mask
        img = preprocess_image(image_path)
        mask = preprocess_mask(mask_path)

        # Predict mask
        predicted_mask = model.predict(np.expand_dims(img, axis=0))[0]

        # Flatten ground truth and predictions
        y_true.append(mask.flatten())
        y_pred.append((predicted_mask.flatten() > threshold).astype(int))  # Apply threshold to binarize

    # Convert to numpy arrays
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Calculate F1 score
    f1 = f1_score(y_true, y_pred, zero_division=1)

    # Generate classification report
    report = classification_report(y_true, y_pred, zero_division=1)

    return f1, report

# Calculate F1 score and report
f1_score_value, f1_report = calculate_f1_and_report(model, image_paths, mask_paths)

# Save the report to a file
with open("f1_report.txt", "w") as f:
    f.write(f"F1 Score: {f1_score_value}\n\n")
    f.write(f1_report)

print(f"F1 Score: {f1_score_value}")
print(f"Classification Report:\n{f1_report}")
