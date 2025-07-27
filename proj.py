import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

H, W = 256, 256
batch_size = 8
num_epochs = 10
learning_rate = 1e-4

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

image_paths = sorted(glob('/kaggle/input/isic2018-challenge-task1-data-segmentation/ISIC2018_Task1-2_Training_Input/*.jpg'))
mask_paths = sorted(glob('/kaggle/input/isic2018-challenge-task1-data-segmentation/ISIC2018_Task1_Training_GroundTruth/*.png'))

print(f"Total images: {len(image_paths)}, Total masks: {len(mask_paths)}")

train_images, val_images, train_masks, val_masks = train_test_split(image_paths, mask_paths, test_size=0.1, random_state=42)

def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (W, H))
    image = image / 255.0
    return image.astype(np.float32)

def read_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask.astype(np.float32)

def tf_parse(image_path, mask_path):
    def _parse(image_path, mask_path):
        image = read_image(image_path.decode())
        mask = read_mask(mask_path.decode())
        return image, mask
    image, mask = tf.numpy_function(_parse, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, 1])
    return image, mask

def tf_dataset(images, masks, batch):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def build_unet_with_hooks(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
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

    b1 = tf.keras.layers.Conv2D(1024, (3, 3), activation="relu", padding="same", name="bridge1")(p4)
    b1 = tf.keras.layers.Conv2D(1024, (3, 3), activation="relu", padding="same", name="bridge2")(b1)

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

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", name="output")(d4)

    model = tf.keras.Model(inputs, outputs)
    return model

train_dataset = tf_dataset(train_images, train_masks, batch_size)
val_dataset = tf_dataset(val_images, val_masks, batch_size)

model = build_unet_with_hooks((H, W, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(train_dataset, validation_data=val_dataset, epochs=num_epochs)

# Plot training curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_curve.png")
plt.show()

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (H, W))
    img = img / 255.0
    return img.astype(np.float32)

def preprocess_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (H, W), interpolation=cv2.INTER_NEAREST)
    mask = mask / 255.0
    return (mask > 0.5).astype(np.uint8)

def calculate_f1_and_report(model, image_paths, mask_paths, num_samples=10, threshold=0.5):
    y_true, y_pred = [], []
    for i, (image_path, mask_path) in enumerate(zip(image_paths[:num_samples], mask_paths[:num_samples])):
        img = preprocess_image(image_path)
        mask = preprocess_mask(mask_path)
        predicted_mask = model.predict(np.expand_dims(img, axis=0))[0]
        y_true.append(mask.flatten())
        y_pred.append((predicted_mask.flatten() > threshold).astype(int))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    report = classification_report(y_true, y_pred, zero_division=1)
    return f1, report

f1_score_value, f1_report = calculate_f1_and_report(model, val_images, val_masks)

with open("f1_report.txt", "w") as f:
    f.write(f"F1 Score: {f1_score_value}\n\n")
    f.write(f1_report)

print(f"F1 Score: {f1_score_value}")
print(f"Classification Report:\n{f1_report}")


def visualize_predictions(model, image_paths, mask_paths, num_samples=5):
    for i in range(num_samples):
        image = preprocess_image(image_paths[i])
        mask = preprocess_mask(mask_paths[i])
        pred = model.predict(np.expand_dims(image, axis=0))[0, :, :, 0]
        pred = (pred > 0.5).astype(np.uint8)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"prediction_{i+1}.png")
        plt.show()

visualize_predictions(model, val_images, val_masks)
