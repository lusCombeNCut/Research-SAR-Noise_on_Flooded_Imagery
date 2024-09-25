import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '2' to also show warnings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # change this to the specific GPU you want to use

import json
import platform
import sys
import matplotlib.pyplot as plt
import architectures as arches
# Apply the custom import hook for handling 'resource' module on Windows
if platform.system() == 'Windows':
    import sys
    sys.path.insert(0, os.path.abspath('.'))
    import UnusedScripts.custom_import_hook as custom_import_hook

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras as keras
import tensorflow as tf
from preprocessing import data_generator

INPUT_CHANNELS = 3
IMAGE_SHAPE = (256, 256)
BATCH_SIZE = 1
INITIAL_LR = 0.01 * BATCH_SIZE / 16
EPOCHS = 1
NUM_CLASSES = 1
learning_rate = keras.optimizers.schedules.CosineDecay(
    INITIAL_LR,
    decay_steps=EPOCHS * 2124,
)

def dict_to_tuple(x):
    return x["images"], tf.expand_dims(tf.cast(x["segmentation_masks"], "int32"), axis=-1)

def tf_dataset(data_path, json_data, subset="test"):
    return tf.data.Dataset.from_generator(
    lambda: data_generator(data_path, json_data, subset, IMAGE_SHAPE),
    output_signature=(
        {
            "images": tf.TensorSpec(shape=IMAGE_SHAPE+(INPUT_CHANNELS,), dtype=tf.float32),
            "segmentation_masks": tf.TensorSpec(shape=IMAGE_SHAPE, dtype=tf.int32),
        }
    )).map(dict_to_tuple).batch(BATCH_SIZE).shuffle(1000)
    
def load_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

# Paths
base_path = 'mmflood'
data_path = os.path.join(base_path, 'mmflood')
json_path = os.path.join(base_path, 'activations.json')
json_data = load_json(json_path)

train_gen = tf_dataset(data_path, json_data, subset="train")
val_gen = tf_dataset(data_path, json_data, subset="test")

# ~~~ Chose Model Architecture ~~~
model = arches.simple(input_shape=IMAGE_SHAPE+(INPUT_CHANNELS,), learning_rate=INITIAL_LR)
model.summary()
tf.keras.utils.plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

# Train the model and store the training history
history = model.fit(
    train_gen,
    steps_per_epoch=100,
    validation_data=val_gen,
    validation_steps=50,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

while True:
    try:
            # Ask the user for a model name
        model_name = input("Enter the name for the saved model: ")

        # Check if the file already exists and delete if necessary
        if os.path.exists(f'{model_name}.keras'):
            os.remove(f'{model_name}.keras')
        # Save the entire model in the new Keras format
        model.save(f'{model_name}.keras')
        print(f"Model saved as {model_name}.keras")
        break  # Exit the loop if save is successful
    except Exception as e:
        print(f"An error occurred: {e}. Retrying...")


# Plot training & validation accuracy and loss values
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')

plt.savefig('training_validation_plots.png')
plt.show()

