# %%
#Import Liberaries
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


# Set all the Constants (Hyperparameters)
# Hyperparameters are settings that define important aspects of the training process.
# 
# BATCH_SIZE:
# This is the number of training examples used in one go.
# During each training step, the model updates its weights based on this subset of the dataset.
# The batch size can affect how fast the model trains and how much memory it uses.
# IMAGE_SIZE:
# This refers to the width and height of the input images.
# The size of the images needs to be consistent for the neural network to process them correctly.
# CHANNEL:
# This is the number of color channels in the images.
# For example, RGB images have three channels (Red, Green, Blue), while grayscale images have one channel.
# The number of channels affects how the model processes the images.

#Importing Data


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory='PlantVillage',  # Folder with the images
    seed=123,                  # Random seed for shuffling
    shuffle=True,              # Shuffle the images
    image_size=(IMAGE_SIZE, IMAGE_SIZE),  # Resize images to this size
    batch_size=BATCH_SIZE      # Number of images to process at once
)


# Sure, here's a simpler explanation:
# 
# ---
# 
# ### Import Data into TensorFlow Dataset Object
# 
# We will use the `image_dataset_from_directory` API to load all images into a TensorFlow dataset:
# 
# 1.  directory='PlantVillage:
#    - This is the path to the folder with your images. Here, it's the 'PlantVillage' directory.
# 
# 2. seed=123:
#    - This sets a random seed for shuffling the data. Using the same seed every time ensures you get the same shuffle order, making results reproducible.
# 
# 3. shuffle=True:
#    - This shuffles the images each time you go through the dataset. Shuffling helps the model learn better by preventing it from memorizing the order of the images.
# 
# 4. image_size=(IMAGE_SIZE, IMAGE_SIZE):
#    - This resizes all images to the specified dimensions (width and height). The `IMAGE_SIZE` variable holds the size you want.
# 
# 5. batch_size=BATCH_SIZE:
#    - This sets how many images to process at once. The `BATCH_SIZE` variable holds this number. Using batches helps manage memory and speeds up training.
# 

class_names = dataset.class_names
class_names
for image_batch, label_batch in dataset.take(1):
    print("Image Batch Shape:", image_batch.shape)  # Shows the shape of the batch of images
    print("Single Image:", image_batch[0])          # Displays the first image in the batch
    print("Label Image numpy:", label_batch.numpy)  # Converts the labels to numpy format and displays them



len(class_names)


#Visualize some of the images from our dataset
plt.figure(figsize=(10, 10))  # Create a figure that's 10x10 inches
for image_batch, labels_batch in dataset.take(1):  # Take one batch of images and labels from the dataset
    for i in range(12):  # Loop through the first 12 images in the batch
        ax = plt.subplot(3, 4, i + 1)  # Create a subplot in a 3x4 grid
        plt.imshow(image_batch[i].numpy().astype("uint8"))  # Show the image
        plt.title(class_names[labels_batch[i]])  # Set the title to the class name of the image
        plt.axis("off")  # Hide the axis

#Function to Split Dataset
#We need to divide the dataset into three parts:

#Training: Used to train the model.
#Validation: Used to check the model’s performance during training.
#Test: Used to evaluate the model after training is complete.

len(dataset)

train_size = 0.8
len(dataset) * train_size
train_ds = dataset.take(54)
len(train_ds)

val_size = 0.1
len(dataset)*val_size

val_ds = test_ds.take(6)
len(val_ds)

test_ds = test_ds.skip(6)
len(test_ds)

def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)


len(train_ds)



len(val_ds)


len(test_ds)


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),  # Resize images to IMAGE_SIZE x IMAGE_SIZE
    layers.experimental.preprocessing.Rescaling(1./255),  # Scale pixel values to be between 0 and 1
])


#Data Augmentation

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),  # Randomly flip images horizontally and vertically
    layers.experimental.preprocessing.RandomRotation(0.2),  # Randomly rotate images by 20%
])


# This code creates a sequence of two steps to augment the data:
# 
# Random Flip: Randomly flips the images both horizontally and vertically.
# Random Rotation: Randomly rotates the images by up to 20%.

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNEL)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,  # Resize and rescale the images
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),  # First convolutional layer with 32 filters
    layers.MaxPooling2D((2, 2)),  # First max pooling layer
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),  # Second convolutional layer with 64 filters
    layers.MaxPooling2D((2, 2)),  # Second max pooling layer
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),  # Third convolutional layer with 64 filters
    layers.MaxPooling2D((2, 2)),  # Third max pooling layer
    layers.Conv2D(64, (3, 3), activation='relu'),  # Fourth convolutional layer with 64 filters
    layers.MaxPooling2D((2, 2)),  # Fourth max pooling layer
    layers.Conv2D(64, (3, 3), activation='relu'),  # Fifth convolutional layer with 64 filters
    layers.MaxPooling2D((2, 2)),  # Fifth max pooling layer
    layers.Conv2D(64, (3, 3), activation='relu'),  # Sixth convolutional layer with 64 filters
    layers.MaxPooling2D((2, 2)),  # Sixth max pooling layer
    layers.Flatten(),  # Flatten the output
    layers.Dense(64, activation='relu'),  # Fully connected layer with 64 units
    layers.Dense(n_classes, activation='softmax'),  # Output layer with 'n_classes' units
])

model.build(input_shape=input_shape)  # Build the model with the specified input shape

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=20,
)

scores = model.evaluate(test_ds)

model.save("model.h5")

#Plotting the Accuracy and Loss Curves
print(history)
print(history.params)
print(history.history.keys())

history.history['loss'][:5] # show loss for first 5 epochs

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# Create a figure that's 8x8 inches
plt.figure(figsize=(8, 8))

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 1)  # Create the first subplot in a 1x2 grid
plt.plot(range(EPOCHS), acc, label='Training Accuracy')  # Plot training accuracy
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')  # Plot validation accuracy
plt.legend(loc='lower right')  # Add a legend in the lower right corner
plt.title('Training and Validation Accuracy')  # Set the title

# Plot Training and Validation Loss
plt.subplot(1, 2, 2)  # Create the second subplot in a 1x2 grid
plt.plot(range(EPOCHS), loss, label='Training Loss')  # Plot training loss
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')  # Plot validation loss
plt.legend(loc='upper right')  # Add a legend in the upper right corner
plt.title('Training and Validation Loss')  # Set the title

# Display the plots
plt.show()

#Run prediction on a sample image
model = tf.keras.models.load_model('model.h5')


import numpy as np
for images_batch, labels_batch in test_ds.take(1):
    
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction = model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])

#Write a function for inference


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]] 
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")



