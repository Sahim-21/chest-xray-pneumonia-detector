from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))

classifier.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

training_set = train_datagen.flow_from_directory(
    "C:/Users/sahim/OneDrive/Documents/NNL/chest_xray/chest_xray/train",
    target_size=(64, 64),
    batch_size=32,
    class_mode="binary"
)

test_set = test_datagen.flow_from_directory(
    "C:/Users/sahim/OneDrive/Documents/NNL/chest_xray/chest_xray/test",
    target_size=(64, 64),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)


history = classifier.fit(
    training_set,
    epochs=10,
    validation_data=test_set
)


print(history.history.keys())

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Training Set", "Test Set"], loc="upper left")
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training Set", "Test Set"], loc="upper left")
plt.show()

print("Saving model to disk...")
classifier.save('pneumonia_model.h5')
print("Model saved successfully!")