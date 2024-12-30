import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import mean_squared_error, classification_report

# Create evaluation folder
os.makedirs('evaluation4', exist_ok=True)

# Parameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
NUM_CLASSES = 3  # david, naomi, pacheco

def preprocess_image(image, label, is_training=True):
    image = tf.cast(image, tf.float32) / 255.0
    
    if is_training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
    
    image = tf.image.resize(image, IMG_SIZE)
    return image, label

def load_dataset(directory, is_training=True):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        class_names=['david', 'naomi', 'pacheco'],
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=is_training
    )
    
    dataset = dataset.map(
        lambda x, y: preprocess_image(x, y, is_training),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return dataset.prefetch(tf.data.AUTOTUNE)

def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)

# Load datasets
train_dataset = load_dataset('dataset/train', is_training=True)
val_dataset = load_dataset('dataset/train', is_training=False)
test_dataset = load_dataset('dataset/test', is_training=False)

# Create and compile model
model = create_model()
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
]

# Train model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=callbacks
)

# Save model
model.save("evaluation4/model_resnet50.h5")

# Plot training history
plt.figure(figsize=(20, 15))

metrics = ['accuracy', 'loss']
titles = ['Model Accuracy', 'Model Loss']

for i, metric in enumerate(metrics):
    plt.subplot(2, 1, i+1)
    plt.plot(history.history[metric], label=f'Training {metric.capitalize()}')
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
    plt.title(titles[i])
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()

plt.tight_layout()
plt.savefig('evaluation4/training_history.png')
plt.close()

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_dataset)

# Make predictions
predictions = model.predict(test_dataset)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(np.concatenate([y for x, y in test_dataset], axis=0), axis=1)

# Calculate metrics
cm = confusion_matrix(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
mse = mean_squared_error(y_true, y_pred)

# Generate classification report
class_names = ['david', 'naomi', 'pacheco']
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:")
print(report)

# Save metrics and reports
with open('evaluation4/classification_report.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)

with open('evaluation4/metrics.txt', 'w') as f:
    f.write(f"Test Accuracy: {test_accuracy}\n")
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Macro Precision: {precision}\n")
    f.write(f"Macro Recall: {recall}\n")
    f.write(f"Macro F1 Score: {f1}\n")
    f.write(f"MSE: {mse}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

# Plot confusion matrix
plt.figure(figsize=(10,8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(NUM_CLASSES)
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, str(cm[i, j]), 
                horizontalalignment="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.savefig('evaluation4/confusion_matrix.png')
plt.close()

print("Training, evaluation, and visualizations complete. Results saved in the 'evaluation4' folder.")