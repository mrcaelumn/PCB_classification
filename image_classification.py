#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import itertools
import tensorflow as tf
import numpy as np
from packaging import version

from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score

from matplotlib import pyplot as plt



print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2,     "This notebook requires TensorFlow 2.0 or above."

# Weight initializers for the Generator network
WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2)


# In[ ]:


def plot_roc_curve(fpr, tpr, name_model):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(name_model+'_roc_curve.png')
    plt.show()
    plt.clf()
    


''' calculate the auc value for labels and scores'''
def roc(labels, scores, name_model):
    """Compute ROC curve and ROC area for each class"""

    # True/False Positive Rates.
    fpr, tpr, threshold = roc_curve(labels, scores)
    print("threshold: ", threshold)
    roc_auc = auc(fpr, tpr)
    # get a threshold that perform very well.
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    # draw plot for ROC-Curve
    plot_roc_curve(fpr, tpr, name_model)
    
    return roc_auc, optimal_threshold


# In[ ]:


def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        c_map=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=c_map)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title+'_cm.png')
    plt.show()
    plt.clf()


# In[ ]:


class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self,
                 model_path,
                 name_model
                ):
        super(CustomSaver, self).__init__()
        self.model_path = model_path
    
        self.name_model = name_model
        self.custom_loss = []
        self.epochs_list = []
        
    
    def on_train_begin(self, logs=None):
        if not hasattr(self, 'epoch'):
            self.epoch = []
            self.history = {}
            
    def on_train_end(self, logs=None):
        self.model.save(self.model_path)
        
        self.plot_epoch_result(self.epochs_list, self.custom_loss, "Loss", self.name_model, "g")

    
    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
#             print(k, v)
            self.history.setdefault(k, []).append(v)
        
        self.epochs_list.append(epoch)
        self.custom_loss.append(logs["loss"])

        
        if (epoch + 1) % 15 == 0 or (epoch + 1) <= 15:
            self.model.save(self.model_path)
            print('saved for epoch',epoch + 1)
            
    def plot_epoch_result(self, epochs, loss, name, model_name, colour):
        plt.plot(epochs, loss, colour, label=name)
    #     plt.plot(epochs, disc_loss, 'b', label='Discriminator loss')
        plt.title(name)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(model_name+ '_'+name+'_epoch_result.png')
        plt.show()


# In[ ]:


if __name__ == "__main__":
    
    '''
    Model for computer-vision-based Printed Circuit Board (PCB).
    analysis dataset used for classification of defects.
    '''
    
    
    # run the function here
    """ Set Hyper parameters """
    batch_size = 25
    num_epochs = 100
    
    IMG_H = 110
    IMG_W = 42
    IMG_C = 3  ## Change this to 1 for grayscale.

    name_model = str(IMG_H)+"_pcb_"+str(num_epochs)
    print("start: ", name_model)
    base_learning_rate = 0.0001
    num_classes = 8
    
    # set dir of files
    train_data_path = "image_dataset/training_dataset"
    test_data_path = "image_dataset/evaluation_dataset"
    saved_model_path = "saved_model/"
    
    input_shape = (IMG_W, IMG_H, IMG_C)
    # print(input_shape)

    path_model = saved_model_path + name_model + "_model" + ".h5"
    
    # set input 
    inputs = tf.keras.layers.Input(input_shape, name="input_1")


    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_data_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_H, IMG_W),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

#     plt.figure(figsize=(10, 10))
#     for images, labels in train_ds.take(1):
#         for i in range(9):
#             ax = plt.subplot(3, 3, i + 1)
#             plt.imshow(images[i].numpy().astype("uint8"))
#             plt.title(class_names[labels[i]])
#             plt.axis("off")
#     plt.show()
    
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal',
                                                            input_shape=(IMG_H,
                                                                         IMG_W,
                                                                         IMG_C)),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    our_model = tf.keras.models.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1./255, input_shape=(IMG_H, IMG_W, IMG_C)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])
    
    our_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate/10),
              metrics=['accuracy'])
    
#     our_model.summary()
    
    
    saver_callback = CustomSaver(
        path_model,
        name_model
    )
    
    fit_history = our_model.fit(
        train_ds,
        epochs=num_epochs,
        callbacks=[saver_callback]
    )
    
    
#     acc = fit_history.history['accuracy']
#     val_acc = fit_history.history['val_accuracy']
#     loss = fit_history.history['loss']
#     val_loss = fit_history.history['val_loss']

#     epochs = range(len(acc))

#     plt.plot(epochs, acc, 'r', label='Training accuracy')
#     plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
#     plt.title('Training and validation accuracy')

#     plt.figure()

#     plt.plot(epochs, loss, 'r', label='Training Loss')
#     plt.plot(epochs, val_loss, 'b', label='Validation Loss')
#     plt.title('Training and validation loss')

#     plt.legend()

#     plt.show()
    
    
    """
    Evaluation Area
    """
    our_model.load_weights(path_model)
    evaluation_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_path,
        seed=123,
        image_size=(IMG_H, IMG_W),
        batch_size=batch_size
    )
    numpy_labels = None
    for _, labels in evaluation_ds.take(1):  # only take first element of dataset
#     numpy_images = images.numpy()
        numpy_labels = labels.numpy()
    
    
    y = np.concatenate([y for x, y in evaluation_ds], axis=0)
    print(y)
    # Evaluate the model on the test data using `evaluate`
    # You can also evaluate or predict on a dataset.
    print("Evaluate")
    result = our_model.evaluate(evaluation_ds)
    print(result)
    dict(zip(our_model.metrics_names, result))

        
    pred_result = our_model.predict(evaluation_ds)
    print(pred_result)
    
    # Generate arg maxes for predictions
#     score = tf.nn.softmax(pred_result)
    classes = np.argmax(pred_result, axis = 1)
    print(classes)
#     print(len(classes))
#     
    
#     print(score)
#     print(
#         "This image most likely belongs to {} with a {:.2f} percent confidence."
#         .format(class_names[np.argmax(score)], 100 * np.max(score))
#     )

