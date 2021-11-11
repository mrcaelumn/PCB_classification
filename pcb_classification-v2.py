#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing Neccessary Library and constant variable

# !pip install tf_clahe
# !pip install -U scikit-learn
# !pip install matplotlib
# !pip install pandas
# !pip install tensorflow_io
# !pip install tensorflow_addons
# !pip install tf_clahe


# In[ ]:


import tensorflow as tf
# import tensorflow_io as tfio
# import tensorflow_addons as tfa

import numpy as np
# import pandas as pd
import os
import csv
# import tf_clahe
# import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
from packaging import version
from matplotlib import pyplot as plt

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2,     "This notebook requires TensorFlow 2.0 or above."

""" Set Hyper parameters """
NUM_EPOCHS = 50
CHOOSEN_MODEL = 2 # 1 == our model, 2 == mobilenet, 3 == resnet50
IMG_H = 224
IMG_W = 224
IMG_C = 3  ## Change this to 1 for grayscale.
COLOUR_MODE = "rgb"
BATCH_SIZE = 32

# set dir of files
TRAIN_DATASET_PATH = "image_dataset_final/test_training_dataset/"
TEST_DATASET_PATH = "image_dataset_final/evaluation_dataset/"
SAVED_MODEL_PATH = "saved_model/"
    
FORMAT_IMAGE = [".jpg",".png",".jpeg", ".bmp"]
HIGH_CLASS = [0]
MID_CLASS = [1, 4]
LOW_CLASS = [2, 3, 5, 6, 7]
CLASS_NAME = ["0","1","2", "3","4", "5", "6", "7"]

AUTOTUNE = tf.data.AUTOTUNE
AUGMENTATION = False
AUGMENTATION_REPEAT = True


# In[ ]:


"""
Custom Layer
"""
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical", input_shape=(IMG_H, IMG_W, IMG_C)),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.GaussianNoise(0.1),
])

def prep_image(image):
    """
    Preparation
    """
    if COLOUR_MODE == "grayscale":
        image = tf.image.rgb_to_grayscale(image)
        image = tf_clahe.clahe(image, tile_grid_size=(4, 4), clip_limit=4.0) 
    
    if COLOUR_MODE == "grayscale" and IMG_C == 3:
        image = tf.image.grayscale_to_rgb(image)
    
    # image = tf.image.resize(image, (IMG_H, IMG_W))
    # image = tf.cast(image, tf.float32)
    # image = (image / 255.0)  # rescailing image from 0,255 to 0, 1
    # img = (img - 127.5) / 127.5 # rescailing image from 0,255 to -1,1
    
    return image


# In[ ]:


def plot_roc_curve(fpr, tpr, n_model):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(n_model+'_roc_curve.png')
    plt.show()
    plt.clf()
    


''' calculate the auc value for labels and scores'''
def roc(labels, scores, n_model):
    """Compute ROC curve and ROC area for each class"""
    # True/False Positive Rates.
    fpr, tpr, threshold = roc_curve(labels, scores)
    print("threshold: ", threshold)
    roc_auc = auc(fpr, tpr)
    # get a threshold that perform very well.
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    # draw plot for ROC-Curve
    plot_roc_curve(fpr, tpr, n_model)
    
    return roc_auc, optimal_threshold


# In[ ]:


def confusion_matrix_report(labels, predicts, target_names):
    confusion = confusion_matrix(labels, predicts)
    print('Confusion Matrix\n')
    print(confusion)
    
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(labels, predicts)))

    print('Micro Precision: {:.2f}'.format(precision_score(labels, predicts, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(labels, predicts, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(labels, predicts, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(labels, predicts, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(labels, predicts, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(labels, predicts, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(labels, predicts, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(labels, predicts, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(labels, predicts, average='weighted')))

    print('\nClassification Report\n')
    print(classification_report(labels, predicts, target_names=target_names))


# In[ ]:


def plot_epoch_result(epochs, loss, name, model_name, colour):
    plt.plot(epochs, loss, colour, label=name)
#     plt.plot(epochs, disc_loss, 'b', label='Discriminator loss')
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(model_name+ '_'+name+'_epoch_result.png')
    plt.show()


class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self,
                 model_path,
                 n_model
                ):
        super(CustomSaver, self).__init__()
        self.history = {}
        self.epoch = []
        self.model_path = model_path
    
        self.name_model = n_model
        self.custom_loss = []
        self.epochs_list = []
            
    def on_train_end(self, logs=None):
        print(self.model_path)
        self.model.save(self.model_path)
        
        plot_epoch_result(self.epochs_list, self.custom_loss, "Loss", self.name_model, "g")

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.epoch.append(epoch)
        for k, v in logs.items():
#             print(k, v)
            self.history.setdefault(k, []).append(v)
        
        self.epochs_list.append(epoch)
        self.custom_loss.append(logs["loss"])

        if (epoch + 1) % 15 == 0:
            self.model.save(self.model_path)
            print('saved for epoch',epoch + 1)


# In[ ]:


def our_mobilenet(i_shape, base_lr, n_class):
    model = tf.keras.models.Sequential()
    
    base_model = tf.keras.applications.MobileNet(weights="imagenet", input_shape=i_shape, include_top=False)
    base_model.trainable = True
        
    if AUGMENTATION:
        model.add(data_augmentation)    
    
    model.add(base_model)
    
    
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.BatchNormalization())
    
    model.add(tf.keras.layers.Dense(1024
                                    ,activation = 'relu'
                                   ))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1024
                                    ,activation = 'relu'
                                   ))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(512
                                    ,activation = 'relu'
                                   ))
    model.add(tf.keras.layers.Dropout(0.5))
    
    
    
    model.add(tf.keras.layers.Dense(n_class
                                    ,activation="softmax"
                                   ))
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr),
                  metrics=['accuracy'])
    
    return model


# In[ ]:


def build_our_model(i_shape, base_lr, n_class):
    
    model = tf.keras.models.Sequential()
    
    if AUGMENTATION:
        model.add(data_augmentation)
        
    model.add(tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer="he_uniform", padding='same', input_shape=(IMG_H, IMG_W, IMG_C)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    

    model.add(tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer="he_uniform", padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer="he_uniform", padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    

    model.add(tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer="he_uniform", padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer="he_uniform", padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer="he_uniform", padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.4))
    
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128,
                                    kernel_initializer="he_uniform",
                                    activity_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_class, activation="tanh"))
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr),
                  metrics=['accuracy'])
    
    return model


# In[ ]:


def our_resnet50(i_shape, base_lr, n_class):
    model = tf.keras.models.Sequential()
    
    base_model = tf.keras.applications.ResNet50(weights=None, input_shape=i_shape, include_top=False)
    base_model.trainable = True
        
    if AUGMENTATION:
        model.add(data_augmentation)    
    
    model.add(base_model)
    
    
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256
                                    ,activation = 'relu'
                                    # ,kernel_regularizer=tf.keras.regularizers.l2(0.01)
                                   ))
    model.add(tf.keras.layers.Dropout(0.5))
    
    model.add(tf.keras.layers.Dense(n_class
                                    ,activation="softmax"
                                   ))
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr),
                  metrics=['accuracy'])
    
    return model


# In[ ]:


def evaluate_and_testing(this_model, p_model, test_dataset_path, c_names):
    """
    Evaluation Area
    """
    this_model.load_weights(p_model)
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_dataset = test_datagen.flow_from_directory(
        directory=test_dataset_path,
        target_size=(IMG_H, IMG_W),
        color_mode=COLOUR_MODE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=True,
        seed=42
    )
    
    print("Evaluate")
    result = this_model.evaluate(test_dataset)
    print(result)
    dict(zip(this_model.metrics_names, result))

    """
    Testing Area
    """
    
    pred_list = []
    name_image_list = []
    label_list = []
    
    probability_model = tf.keras.Sequential([
        this_model,
    ])
    for class_n in c_names:
        path = os.path.join(test_dataset_path, class_n)
        class_num = c_names.index(class_n)

        for img in tqdm(os.listdir(path)):  
            if img.endswith(tuple(FORMAT_IMAGE)):
                filepath = os.path.join(path, img)
                name_image = os.path.basename(filepath)
#                 print("name image: ", name_image, "label image: ", class_num)
                
                img = tf.keras.utils.load_img(
                    filepath, target_size=(IMG_H, IMG_W)
                )
                
                img_array = tf.keras.utils.img_to_array(img)
                
                img_array = tf.expand_dims(img_array, 0) # Create a batch

                pred_result = this_model.predict(img_array)
                # Generate arg maxes for predictions
                   
                pred_classes = np.argmax(pred_result[0])
                
                pred_list.append(pred_classes)
                name_image_list.append(name_image)
                label_list.append(class_num)

    # final_result = zip(name_image_list, label_list, pred_list)
    # for n, l, p in final_result:
    #      print("name image: ", n, "label image: ", l, "prediction class: ", p)
            
#     print("final result: ", final_result)
    confusion_matrix_report(label_list, pred_list, c_names)
    
    
    print("created csv for the result.")
    with open('predictions_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageName', 'Label'])
        writer.writerows(zip(name_image_list, pred_list))


# In[ ]:


# @tf.function
def dataset_manipulation(train_data_path, val_data_path):
    

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        
    )
    train_dataset = train_datagen.flow_from_directory(
        directory=train_data_path,
        target_size=(IMG_H, IMG_W),
        color_mode=COLOUR_MODE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=True,
        seed=42
    )
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
    )
    
    valid_dataset = validation_datagen.flow_from_directory(
        directory=val_data_path,
        target_size=(IMG_H, IMG_W),
        color_mode=COLOUR_MODE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=True,
        seed=42
    )
    
    
    
    return train_dataset, valid_dataset


# In[ ]:


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn


def __run__(our_model, train_dataset, val_dataset, num_epochs, path_model, name_model, class_name):
    
#     y = np.concatenate([y for x, y in train_dataset], axis=0)
#     print(dict(zip(*np.unique(y, return_counts=True))))
#     class_weights = class_weight.compute_class_weight(
#         class_weight='balanced',
#         classes=np.unique(y), 
#         y=y
#     )
    
#     train_class_weights = dict(enumerate(class_weights))
    
#     print("class_weights: ", train_class_weights)

    
    saver_callback = CustomSaver(
        path_model,
        name_model
    )
    
    
    exponential_decay_fn = exponential_decay(0.01, 20)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
    
    
    fit_history_our_model = our_model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
        batch_size=BATCH_SIZE,
        # class_weight=train_class_weights,
        callbacks=[
            saver_callback, 
            # lr_scheduler
        ]   
    )
    
    acc = fit_history_our_model.history['accuracy']
    val_acc = fit_history_our_model.history['val_accuracy']

    loss = fit_history_our_model.history['loss']
    val_loss = fit_history_our_model.history['val_loss']

    epochs_range = range(num_epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    plt.savefig(name_model+'_trainning_result.png')

    evaluate_and_testing(our_model, path_model, TEST_DATASET_PATH, class_name)


# In[ ]:


if __name__ == "__main__":
    
    '''
    Model for computer-vision-based Printed Circuit Board (PCB).
    analysis dataset used for classification of defects.
    '''
    
    # run the function here
    
    name_model = str(IMG_H)+"_pcb_"+str(NUM_EPOCHS)
    if CHOOSEN_MODEL == 1:
        name_model = name_model + "-custom_model"
    elif CHOOSEN_MODEL == 2:
        name_model = name_model + "-mobilenet"
    elif CHOOSEN_MODEL == 3:
        name_model = name_model + "-resnet50"
        
    print("start: ", name_model)
    base_learning_rate = 0.00001
    num_classes = 8
    class_name = ["0", "1", "2", "3", "4", "5", "6", "7"]
    
    
    input_shape = (IMG_H, IMG_W, IMG_C)
    
    path_model = SAVED_MODEL_PATH + name_model + "_model" + ".h5"
    
    train_dataset, val_dataset = dataset_manipulation(TRAIN_DATASET_PATH, TEST_DATASET_PATH)
    
        
    if CHOOSEN_MODEL == 1:
        """
        our custom model
        """ 
        print("running", name_model)
        our_model = build_our_model(input_shape, base_learning_rate, num_classes)
        # our_model.summary()
        __run__(our_model, train_dataset, val_dataset, NUM_EPOCHS, path_model, name_model, class_name)
    elif CHOOSEN_MODEL == 2:
        """
        mobilenet
        """
        print("running", name_model)
        our_model = our_mobilenet(input_shape, base_learning_rate, num_classes)
        # our_model.summary()
        __run__(our_model, train_dataset, val_dataset, NUM_EPOCHS, path_model, name_model, class_name)
    elif CHOOSEN_MODEL == 3:
        """
        mobilenet
        """
        print("running", name_model)
        our_model = our_resnet50(input_shape, base_learning_rate, num_classes)
        # our_model.summary()
        __run__(our_model, train_dataset, val_dataset, NUM_EPOCHS, path_model, name_model, class_name)


# In[ ]:




