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
# !pip install git+https://github.com/qubvel/classification_models.git


# In[ ]:


import tensorflow as tf
# import tensorflow_io as tfio
# import tensorflow_addons as tfa

import numpy as np
# import pandas as pd
import os
import csv
import tf_clahe
# import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
import math
from packaging import version
from matplotlib import pyplot as plt

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2,     "This notebook requires TensorFlow 2.0 or above."

""" Set Hyper parameters """
NUM_EPOCHS = 150
CHOOSEN_MODEL = 1 # 1 == resnet18, 2 == custom_our_model, 3 == VGG16
IMG_H = 42
IMG_W = 110
IMG_C = 3  ## Change this to 1 for grayscale.
COLOUR_MODE = "rgb"
BATCH_SIZE = 32

# set dir of files
TRAIN_DATASET_PATH = "image_dataset_final_grayscale_clahe/training_dataset/"
TEST_DATASET_PATH = "image_dataset_final_grayscale_clahe/clahe_evaluation_dataset/"
SAVED_MODEL_PATH = "saved_model/"
    
FORMAT_IMAGE = [".jpg",".png",".jpeg", ".bmp"]
HIGH_CLASS = [0]
MID_CLASS = [1, 4]
LOW_CLASS = [2, 3, 5, 6, 7]
CLASS_NAME = ["0","1","2", "3","4", "5", "6", "7"]

AUTOTUNE = tf.data.AUTOTUNE
AUGMENTATION = False
TRAIN_MODE = True


# In[ ]:


"""
Custom Layer
"""
data_augmentation = tf.keras.Sequential([
    # tf.keras.layers.RandomFlip("horizontal_and_vertical", input_shape=(IMG_H, IMG_W, IMG_C)),
    tf.keras.layers.RandomContrast(0.1),
    # tf.keras.layers.GaussianNoise(0.1),
])


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
        self.model.save_weights(self.model_path)
        
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
            self.model.save_weights(self.model_path)
            print('saved for epoch',epoch + 1)


# In[ ]:


def build_our_model(i_shape, base_lr, n_class):
    
    model = tf.keras.models.Sequential()
    
    if AUGMENTATION:
        model.add(data_augmentation)
        
    model.add(tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer="he_uniform", padding='same', input_shape=(IMG_H, IMG_W, IMG_C)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer="he_uniform", padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer="he_uniform", padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer="he_uniform", padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer="he_uniform", padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer="he_uniform", padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.4))
    
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_class, activation="softmax"))
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr),
                  metrics=['accuracy'])
    
    return model


# In[ ]:


def build_VGG16(i_shape, base_lr, n_class):
    inputs = tf.keras.layers.Input(i_shape)
    
    base_model = tf.keras.applications.vgg16.VGG16(weights="imagenet", input_shape=i_shape, include_top=False)
    base_model.trainable = False
    
    # flatten the output of the convolutional part: 
    x = tf.keras.layers.Flatten()(base_model.output)
    # three hidden layers
    x = tf.keras.layers.Dense(100, activation='relu')(x)
    x = tf.keras.layers.Dense(100, activation='relu')(x)
    x = tf.keras.layers.Dense(100, activation='relu')(x)

    # bn = tf.keras.layers.BatchNormalization()(inputs)
    # x = base_model(bn)
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    out =tf.keras.layers.Dense(n_class, activation="softmax")(x)
    
    model = tf.keras.models.Model(inputs, out)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr),
                  metrics=['accuracy'])
    
    model.summary()
    return model


# In[ ]:


def relu_bn(inputs: tf.Tensor) -> tf.Tensor:
    relu = tf.keras.layers.ReLU()(inputs)
    bn = tf.keras.layers.BatchNormalization()(relu)
    return bn

def residual_block(x: tf.Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> tf.Tensor:
    y = tf.keras.layers.Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = tf.keras.layers.Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = tf.keras.layers.Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = tf.keras.layers.Add()([x, y])
    out = relu_bn(out)
    return out

def build_resnet18(i_shape, base_lr, n_class):
    
    inputs = tf.keras.layers.Input(shape=i_shape)
    num_filters = 64
    
    t = tf.keras.layers.BatchNormalization()(inputs)
    t = tf.keras.layers.Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2
    
    
    t = tf.keras.layers.AveragePooling2D(4)(t)
    t = tf.keras.layers.Dense(1024, activation='relu')(t)    
    t = tf.keras.layers.Flatten()(t)
    t = tf.keras.layers.ReLU()(t)
    t = tf.keras.layers.Dropout(0.5)(t)
    outputs = tf.keras.layers.Dense(n_class, activation='softmax')(t)
    
    model = tf.keras.models.Model(inputs, outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# In[ ]:


# @tf.function
def prep_image(image, label = None):
    """
    Preparation
    """
    if COLOUR_MODE == "grayscale":
        image = tf.image.rgb_to_grayscale(image)
        image = tf_clahe.clahe(image, tile_grid_size=(8, 8), clip_limit=2.0) 
    
    if COLOUR_MODE == "grayscale" and IMG_C == 3:
        image = tf.image.grayscale_to_rgb(image)
    
    # image = tf.image.resize(image, (IMG_H, IMG_W))
    # image = tf.cast(image, tf.float32)
    # image = (image / 255.0)  # rescailing image from 0,255 to 0, 1
    # img = (img - 127.5) / 127.5 # rescailing image from 0,255 to -1,1
    
    return image, label


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
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=True,
        seed=42
    )
    
    if COLOUR_MODE == "grayscale":
        f_test_dataset = tf.data.Dataset.from_generator(
            lambda: test_dataset,
            output_types = (tf.float32, tf.int64),
        )
        # plt.figure(figsize=(10, 10))
        # for images, labels in f_test_dataset.take(1):
        #     for i in range(9):
        #         ax = plt.subplot(3, 3,i+1)
        #         plt.imshow(images[i].numpy().astype("uint8"))
        #         plt.title(CLASS_NAME[labels[i]])
        #         plt.axis("off")

        test_dataset = f_train_dataset.map(prep_image)
        
        # plt.figure(figsize=(10, 10))
        # for images, labels in test_dataset.take(1):
        #     for i in range(9):
        #         ax = plt.subplot(3, 3,i+1)
        #         plt.imshow(images[i].numpy().astype("uint8"))
        #         plt.title(CLASS_NAME[labels[i]])
        #         plt.axis("off")
    
    print("=======Evaluating=======")
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
                img , _ = prep_image(img, class_num)
                img_array = tf.keras.utils.img_to_array(img)
                img_array = img_array/255.5 # normalize
                img_array = tf.expand_dims(img_array, axis=0) # Create a batch

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


def color_jitter(image, brightness=25, contrast=0.2, saturation=0.2, hue=0.1):
    """Distort the color of the image."""
    if brightness > 0:
        image = tf.image.random_brightness(image, max_delta=brightness)
    if contrast > 0:
        image = tf.image.random_contrast(image, lower=1-contrast, upper=1+contrast)
    if saturation > 0:
        image = tf.image.random_saturation(image, lower=1-saturation, upper=1+saturation)
    if hue > 0:
        image = tf.image.random_hue(image, max_delta=hue)
    return image 

def random_color_jitter(image):
    image = color_jitter(image)
    return image

def dataset_manipulation(train_data_path, val_data_path):
    

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # rotation_range=20,
        rescale=1./255,
        # shear_range=0.1,
        # zoom_range=0.1,
        brightness_range=[1.1, 1.9],
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        zca_whitening=True,
        height_shift_range=0.1,
        # preprocessing_function=random_color_jitter,
        # validation_split=0.2,
    )
    
    train_dataset = train_datagen.flow_from_directory(
        directory=train_data_path,
        target_size=(IMG_H, IMG_W),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=True,
        seed=32,
        # subset='training'
    )
        
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
    )
        
    valid_dataset = valid_datagen.flow_from_directory(
        directory=val_data_path,
        target_size=(IMG_H, IMG_W),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=True,
        seed=123,
        # subset='validation'
    )
    
    
    
    return train_dataset, valid_dataset


# In[ ]:


def get_callbacks(path_model, name_model):
    saver_callback = CustomSaver(
            path_model,
            name_model
        )
        
    checkpoint_filepath = SAVED_MODEL_PATH + name_model + '_weights.{epoch:02d}-{val_accuracy:.2f}.h5'
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=7, min_lr=0.000001)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    early_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        verbose=1, 
        patience=20, 
        restore_best_weights=True
    )

    
    return [
        saver_callback,
        # early_callback,
        reduce_lr,
        model_checkpoint_callback,
    ]


# In[ ]:


def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    
    for key in keys:
        score = total/(len(labels_dict)*float(labels_dict[key]))
        class_weight[key] = score
    
    return class_weight

def __run__(our_model, train_dataset, val_dataset, num_epochs, path_model, name_model, class_name):
    print("running", name_model)

    if TRAIN_MODE:
        amountofimages = {
            0: 7182,
            1: 3508,
            2: 3360,
            3: 3344,
            4: 3648,
            5: 3186,
            6: 3277,
            7: 3264
        }
        train_class_weights = create_class_weight(amountofimages)
        print(train_class_weights)
        train_class_weights = dict(enumerate(train_class_weights))
        callbacks_func = get_callbacks(path_model, name_model)

        fit_history_our_model = our_model.fit(
            train_dataset,
            epochs=num_epochs,
            validation_data=val_dataset,
            batch_size=BATCH_SIZE,
            # class_weight=train_class_weights,
            callbacks=callbacks_func,
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
    # evaluate_and_testing(our_model, "saved_model/110_pcb_150-resnet18_weights.84-0.90.h5", TEST_DATASET_PATH, class_name)    


# In[ ]:


if __name__ == "__main__":
    
    '''
    Model for computer-vision-based Printed Circuit Board (PCB).
    analysis dataset used for classification of defects.
    '''
    
    # run the function here
    
    name_model = str(IMG_H)+"_pcb_"+str(NUM_EPOCHS)
    if CHOOSEN_MODEL == 1:
        name_model = name_model + "-resnet18"
    elif CHOOSEN_MODEL == 2:
        name_model = name_model + "-resnet18-v2"
    elif CHOOSEN_MODEL == 3:
        name_model = name_model + "-VGG16"
        
    print("start: ", name_model)
    base_learning_rate = 0.00002
    num_classes = 8
    class_name = ["0", "1", "2", "3", "4", "5", "6", "7"]
    
    
    input_shape = (IMG_H, IMG_W, IMG_C)
    
    path_model = SAVED_MODEL_PATH + name_model + "_model" + ".h5"
    
    train_dataset, val_dataset = dataset_manipulation(TRAIN_DATASET_PATH, TEST_DATASET_PATH)
    # generate_image_dataset(TRAIN_DATASET_PATH, TEST_DATASET_PATH)
    
    our_model = build_resnet18(input_shape, base_learning_rate, num_classes)
    
    # our_model.summary()
        
    if CHOOSEN_MODEL == 2:
        our_model = build_our_model(input_shape, base_learning_rate, num_classes)
    elif CHOOSEN_MODEL == 3:
        our_model = build_VGG16(input_shape, base_learning_rate, num_classes)
    
    __run__(our_model, train_dataset, val_dataset, NUM_EPOCHS, path_model, name_model, class_name)


# In[ ]:




