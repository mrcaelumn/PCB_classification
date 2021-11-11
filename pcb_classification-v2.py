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
import tf_clahe
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
CHOOSEN_MODEL = 3 # 1 == our model, 2 == mobilenet, 3 == resnet18
IMG_H = 110
IMG_W = 42
IMG_C = 3  ## Change this to 1 for grayscale.
COLOUR_MODE = "rgb"
BATCH_SIZE = 32

# set dir of files
TRAIN_DATASET_PATH = "image_dataset_final/training_dataset/"
TEST_DATASET_PATH = "image_dataset_final/evaluation_dataset/"
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
    tf.keras.layers.RandomFlip("horizontal_and_vertical", input_shape=(IMG_H, IMG_W, IMG_C)),
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
    
    model.add(tf.keras.layers.Dense(1024,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    
    model.add(tf.keras.layers.Dense(1024,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    
    
    model.add(tf.keras.layers.Dense(512,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(tf.keras.layers.LeakyReLU())
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

    model.add(tf.keras.layers.Dense(512,
                                    kernel_initializer="he_uniform"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_class, activation="softmax"))
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr),
                  metrics=['accuracy'])
    
    return model


# In[ ]:


class ResnetBlock(tf.keras.models.Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = tf.keras.layers.Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.merge = tf.keras.layers.Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = tf.keras.layers.Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18(tf.keras.models.Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = tf.keras.layers.Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = tf.keras.layers.BatchNormalization()
        self.pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.flat = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

def get_model_resnet18(n_class):
    resnet = ResNet18(num_classes=n_class)
    resnet.build(input_shape = (None, IMG_H, IMG_W, IMG_C))
    return resnet

def our_resnet18(i_shape, base_lr, n_class):
    model = tf.keras.models.Sequential()
    resnet = get_model_resnet18(n_class)
    
    
    if AUGMENTATION:
        model.add(data_augmentation)
        
    model.add(resnet)
    model.add(tf.keras.layers.Dense(128,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    
    
    model.add(tf.keras.layers.Dense(n_class
                                    ,activation="softmax"
                                   ))
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr),
                  metrics=['accuracy'])
    
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


def dataset_manipulation(train_data_path, val_data_path):
    

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # rotation_range=20,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        # vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        
    )
    for i, one_class in enumerate(os.listdir(TRAIN_DATASET_PATH)):
        train_dataset = train_datagen.flow_from_directory(
            directory=train_data_path,
            target_size=(IMG_H, IMG_W),
            color_mode="rgb",
            batch_size=BATCH_SIZE,
            class_mode="sparse",
            # shuffle=True,
            seed=42,
        )
        
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
    )
    
    valid_dataset = validation_datagen.flow_from_directory(
        directory=val_data_path,
        target_size=(IMG_H, IMG_W),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        # shuffle=True,
        seed=42
    )
    
    if COLOUR_MODE == "grayscale":
    
        f_train_dataset = tf.data.Dataset.from_generator(
            lambda: train_dataset,
            output_types = (tf.float32, tf.int64),
            output_shapes = ([None, IMG_H, IMG_H, IMG_C], [None,]),
        )
        f_valid_dataset = tf.data.Dataset.from_generator(
            lambda: valid_dataset,
            output_types = (tf.float32, tf.int64),
            output_shapes = ([None, IMG_H, IMG_H, IMG_C], [None,]),
        )
        
        # plt.figure(figsize=(10, 10))
        # for images, labels in f_valid_dataset.take(1):
        #     for i in range(9):
        #         ax = plt.subplot(3, 3,i+1)
        #         plt.imshow(images[i].numpy().astype("uint8"))
        #         plt.title(CLASS_NAME[labels[i]])
        #         plt.axis("off")
        
        train_ds = (
            f_train_dataset
            # .shuffle(1000)
            .map(prep_image, num_parallel_calls=AUTOTUNE)
            # .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )
        
        valid_ds = (
            f_valid_dataset
            # .shuffle(1000)
            .map(prep_image, num_parallel_calls=AUTOTUNE)
            # .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )

        
        # plt.figure(figsize=(10, 10))
        # for images, labels in valid_ds.take(1):
        #     for i in range(9):
        #         ax = plt.subplot(3, 3,i+1)
        #         plt.imshow(images[i].numpy().astype("uint8"))
        #         plt.title(CLASS_NAME[labels[i]])
        #         plt.axis("off")
                
        return train_ds, valid_ds
    
    
    return train_dataset, valid_dataset


# In[ ]:


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn


def __run__(our_model, train_dataset, val_dataset, num_epochs, path_model, name_model, class_name):
    print("running", name_model)
#     y = np.concatenate([y for x, y in train_dataset], axis=0)
#     print(dict(zip(*np.unique(y, return_counts=True))))
#     class_weights = class_weight.compute_class_weight(
#         class_weight='balanced',
#         classes=np.unique(y), 
#         y=y
#     )
    
#     train_class_weights = dict(enumerate(class_weights))
    
#     print("class_weights: ", train_class_weights)

    if TRAIN_MODE:
        saver_callback = CustomSaver(
            path_model,
            name_model
        )

        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            verbose=1, 
            patience=15, 
            restore_best_weights=True
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
                es
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
        name_model = name_model + "-resnet18"
        
    print("start: ", name_model)
    base_learning_rate = 0.00001
    num_classes = 8
    class_name = ["0", "1", "2", "3", "4", "5", "6", "7"]
    
    
    input_shape = (IMG_H, IMG_W, IMG_C)
    
    path_model = SAVED_MODEL_PATH + name_model + "_model" + ".h5"
    
    train_dataset, val_dataset = dataset_manipulation(TRAIN_DATASET_PATH, TEST_DATASET_PATH)
    # generate_image_dataset(TRAIN_DATASET_PATH, TEST_DATASET_PATH)
    
        
    """
    our custom model
    """ 
    our_model = build_our_model(input_shape, base_learning_rate, num_classes)
    # our_model.summary()
        
    if CHOOSEN_MODEL == 2:
        """
        mobilenet
        """
        our_model = our_mobilenet(input_shape, base_learning_rate, num_classes)
    elif CHOOSEN_MODEL == 3:
        """
        mobilenet
        """
        our_model = our_resnet18(input_shape, base_learning_rate, num_classes)
    
    __run__(our_model, train_dataset, val_dataset, NUM_EPOCHS, path_model, name_model, class_name)


# In[ ]:




