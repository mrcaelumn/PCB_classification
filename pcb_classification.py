#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing Neccessary Library and constant variable

# !pip install tf_clahe
# !pip install -U scikit-learn
# !pip install matplotlib
# !pip install pandas


# In[ ]:


import tensorflow as tf
import tensorflow_io as tfio

import numpy as np
import os
import csv
import tf_clahe

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
from packaging import version
from matplotlib import pyplot as plt

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2,     "This notebook requires TensorFlow 2.0 or above."

IMG_H = 110
IMG_W = 42
IMG_C = 3  ## Change this to 1 for grayscale.
BATCH_SIZE = 32
FORMAT_IMAGE = [".jpg",".png",".jpeg", ".bmp"]
HIGH_CLASS = [0]
LOW_CLASS = [1 ,2, 3, 4 , 5, 6, 7]
AUTOTUNE = tf.data.AUTOTUNE
AUGMENTATION = True
AUGMENTATION_REPEAT = False


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

@tf.function
def random_color_jitter(tensor):
    if  tf.random.uniform([]) < 0.2:
        tensor = color_jitter(tensor)
    return tensor

@tf.function
def random_rgb_to_bgr(tensor):
    if  tf.random.uniform([]) < 0.2:
        tensor = tfio.experimental.color.rgb_to_bgr(tensor)
    return tensor


@tf.function
def random_hue(tensor):
    if  tf.random.uniform([]) < 0.2:
        tensor = tf.image.random_hue(tensor, 0.2)
    return tensor

def rescale_dataset(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 255.0)
    return image, label

# def gray_to_rgb(img):
#     return np.repeat(img, 3, 2)
@tf.function
def enchantment_image(image):
    image = tf.cast(image, tf.float32)
    image = (image / 255.0) # range 0 to 1
    # image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    image = tf_clahe.clahe(image)
    
    return image

@tf.function
def enchantment_dataset(dataset_batch):
    final_dataset = dataset_batch.map(lambda image, label: (enchantment_image(image), label),
                                     num_parallel_calls=AUTOTUNE)
    return final_dataset

@tf.function
def augment_dataset_batch_test(dataset_batch):
    
    
    flip_up_down = dataset_batch.map(lambda image, label: (tf.image.flip_up_down(image), label),
                                     num_parallel_calls=AUTOTUNE)

    flip_left_right = dataset_batch.map(lambda image, label: (tf.image.flip_left_right(image), label),
                                        num_parallel_calls=AUTOTUNE)
    
#     rgb_to_bgr = dataset_batch.map(lambda image, label: (tfio.experimental.color.rgb_to_bgr(image), label),
#                                         num_parallel_calls=AUTOTUNE)
    
#     rgb_to_xyz = dataset_batch.map(lambda image, label: (tfio.experimental.color.rgb_to_xyz(image), label),
#                                         num_parallel_calls=AUTOTUNE)
    
    # rgb_to_ycbcr = dataset_batch.map(lambda image, label: (tfio.experimental.color.rgb_to_ycbcr(image), label),
    #                                     num_parallel_calls=AUTOTUNE)
    # random_hue = dataset_batch.map(lambda image, label: (tf.image.random_hue(image, 0.2), label),
    #                                       num_parallel_calls=AUTOTUNE)
    
    dataset_batch = dataset_batch.concatenate(flip_up_down)
    dataset_batch = dataset_batch.concatenate(flip_left_right)
    # dataset_batch = dataset_batch.concatenate(colour_jitter)
    # dataset_batch = dataset_batch.concatenate(rgb_to_bgr)
    # dataset_batch = dataset_batch.concatenate(rgb_to_xyz)
    # dataset_batch = dataset_batch.concatenate(rgb_to_ycbcr)
    # dataset_batch = dataset_batch.concatenate(random_hue)
    # dataset_batch = dataset_batch.concatenate(adjust_brightness)
    # dataset_batch = dataset_batch.concatenate(adjust_saturation)
    
    return dataset_batch


# In[ ]:


"""
Custom Layer
"""
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    #tf.keras.layers.Lambda(random_color_jitter),
    #tf.keras.layers.Lambda(random_rgb_to_bgr),
    # tf.keras.layers.Lambda(random_hue),
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


def build_our_model(i_shape, base_lr, n_class):
    
    model = tf.keras.models.Sequential()
    
    if AUGMENTATION:
        model.add(data_augmentation)
        
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))
    
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_class, activation="tanh"))
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr),
                  metrics=['accuracy'])
    
    return model


# In[ ]:


def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.SeparableConv2D(filters, 3, padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ])
    
    return block

def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    
    return block

def build_our_model_v2(i_shape, base_lr, n_class):
    
    model = tf.keras.models.Sequential()
    
    if AUGMENTATION:
        model.add(data_augmentation)
        
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=i_shape))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    model.add(conv_block(64))
    model.add(conv_block(128))
    
    model.add(conv_block(256))
    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(conv_block(512))
    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.Flatten())
    
    model.add(dense_block(1024, 0.7))
    model.add(dense_block(512, 0.5))
    model.add(dense_block(256, 0.3))
    
    model.add(tf.keras.layers.Dense(n_class, activation="tanh"))
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr),
                  metrics=['accuracy'])
    
    return model


# In[ ]:


def our_resnet50(i_shape, base_lr, n_class):
    model = tf.keras.models.Sequential()
    
    base_model = tf.keras.applications.ResNet50V2(weights='imagenet', input_shape=i_shape, include_top=False)

    for layer in base_model.layers:
        layer.trainable = False
        
    if AUGMENTATION:
        model.add(data_augmentation)    
    
    model.add(base_model)
    
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(n_class, activation="tanh"))
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr),
                  metrics=['accuracy'])
    
    return model


# In[ ]:


def our_efficientnet(i_shape, base_lr, n_class):
    model = tf.keras.models.Sequential()
    
    base_model = tf.keras.applications.efficientnet.EfficientNetB0(input_shape = i_shape, include_top = False, weights = "imagenet", classes=n_class)
    # base_model.summary()
    # base_model.trainable = True
        
    
    if AUGMENTATION:
        model.add(data_augmentation)
        
    
    model.add(base_model)
    
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_class, activation="tanh"))
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr),
                  metrics=['accuracy'])
    
    return model


# In[ ]:


def our_desnet(i_shape, base_lr, n_class):
    model = tf.keras.models.Sequential()
    
    base_model = tf.keras.applications.DenseNet121(input_shape = i_shape, include_top = False, weights = "imagenet")
    # base_model.summary()
    # base_model.trainable = True
        
    
    if AUGMENTATION:
        model.add(data_augmentation)
        
    
    model.add(base_model)
    
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    
    # model.add(tf.keras.layers.Dense(512))
    # model.add(tf.keras.layers.LeakyReLU())
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_class, activation="tanh"))
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr),
                  metrics=['accuracy'])
    
    return model


# In[ ]:


def evaluate_and_testing(this_model, p_model, test_dataset_path, c_names):
    """
    Evaluation Area
    """
    this_model.load_weights(p_model)
    evaluation_ds = tf.keras.utils.image_dataset_from_directory(
        test_dataset_path,
#         seed=123,
        image_size=(IMG_H, IMG_W),
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )
    
    evaluation_ds = enchantment_image(evaluation_ds)
    # Evaluate the model on the test data using `evaluate`
    # You can also evaluate or predict on a dataset.
    print("Evaluate")
    result = this_model.evaluate(evaluation_ds)
    print(result)
    dict(zip(this_model.metrics_names, result))

    """
    Testing Area
    """
    
    pred_list = []
    name_image_list = []
    label_list = []
    
    probability_model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(scale=1./255, input_shape=(IMG_H, IMG_W, IMG_C)), 
        this_model
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

    final_result = zip(name_image_list, label_list, pred_list)
    for n, l, p in final_result:
         print("name image: ", n, "label image: ", l, "prediction class: ", p)
            
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
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_data_path,
        # validation_split=0.15,
        # subset="training",
        # seed=123,
        color_mode="grayscale",
        image_size=(IMG_H, IMG_W))

    
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_data_path,
        # validation_split=0.15,
        # subset="validation",
        # seed=123,
        color_mode="grayscale",
        image_size=(IMG_H, IMG_W))
    
    class_names = train_dataset.class_names
    print("name of classes: ", class_names, ", Size of classes: ", len(class_names))
    
    # print("Before: ")
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_dataset.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3,i+1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    # print(train_dataset)
    
    print("enchantment_dataset")
    enchantmented_train_dataset = enchantment_dataset(train_dataset)
    enchantmented_val_dataset = enchantment_dataset(val_dataset)
    

    # print(train_dataset)
    # print("after: ")
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_dataset.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i+1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    
    
    
    
    # if AUGMENTATION_REPEAT:
    #     train_dataset = augment_dataset_batch_test(train_dataset)
    #     val_dataset = augment_dataset_batch_test(val_dataset)
    
    
    
    # return train_dataset, val_dataset
    if AUGMENTATION_REPEAT:
        print("AUGMENTATION_REPEAT")
        enchantmented_train_dataset = enchantmented_train_dataset.unbatch()

    #     print(len(list(train_dataset)))
        train_dataset_dict = {}
        top_number_of_dataset = 0
        # print("before preprocessing")
        for a in range(0, 8):
            filtered_dataset = enchantmented_train_dataset.filter(lambda x,y: tf.reduce_all(tf.equal(y, [a])))
            len_current_dataset = len(list(filtered_dataset))
            print("class: ", a, len_current_dataset)
            if a in LOW_CLASS:
                filtered_dataset = augment_dataset_batch_test(filtered_dataset)

            train_dataset_dict[a] = filtered_dataset


        # print("after preprocessing")
        final_dataset = train_dataset_dict[0]
        len_current_dataset = len(list(final_dataset))
        print("class: ", 0, len_current_dataset)
        for a in range (1, 8):
            len_current_dataset = len(list(train_dataset_dict[a]))
            print("class: ", a, len_current_dataset)
            final_dataset = final_dataset.concatenate(train_dataset_dict[a])

        enchantmented_train_dataset = final_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    train_dataset = enchantmented_train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    val_dataset = enchantmented_val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    return train_dataset, val_dataset


# In[ ]:


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn


def __run__(our_model, train_dataset, val_dataset, num_epochs, path_model, name_model, class_name):
    
    y = np.concatenate([y for x, y in train_dataset], axis=0)
    print(dict(zip(*np.unique(y, return_counts=True))))
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y), 
        y=y
    )
    
    train_class_weights = dict(enumerate(class_weights))
    
    print("class_weights: ", train_class_weights)

    
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
        # batch_size=BATCH_SIZE,
#         class_weight=train_class_weights,
        callbacks=[
            saver_callback, 
            # lr_scheduler
        ]   
    )
    
    evaluate_and_testing(our_model, path_model, test_data_path, class_name)


# In[ ]:


if __name__ == "__main__":
    
    '''
    Model for computer-vision-based Printed Circuit Board (PCB).
    analysis dataset used for classification of defects.
    '''
    
    # run the function here
    """ Set Hyper parameters """
    num_epochs = 100
    choosen_model = 5 # 1 == our model, 2 == resnet50, 3 == efficientnet, 4 == desnet, 5 == custom_model_v2
    
    name_model = str(IMG_H)+"_pcb_"+str(num_epochs)
    
    if choosen_model == 1:
        name_model = name_model + "-custom_model"
    elif choosen_model == 2:
        name_model = name_model + "-resnet50"
    elif choosen_model == 3:
        name_model = name_model + "-efficientnet"
    elif choosen_model == 4:
        name_model = name_model + "-desnet"
    elif choosen_model == 5:
        name_model = name_model + "-custom_model_v2"
        
    print("start: ", name_model)
    base_learning_rate = 0.00002
    num_classes = 8
    class_name = ["0", "1", "2", "3", "4", "5", "6", "7"]
    
    # set dir of files
    train_data_path = "image_dataset_ori/training_dataset"
    test_data_path = "image_dataset_ori/evaluation_dataset"
    saved_model_path = "saved_model/"
    
    input_shape = (IMG_H, IMG_W, IMG_C)
    
    path_model = saved_model_path + name_model + "_model" + ".h5"
    
    train_dataset, val_dataset = dataset_manipulation(train_data_path, test_data_path)
    
    if choosen_model == 1:
        """
        our custom model
        """ 
        print("running", name_model)
        our_model = build_our_model(input_shape, base_learning_rate, num_classes)
        # our_model.summary()
        __run__(our_model, train_dataset, val_dataset, num_epochs, path_model, name_model, class_name)
    elif choosen_model == 2:
        """
        resnet50
        """
        print("running", name_model)
        our_model = our_resnet50(input_shape, base_learning_rate, num_classes)
        __run__(our_model, train_dataset, val_dataset, num_epochs, path_model, name_model, class_name)
    elif choosen_model == 3:
        """
        efficientnet
        """
        print("running", name_model)
        our_model = our_efficientnet(input_shape, base_learning_rate, num_classes)
        __run__(our_model, train_dataset, val_dataset,num_epochs, path_model, name_model, class_name)
    elif choosen_model == 4:
        """
        desnet
        """
        print("running", name_model)
        our_model = our_desnet(input_shape, base_learning_rate, num_classes)
        __run__(our_model, train_dataset, val_dataset,num_epochs, path_model, name_model, class_name)
    elif choosen_model == 5:
        """
        our custom model v2
        """
        print("running", name_model)
        our_model = build_our_model_v2(input_shape, base_learning_rate, num_classes)
        __run__(our_model, train_dataset, val_dataset,num_epochs, path_model, name_model, class_name)

