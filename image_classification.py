#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import itertools
import tensorflow as tf
import numpy as np
import os
import csv
     
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
from packaging import version
from matplotlib import pyplot as plt

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2,     "This notebook requires TensorFlow 2.0 or above."


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
    batch_size = 32
    num_epochs = 10
    
    IMG_H = 110
    IMG_W = 42
    IMG_C = 3  ## Change this to 1 for grayscale.

    name_model = str(IMG_H)+"_pcb_"+str(num_epochs)
    print("start: ", name_model)
    base_learning_rate = 0.0002
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
#         validation_split=0.2,
#         subset="training",
#         seed=123,
        image_size=(IMG_H, IMG_W),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print("name of classes: ", class_names, "Size of classes: ", len(class_names))
    
    y = np.concatenate([y for x, y in train_ds], axis=0)
    class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(y), 
            y)
    
    train_class_weights = dict(enumerate(class_weights))
#     val_ds = tf.keras.utils.image_dataset_from_directory(
#         train_data_path,
#         validation_split=0.2,
#         subset="validation",
#         seed=123,
#         image_size=(IMG_H, IMG_W),
#         batch_size=batch_size)
    
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
      # tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

#     normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1,input_shape=(IMG_H, IMG_W, IMG_C))
    normalization_layer = tf.keras.layers.Rescaling(1./255,input_shape=(IMG_H, IMG_W, IMG_C))

    our_model = tf.keras.models.Sequential()
#     our_model.add(data_augmentation)
    our_model.add(normalization_layer)

    our_model.add(tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same'))
    our_model.add(tf.keras.layers.LeakyReLU())
    our_model.add(tf.keras.layers.BatchNormalization())

    our_model.add(tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same'))
    our_model.add(tf.keras.layers.LeakyReLU())
    our_model.add(tf.keras.layers.BatchNormalization())
    our_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    our_model.add(tf.keras.layers.Dropout(0.2))

    our_model.add(tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same'))
    our_model.add(tf.keras.layers.LeakyReLU())
    our_model.add(tf.keras.layers.BatchNormalization())

    our_model.add(tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same'))
    our_model.add(tf.keras.layers.LeakyReLU())
    our_model.add(tf.keras.layers.BatchNormalization())
    our_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    our_model.add(tf.keras.layers.Dropout(0.3))

    our_model.add(tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same'))
    our_model.add(tf.keras.layers.BatchNormalization())

    our_model.add(tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same'))
    our_model.add(tf.keras.layers.LeakyReLU())
    our_model.add(tf.keras.layers.BatchNormalization())
    our_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    our_model.add(tf.keras.layers.Dropout(0.4))
    our_model.add(tf.keras.layers.Flatten())

    our_model.add(tf.keras.layers.Dense(128, kernel_initializer='he_uniform'))
    our_model.add(tf.keras.layers.LeakyReLU())
    our_model.add(tf.keras.layers.BatchNormalization())
    our_model.add(tf.keras.layers.Dropout(0.5))
    our_model.add(tf.keras.layers.Dense(num_classes))

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
#         validation_data=val_ds,
        class_weight=train_class_weights,
        callbacks=[saver_callback]
        
    )
    
    
    print("class_weights: ", class_weights)
    
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
    
    evaluation_ds = evaluation_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Evaluate the model on the test data using `evaluate`
    # You can also evaluate or predict on a dataset.
    print("Evaluate")
    result = our_model.evaluate(evaluation_ds)
    print(result)
    dict(zip(our_model.metrics_names, result))

    """
    Testing Area
    """
    
    pred_list = []
    name_image_list = []
    label_list = []
    format_image = [".jpg",".png",".jpeg"]
    
    probability_model = tf.keras.Sequential([our_model, 
                                         tf.keras.layers.Softmax()])
    for class_n in class_names: 
        path = os.path.join(test_data_path,class_n)  
        class_num = class_names.index(class_n)  

        for img in tqdm(os.listdir(path)):  
            if img.endswith(tuple(format_image)):
                filepath = os.path.join(path, img)
                name_image = os.path.basename(filepath)
#                 print("name image: ", name_image, "label image: ", class_num)
                
                img = tf.keras.utils.load_img(
                    filepath, target_size=(IMG_H, IMG_W)
                )
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0) # Create a batch

                pred_result = probability_model.predict(img_array)
                # Generate arg maxes for predictionsß
                   
                pred_classes = np.argmax(pred_result[0])
                
                pred_list.append(pred_classes)
                name_image_list.append(name_image)
                label_list.append(class_num)

    final_result = zip(name_image_list, label_list, pred_list)
    for n, l, p in final_result:
         print("name image: ", n, "label image: ", l, "prediction class: ", p)
            
#     print("final result: ", final_result)
    confusion_matrix_report(label_list, pred_list, class_names)
    
    
    print("created csv for the result.")
    with open('predictions_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ImageName', 'Label'])
        writer.writerows(zip(name_image_list, pred_list))


# In[ ]:




