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

IMG_H = 110
IMG_W = 42
IMG_C = 3  ## Change this to 1 for grayscale.
FORMAT_IMAGE = [".jpg",".png",".jpeg", ".bmp"]


# In[ ]:


def augment_dataset_batch_test(dataset_batch):
    AUTOTUNE = tf.data.AUTOTUNE
    
    
    flip_up_down = dataset_batch.map(lambda image, label: (tf.image.flip_up_down(image), label),
              num_parallel_calls=AUTOTUNE)

    flip_left_right = dataset_batch.map(lambda image, label: (tf.image.flip_left_right(image), label),
              num_parallel_calls=AUTOTUNE)
    
    rotate = dataset_batch.map(lambda image, label: (tf.image.rot90(image, k=2), label),
              num_parallel_calls=AUTOTUNE)
    
    

    dataset_batch = dataset_batch.concatenate(flip_up_down)
    dataset_batch = dataset_batch.concatenate(flip_left_right)
    dataset_batch = dataset_batch.concatenate(rotate)
    
    
    return dataset_batch


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

        if (epoch + 1) % 15 == 0 or (epoch + 1) <= 15:
            self.model.save(self.model_path)
            print('saved for epoch',epoch + 1)


# In[ ]:


def build_our_model(i_shape, base_lr, n_class):
    
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', input_shape=i_shape))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_class))
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr/10, beta_1=0.5, beta_2=0.999),
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
        seed=123,
        image_size=(IMG_H, IMG_W),
        batch_size=batch_size
    )
    
    
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
    probability_model = tf.keras.Sequential([this_model, 
                                         tf.keras.layers.Softmax()])
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
def dataset_manipulation(train_data_path):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_data_path,
#         validation_split=0.2,
#         subset="training",
#         seed=123,
        image_size=(IMG_H, IMG_W))

    class_names = train_dataset.class_names
    print("name of classes: ", class_names, ", Size of classes: ", len(class_names))
    
    train_dataset = augment_dataset_batch_test(train_dataset)
    
#     train_dataset = train_dataset.unbatch()
# #     print(len(list(train_dataset)))
#     train_dataset_dict = {}
#     top_number_of_dataset = 0
#     for a in range(0, 8):
#         filtered_dataset = train_dataset.filter(lambda x,y: tf.reduce_all(tf.equal(y, [a])))
#         len_current_dataset = len(list(filtered_dataset))
#         train_dataset_dict[a] = filtered_dataset
# #         train_dataset_dict.append(filtered_dataset)
#         if len_current_dataset > top_number_of_dataset:
#             top_number_of_dataset = len_current_dataset
        
#     print(top_number_of_dataset)

    
# #     train_dataset = train_dataset.filter(lambda x,y: tf.reduce_all(tf.not_equal(y, [1,2,3,4,5,6,7])))
# #     print(train_dataset)
# #     print(len(list(train_dataset)))
#     for a in range(0, 8):
#         current_dataset = train_dataset_dict[a]
#         len_current_dataset = len(list(train_dataset_dict[a]))
#         times_multi = int(top_number_of_dataset/len_current_dataset)
#         print("times_multi: ", times_multi)
#         current_dataset = current_dataset.repeat(times_multi)
#         print(len(list(current_dataset)))
#         train_dataset_dict[a] = current_dataset
    
#     final_dataset = train_dataset_dict[0]
#     for a in range (1, 8):
#         final_dataset = final_dataset.concatenate(train_dataset_dict[a])
        
#     final_dataset = final_dataset.batch(32).prefetch(2)
    
#     return final_dataset
    return train_dataset


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


    name_model = str(IMG_H)+"_pcb_"+str(num_epochs)
    print("start: ", name_model)
    base_learning_rate = 0.0002
    num_classes = 8
    class_name = ["0", "1", "2", "3", "4", "5", "6", "7"]
    
    # set dir of files
    train_data_path = "image_dataset/training_dataset"
    test_data_path = "image_dataset/evaluation_dataset"
    saved_model_path = "saved_model/"
    
    input_shape = (IMG_H, IMG_W, IMG_C)
    # print(input_shape)

    path_model = saved_model_path + name_model + "_model" + ".h5"
    
#     print(train_dataset)

    our_model = build_our_model(input_shape, base_learning_rate, num_classes)

#     our_model.summary()
    
    saver_callback = CustomSaver(
        path_model,
        name_model
    )
    
    train_dataset = dataset_manipulation(train_data_path)
    
    
    print(train_dataset)

    
    y = np.concatenate([y for x, y in train_dataset], axis=0)
    class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(y), 
            y)
    
    train_class_weights = dict(enumerate(class_weights))
    
    print("class_weights: ", train_class_weights)
    
    fit_history = our_model.fit(
        train_dataset,
        epochs=num_epochs,
#         validation_data=val_ds,
        class_weight=train_class_weights,
        callbacks=[saver_callback]   
    )
    
    evaluate_and_testing(our_model, path_model, test_data_path, class_name)


# In[ ]:




