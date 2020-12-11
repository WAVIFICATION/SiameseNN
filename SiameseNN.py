# -*- coding: utf-8 -*-
"""
Program to design siamese neural network on contrastive and triplet loss.
The models are trained and evaluated in main funtion.
Run main function to execute program.
Done by:
    JERIN JACOB
    N10337717
"""

#install libraries
# !pip install -q tfds-nightly 
#imports
import tensorflow_datasets as tfds
from keras.regularizers import l2
import tensorflow as tf
import numpy as np
import random
import keras
import time

from sklearn.utils import shuffle

from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, Dense, Flatten, Activation, Dropout, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from keras.optimizers import RMSprop
import numpy as np
from numpy import random as np_random

def tfds_to_numpy(ds):
    '''  
    Function to convert tensorflow dataset object to numpy variable suitable for the usecase
    
    @param ds: 
        a tfds object

    @return
       A numpy object that has the data corresponding to the dataset input  
    '''
    df_numpy=tfds.as_numpy(ds) # numpy generator
    dict_of_char={}# alphabet + alphabet_char_id used as key, value = greyscale image
    for x in df_numpy:
        # print(x)
        t=str(x['alphabet'])+' '+str(x['alphabet_char_id'])
        if t not in dict_of_char.keys():
            dict_of_char[t]=[]
        img=x['image']
        dimention1=img[:,:,0]//255# converting to greyscale
        dict_of_char[t].append(dimention1.reshape((105,105)))
    return np.array(list(dict_of_char.values())) 


def get_batch(batch_size,ds):
    """
    Function to generate batches for model that uses contrastive loss.
    The batches have targets and pairs.
    First half of the batch has target=0, and pairs have images from different alphabet.
    Second half has targets=1, and pairs have images from same alphabet.

    Args:
        batch_size (integer): the number of rows of data that needs to be generated.
        ds (numpy array): The dataset from which batches must be generated from.

    Returns:
        pairs: pairs of images for comparison
        targets: 1 if images are from same alphabet, else 0
    """
    n_classes, n_examples, w, h = ds.shape
    cat = np_random.choice(n_classes, size=batch_size, replace=False)
    # print(cat)
    targets = np.zeros((batch_size,))
    targets[batch_size//2:] = 1# 1st half with disimilar images, and 2nd half with similar
    pairs = [np.zeros((batch_size,w,h,1)) for _ in range(2)]
    for i in range(batch_size):
        ex_no = np_random.randint(n_examples)
        pairs[0][i,:,:,:] = ds[cat[i],ex_no,:,:].reshape(w,h,1)
        cat2 = 0  
        if i >= batch_size // 2:
            cat2 = cat[i]
        else:
            cat2 = (cat[i] + np_random.randint(1,n_classes)) % n_classes
        ex_no2 = np_random.randint(n_examples)
        pairs[1][i,:,:,:] = ds[cat2,ex_no2,:,:].reshape(w,h,1)
    return pairs,targets

def get_batch_triplet(batch_size,ds):
    """
    Function to generate batches for model that uses triplet loss.
    The batches have targets and pairs.
    pairs[0] has the anchor images.
    pairs[1] has the positive images compared to anchor images.
    pairs[2] has the negative images compared to anchor images.
    targets are placeholders.

    Args:
        batch_size (integer): the number of rows of data that needs to be generated.
        ds (numpy array): The dataset from which batches must be generated from.

    Returns:
        pairs: pairs of images for comparison.
        targets: array of 1.
    """
    n_classes, n_examples, w, h = ds.shape
    cat = np_random.choice(n_classes, size=batch_size, replace=False)
    targets = np.ones((batch_size,))
    pairs = [np.zeros((batch_size,w,h,1)) for _ in range(3)]

    for i in range(batch_size):
        ex_no = np_random.randint(n_examples)
        pairs[0][i,:,:,:] = ds[cat[i],ex_no,:,:].reshape(w,h,1)

        cat2 = cat[i]
        ex_no2 = np_random.randint(n_examples)
        pairs[1][i,:,:,:] = ds[cat2,ex_no2,:,:].reshape(w,h,1)

        cat2 = (cat[i] + np_random.randint(1,n_classes)) % n_classes
        ex_no2 = np_random.randint(n_examples)
        pairs[2][i,:,:,:] = ds[cat2,ex_no2,:,:].reshape(w,h,1)
    return pairs,targets

def euclidean_distance(vectors):
    """Function to calculate euclidean distance between 2 vectors.

    Args:
        vectors : array of vectors.

    Returns:
        euclidean distance between vectors
    """
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
def contrastive_loss(y_true, y_pred):
    """Contrastive loss function

    Args:
        y_true : expected output of model
        y_pred : actual output of model

    Returns:
        loss value
    """
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(1 - y_pred, 0)))
    
def accuracy(y_true, y_pred):
    '''
    function to calculate classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """Triplet loss function

    Args:
        y_true : expected output of model
        y_pred : actual output of model
        alpha (float, optional): Defaults to 0.2.

    Returns:
        Triplet loss
    """
    total_lenght = y_pred.shape.as_list()[-1]
    anchor, positive, negative = y_pred[:,:int(1/3*total_lenght)], y_pred[:,int(1/3*total_lenght):int(2/3*total_lenght)], y_pred[:,int(2/3*total_lenght):]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)




def get_siamese(input_shape):
    """
    Function to generate siamese network for contrastive loss

    Args:
        input_shape (tuple): the shape of input images

    Returns:
        siamese neural network
    """
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    model = Sequential()
    model.add(Conv2D(64,(5,5),input_shape=input_shape,activation='relu',kernel_regularizer='l2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2,strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128,(5,5),kernel_regularizer='l2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2,strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128,(5,5),kernel_regularizer='l2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2,strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    
    model.add(Dense(512,activation='sigmoid',kernel_regularizer='l2'))
    
    left_emb = model(left_input)
    right_emb = model(right_input)
    
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([left_emb, right_emb])

    siamese_net = Model([left_input, right_input], distance)
    
    return siamese_net

def get_siamese_trip(input_shape):
    """
    Function to generate siamese network for triplet loss

    Args:
        input_shape (tuple): the shape of input images

    Returns:
        siamese neural network
    """
    anchor_input = Input(input_shape)# recieves the anchor image
    pos_input = Input(input_shape)# recieves the positive image
    neg_input = Input(input_shape)# recieves the negative image
    
    model = Sequential()
    model.add(Conv2D(64,(5,5),input_shape=input_shape,activation='relu',kernel_regularizer='l2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2,strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128,(5,5),kernel_regularizer='l2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2,strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128,(5,5),kernel_regularizer='l2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2,strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    
    model.add(Dense(512,activation='sigmoid',kernel_regularizer='l2'))
    

    anchor_emb = model(anchor_input)
    pos_emb = model(pos_input)
    neg_emb = model(neg_input)
    

    output=concatenate([anchor_emb, pos_emb, neg_emb], axis=-1)

    siamese_net = Model([anchor_input, pos_input, neg_input], output)
    
    return anchor_input, anchor_emb, siamese_net


def test_contrastive(model, ds, batchSize):
    """ 
    Method to test the siamese network for contrastive loss.
    It uses  batches generated by get_batch method as validation data.

    Args:
        model : the model on which tests should be done.
        ds (numpy arrray): dataset.
        batchSize : size of batches for individual tests.

    Returns:
        prints overall accuracy of the model.
    """
    x,y = get_batch(batchSize, ds)
    model.evaluate(x,y,batch_size=batchSize,verbose=1,workers=6,use_multiprocessing=True)

def map_location(model, ds):
    """
    method to get the approximate vector for first image of every alphabet that can be used for comparison.

    Args:
        model : the model on which the vector must be made with.
        ds (numpy arrray): dataset.

    Returns:
        Dict having keys as alphabet class and values as the vector array.
    """
    n_classes, n_examples, w, h = ds.shape
    dict_map={}
    for i in range(n_classes):
        t=np.zeros((1,w,h,1))
        t[0]=ds[i,0,:,:].reshape(105,105,1)
        dict_map[i]=model.predict(t)
    return dict_map


def get_closest(vec, dictionary):
    """
    Method to identify the closest alphabet class for a vector 'vec' generated using the embedded model.

    Args:
        vec : vector array generated by embedded model.
        dictionary : cc

    Returns:
        alphabet classname to which the vector is closest to.
    """
    max_dist = 10000000
    max_name = None
    for name in dictionary:
        dist=np.linalg.norm(dictionary[name]-vec)
        if max_dist > dist:
            max_dist = dist
            max_name = name
    return max_name

def triple_model_test(model, ds, dict_map, same=True):
    """
    The method to test triplet model.
    Here, 2 images are considered(similar or disimilar).
    The 'get_closest' method identifies the closest alphabet for the images and checks if they are same.

    Args:
        model : the model on which tests should be done.
        ds (numpy arrray): dataset.
        dict_map : Dict having keys as alphabet class and values as the vector array geberated using 'map_location'.
        same (bool, optional): If true, selects images of same alphabet, else choses images from different alphabet. Defaults to True.

    Returns:
        [type]: [description]
    """
    n_classes, n_examples, w, h = ds.shape
    x1=random.randint(0,n_classes-1)
    ex_no = np_random.randint(n_examples)
    t1=np.zeros((1,w,h,1))
    t1[0]=ds[x1,ex_no,:,:].reshape(105,105,1)
    map1=model.predict(t1)
    if same== True:
        x2=x1#selects 2nd image of same alphabet
    else:
        x2=random.randint(0,n_classes)#selects 2nd image from different alphabet
    ex_no = np_random.randint(n_examples)
    t2=np.zeros((1,w,h,1))
    t2[0]=ds[x2,ex_no,:,:].reshape(105,105,1)
    map2=model.predict(t2)
                       
    if(get_closest(map1, dict_map)==get_closest(map2, dict_map)):
        return 1 if same==True else 0
    else:
        return 0 if same==True else 1


def validation_triplet(anchor_model, ds, n_test):
    """
    Method to perform 'triple_model_test' on the model trained using triplet loss function 'n_test' times.

    Args:
        anchor_model : the model on which tests should be done.
        ds (numpy arrray): dataset.
        n_test : Number of tests to be performed
    """
    per=0
    dict_map=map_location(anchor_model, ds)
    for i in range(n_test):
        per+=triple_model_test(anchor_model, ds_train, dict_map, bool(random.getrandbits(1)))
    print(per/n_test)



if __name__ == "__main__":
    ds, ds_info = tfds.load('omniglot', split=['train','test'],with_info=True)
    ds_train=tfds_to_numpy(ds[0])# training dataset
    ds_test=tfds_to_numpy(ds[1])# testing dataset
    ds_test_train=np.concatenate((ds_train, ds_test), axis=0)# combined testing and training dataset



    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.05,
        decay_steps=4000,
        decay_rate=0.0001)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    rms = RMSprop()

    model=get_siamese((105,105,1))# siamese model for training using contrastive loss
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    print('#########################################################')
    print('Siamese network for contrastive loss:')
    model.summary()

    anchor_input_layer, anchor_emb_model, triplet_model=get_siamese_trip((105,105,1))
    triplet_model.compile(loss=triplet_loss, optimizer="adam")# siamese model for training using triplet loss
    print('#########################################################')
    print('Siamese network for triplet loss:')
    triplet_model.summary()
    anchor_model=Model(inputs = anchor_input_layer, outputs=anchor_emb_model)# evaluation model for triplet loss siamese network, using the embedded layer as output layer.

    num_iterations = 7000
    batch_size = 128

    evaluateEvery = 200
    currTime = time.time()
    print('#########################################################')
    print('Siamese network for contrastive loss, Training:')
    for i in range(0,num_iterations+1):
        x,y = get_batch(batch_size, ds_train)
        loss = model.train_on_batch(x,y)
        if i % evaluateEvery == 0:# To print the loss after evey 'evaluateEvery' number of training iteration
            model.save('contrastive_'+str(i))
            print('Iteration',i,'('+str(round(time.time() - currTime,1))+'s) - Loss:',loss[0],'Acc:',round(loss[1],2))
            currTime = time.time()


    num_iterations = 200
    evaluateEvery = 50
    currTime = time.time()
    print('#########################################################')
    print('Siamese network for triplet loss, Training:')
    for i in range(0,num_iterations+1):
        x,y = get_batch_triplet(batch_size, ds_train)
        loss = triplet_model.train_on_batch(x,y)
        if i % evaluateEvery == 0:# To print the loss after evey 'evaluateEvery' number of training iteration
            print('Iteration',i,'('+str(round(time.time() - currTime,1))+'s) - Loss:',loss)
            currTime = time.time() 


    print('#########################################################')
    print('Siamese network for contrastive loss, Testing:')
    test_batch_size=512
    # model.load_weights('/content/drive/My Drive/omniglot/model.h5')
    print('---------------------------------------------------------')
    print('Model evaluation using training data')
    test_contrastive(model, ds_train,test_batch_size)
    print('---------------------------------------------------------')
    print('Model evaluation using testing data')
    test_contrastive(model, ds_test,test_batch_size)
    print('---------------------------------------------------------')
    print('Model evaluation using training and testing data')
    test_contrastive(model, ds_test_train,test_batch_size)


    print('#########################################################')
    print('Siamese network for triplet loss, Testing:')
    n_test=32
    # anchor_model.load_weights('/content/drive/My Drive/omniglot/model_anchor.h5')
    # anchor_model.summary()
    print('---------------------------------------------------------')
    print('Model evaluation using training data')
    validation_triplet(anchor_model, ds_train, n_test)
    print('---------------------------------------------------------')
    print('Model evaluation using testing data')
    validation_triplet(anchor_model, ds_test, n_test)
    print('---------------------------------------------------------')
    print('Model evaluation using training and testing data')
    validation_triplet(anchor_model, ds_test_train, n_test)