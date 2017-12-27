import numpy as np
from PIL import Image
import tensorflow as tf
import glob
def get_image_data(filename):
    imgData = (Image.open(filename))
    imgData = imgData.resize((60,60),Image.ANTIALIAS)
    #imgData = map(list, imgData)
    imgData = np.asarray(imgData)
    img_wide = imgData.reshape(1, 60 , 60 , 3)
    return img_wide
def extract_files(filesPath):
    data = np.empty((1,60,60,3))
    img = glob.glob(filesPath+"/*.jpg")
    for i in img:
        print(i)
        arr = get_image_data(i)
        data = np.append(data,arr,axis=0)
    return data
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.05, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
def CNN_EYEGLASSES(ImageX,ImageY):

    session = tf.InteractiveSession()
    #Initializing placeholders for x and y value
    #None --> First dimension can be of any size
    
    x = tf.placeholder(tf.float32, shape = [None, 10800]) 
    
    y = tf.placeholder(tf.float32, shape = [None,None])
    
    #First convolution layer
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x,[-1, 60, 60, 3])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #Second convolution layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    W_fc1 = weight_variable([15 * 15 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 15*15*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #To reduce overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #Adding a layer
    W_fc2 = weight_variable([1024, 1])
    b_fc2 = bias_variable([1])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-9).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.round(y_conv), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for i in range(0,60000,1000):
            #100 is the step size for training
            batch=ImageX[range(i,i+100),:]
            batchY=ImageY[range(i,i+100),:]
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y: batch[1], keep_prob: 0.8})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob:
            0.8})
        print('Celebs Test accuracy %g' % accuracy.eval(feed_dict={
            x: ImageX, y: ImageY, keep_prob: 1.0}))
//load data from dataset
#Load images from the filesPath
ImageX=extract_files(filesPath)
#reshape loaded images of size [N,60,60,3] to [N,10800]
ImageX_R=ImageX.reshape(N,10800)
#load list attributes: filesPathList is a valid file path
allLabelsArray=np.loadtxt(filesPathList,dtype='str')
#Eyeglasses column is at 16th column
ImageY=allLabelsArray[:,16]
ImageY = np.asarray([int(numeric_string) for numeric_string in arrayCopy]).reshape(202599,1)
#Negative values(-1) are set to 0 and positive values are set to 1
ImageY[ImageY<0]=0
#ImageY=desired_array[range(0,60000),:] Pick 60k image sinitially for testing
CNN_EYEGLASSES(ImageX_R,ImageY)