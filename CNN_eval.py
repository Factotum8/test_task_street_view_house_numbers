import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from six.moves import cPickle as pickle


def gaussian_filter_(kernel_shape, ax):
    x = np.zeros(kernel_shape, dtype = float)
    mid = np.floor(kernel_shape[0] / 2.)

    for kernel_idx in range(0, kernel_shape[2]):
        for i in range(0, kernel_shape[0]):
            for j in range(0, kernel_shape[1]):
                x[i, j, kernel_idx, 0] = gauss(i - mid, j - mid)

    return tf.convert_to_tensor(x / np.sum(x), dtype=tf.float32)


def gaussian_filter(kernel_shape, ax):
    # The Gaussian filter of the desired size initialized to zero
    filter_ = np.zeros(kernel_shape, dtype = float)
    mid = np.floor(kernel_shape[0] / 2.)  # Middle of kernel of Gaussian filter

    for kernel_idx in range(0, kernel_shape[2]):
        for i in range(0, kernel_shape[0]):  # go on width of Gaussian weighting window
            for j in range(0, kernel_shape[1]):  # go on height of Gaussian weighting window
                filter_[i, j, kernel_idx, 0] = gauss(i - mid, j - mid)

    filter_ = filter_ / np.sum(filter_)
    return tf.convert_to_tensor(filter_, dtype=tf.float32), filter_


def gauss(x, y, sigma=3.0):
    Z = 2 * np.pi * sigma ** 2
    return  1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))


def LecunLCN(X, image_shape, threshold=1e-4, radius=7, use_divisor=True):
    """
    Local Contrast Normalization
    :param X: tf_train_dataset
    :param image_shape: [batch_size, image_size, image_size, num_channels]
    """
    # Get Gaussian filter
    filter_shape = (radius, radius, image_shape[3], 1)
    # For process visualisation
#     plt.rcParams['figure.figsize'] = (45.0, 100.0)    
#     f, ax = plt.subplots(nrows= radius * radius + 1, ncols=1)
#     [i.axis('off') for i in ax]
    ax = None

    filters, filters_asarray = gaussian_filter(filter_shape, ax)

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    # Compute the Guassian weighted average by means of convolution
    convout = tf.nn.conv2d(X, filters, [1,1,1,1], 'SAME')

    # Subtractive step
    mid = int(np.floor(filter_shape[1] / 2.))

    # Make filter dimension broadcastable and subtract
    centered_X = tf.subtract(X, convout)

    # Boolean marks whether or not to perform divisive step
    if use_divisor:
        # Note that the local variances can be computed by using the centered_X
        # tensor. If we convolve this with the mean filter, that should give us
        # the variance at each point. We simply take the square root to get our
        # denominator

        # Compute variances
        sum_sqr_XX = tf.nn.conv2d(tf.square(centered_X), filters, [1,1,1,1], 'SAME')

        # Take square root to get local standard deviation
        denom = tf.sqrt(sum_sqr_XX)

        per_img_mean = tf.reduce_mean(denom)
        divisor = tf.maximum(per_img_mean, denom)
        # Divisise step
        new_X = tf.truediv(centered_X, tf.maximum(divisor, threshold))
    else:
        new_X = centered_X

    return new_X


def init_model(graph, test_dataset, image_index):
    image_size = 32
    num_labels = 11  # 0-9, + blank
    num_channels = 1  # grayscale

    patch_size = 5
    depth1 = 16
    depth2 = 32
    depth3 = 64
    num_hidden1 = 64

    # Input data.
    tf_test_dataset = tf.placeholder(tf.float32, shape=(1, 32, 32, 1))
  
    # Variables.
    layer1_weights = tf.get_variable("W1",
                                       shape=[patch_size, patch_size, num_channels, depth1],
                                       initializer=tf.contrib.layers.xavier_initializer_conv2d())
    layer1_biases = tf.Variable(tf.constant(1.0, shape=[depth1]), name='B1')
    layer2_weights = tf.get_variable("W2",
                                       shape=[patch_size, patch_size, depth1, depth2],
                                       initializer=tf.contrib.layers.xavier_initializer_conv2d())
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth2]), name='B2')
    layer3_weights = tf.get_variable("W3",
                                       shape=[patch_size, patch_size, depth2, num_hidden1],
                                       initializer=tf.contrib.layers.xavier_initializer_conv2d())
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden1]), name='B3')

    s1_w = tf.get_variable("WS1", shape=[num_hidden1, num_labels],
                             initializer=tf.contrib.layers.xavier_initializer())
    s1_b = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='BS1')
    s2_w = tf.get_variable("WS2", shape=[num_hidden1, num_labels],
                             initializer=tf.contrib.layers.xavier_initializer())
    s2_b = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='BS2')
    s3_w = tf.get_variable("WS3", shape=[num_hidden1, num_labels],
                             initializer=tf.contrib.layers.xavier_initializer())
    s3_b = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='BS3')
    s4_w = tf.get_variable("WS4", shape=[num_hidden1, num_labels],
                             initializer=tf.contrib.layers.xavier_initializer())
    s4_b = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='BS4')
    s5_w = tf.get_variable("WS5", shape=[num_hidden1, num_labels],
                            initializer=tf.contrib.layers.xavier_initializer())
    s5_b = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='BS5')
  
    # Model.
    def model(data, keep_prob, shape):
        LCN = LecunLCN(data, shape)
        conv = tf.nn.conv2d(LCN, layer1_weights, [1,1,1,1], 'VALID', name='C1')
        hidden = tf.nn.relu(conv + layer1_biases)
        lrn = tf.nn.local_response_normalization(hidden)
        sub = tf.nn.max_pool(lrn, [1,2,2,1], [1,2,2,1], 'SAME', name='S2')
        conv = tf.nn.conv2d(sub, layer2_weights, [1,1,1,1], padding='VALID', name='C3')
        hidden = tf.nn.relu(conv + layer2_biases)
        lrn = tf.nn.local_response_normalization(hidden)
        sub = tf.nn.max_pool(lrn, [1,2,2,1], [1,2,2,1], 'SAME', name='S4')
        conv = tf.nn.conv2d(sub, layer3_weights, [1,1,1,1], padding='VALID', name='C5')
        hidden = tf.nn.relu(conv + layer3_biases)
        hidden = tf.nn.dropout(hidden, keep_prob)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

        logits1 = tf.matmul(reshape, s1_w) + s1_b
        logits2 = tf.matmul(reshape, s2_w) + s2_b
        logits3 = tf.matmul(reshape, s3_w) + s3_b
        logits4 = tf.matmul(reshape, s4_w) + s4_b
        logits5 = tf.matmul(reshape, s5_w) + s5_b

        return [logits1, logits2, logits3, logits4, logits5]
  
    # Training computation.
    [logits1, logits2, logits3, logits4, logits5] = model(tf_test_dataset, 1, [10, 32, 32, 1])
  
    predict = tf.stack([tf.nn.softmax(logits1), tf.nn.softmax(logits2),
                        tf.nn.softmax(logits3), tf.nn.softmax(logits4),
                        tf.nn.softmax(logits5)])
  
    test_prediction = tf.transpose(tf.argmax(predict, 2))
    input_image_array = np.expand_dims(test_dataset[image_index, :, :, :], axis=0)

    return graph, test_prediction, input_image_array, tf_test_dataset


def predict(graph, test_prediction, input_image_array, tf_test_dataset):

    with tf.Session(graph=graph) as session:
        saver = tf.train.Saver()
        saver.restore(session, "./model/CNN_multi2.ckpt")
        print("Model restored.")
        print('Initialized')
        prediction_ = session.run(test_prediction, feed_dict={tf_test_dataset: input_image_array,})
        number_house = "".join([str(digit) for digit in prediction_[0, :] if digit != 10])
        return number_house


def main(filename):
    pickle_file = './model/SVHN_multi.pickle'

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        # print('Test set', test_dataset.shape, test_labels.shape)

    fullname = os.path.join('./data/test', filename)
    im = Image.open(fullname)
    house_num = ''
    image_index, _ = filename.split(".")
    image_index = int(image_index)

    graph = tf.Graph()
    with graph.as_default():
        graph, test_prediction, input_image_array, tf_test_dataset = init_model(graph, test_dataset, image_index)
        number_house = predict(graph, test_prediction, input_image_array, tf_test_dataset)
    print("number_house: {}".format(number_house))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-f", action='store', type=str,
                            dest='FileName', help="Specify file name.")
    args = arg_parser.parse_args()
    if args.FileName:
        file_name = args.FileName
        main(file_name)
    else:
        print("Specify file name")

