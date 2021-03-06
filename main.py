import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import sys

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))




g_lr = 1e-4
g_l2 = 1e-3
g_batch_size = 1

if len(sys.argv) == 4:
    print("Using argv %s" % (" ".join(sys.argv)))
    g_lr = float(sys.argv[1])
    g_l2 = float(sys.argv[2])
    g_batch_size = int(sys.argv[3])


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    with tf.name_scope("32xUpsampled") as scope:
        conv7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,
                                        padding='same', name="32x_1x1_conv",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(g_l2),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv7_2x  = tf.layers.conv2d_transpose(conv7_1x1, num_classes, 4,
                                        strides=2, padding='same', name="32x_conv_trans_upsample",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(g_l2),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("16xUpsampled") as scope:
        conv4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
                                        padding='same', name="16x_1x1_conv",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(g_l2),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv_merge1 = tf.add(conv4_1x1, conv7_2x, name="16x_combined_with_skip")
        conv4_2x  = tf.layers.conv2d_transpose(conv_merge1, num_classes, 4,
                                        strides=2, padding='same', name="16x_conv_trans_upsample",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(g_l2),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("8xUpsampled") as scope:
        conv3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,
                                        padding='same', name="8x_1x1_conv",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(g_l2),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv_merge2 = tf.add(conv3_1x1, conv4_2x, name="8x_combined_with_skip")
        conv3_8x  = tf.layers.conv2d_transpose(conv_merge2, num_classes, 16,
                                        strides=8, padding='same', name="8x_conv_trans_upsample",
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(g_l2),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())

    conv_image_0 = tf.slice(conv3_8x, [0,0,0,0], [-1,-1,-1,1])
    #conv_image_1 = tf.slice(conv3_8x, [0,0,0,1], [-1,-1,-1,2])
    tf.summary.image("conv3_8x_results_0", conv_image_0)
    #tf.summary.image("conv3_8x_results_1", conv_image_1)
    return conv3_8x
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    pred_label = tf.reshape(nn_last_layer, [-1, num_classes], name="predicted_label")
    true_label = tf.reshape(correct_label, [-1, num_classes], name="true_label")
    # sum(y_true_i * log(y_pred_i))

    with tf.name_scope("cross_entropy_loss"):
        entropy_val = tf.nn.softmax_cross_entropy_with_logits(labels=true_label, logits=pred_label)
        cross_entropy_loss = tf.reduce_sum(entropy_val)

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(cross_entropy_loss)

    return pred_label, training_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    #save training results for every eproch
    saver = tf.train.Saver()
    model_dir = './models_l2_norm_lr_%1.2e_l2_%1.2e_e10_batch_%d' % (g_lr, g_l2, g_batch_size)
    log_dir   = "./logs_l2_norm_lr_%1.2e_l2_%1.2e_e10_batch_%d"  % (g_lr, g_l2, g_batch_size)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    sess.run(tf.global_variables_initializer())
    tf.summary.scalar("loss", cross_entropy_loss)
    merged_summary_op = tf.summary.merge_all()

    global_iteration_idx = 0
    for i in range(epochs):
        print("Epoch %d" % i)
        ii = 0
        for batch_image,batch_label in get_batches_fn(batch_size):
            ii += 1
            global_iteration_idx += 1
            train_op_, cross_entropy_loss_,summary = sess.run(
                     [train_op, cross_entropy_loss, merged_summary_op],
                     feed_dict={
                        input_image: batch_image,
                        correct_label: batch_label,
                        learning_rate : g_lr,
                        keep_prob : 0.5
            })
            summary_writer.add_summary(summary, global_iteration_idx)
            print("Iteration %d, loss = %1.5f" % (ii, cross_entropy_loss_))

        # Save the model every eproch
        tf.train.write_graph(sess.graph_def, model_dir,
                        'eproch_%d_loss' % (i), as_text=False)
        saver.save(sess, '%s/eproch_%d_loss' % (model_dir, i))


#tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    #tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        epochs = 10
        batch_size = g_batch_size
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = \
            load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        pred_label, training_op, cross_entropy_loss = \
            optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, training_op, cross_entropy_loss,
                 input_image, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, pred_label, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
