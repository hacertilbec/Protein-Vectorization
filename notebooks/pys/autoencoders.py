import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from functools import partial
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from functools import partial
import sys
import os


class LinearAutoEncoder:
    def __init__(self,n_input, n_hidden, n_iteration=100, learning_rate = 0.001,model_path = "models/autoencoder.ckpt"):
        tf.reset_default_graph()
        tf.set_random_seed(42)
        self.n_inputs = n_input
        self.n_hidden = n_hidden
        self.n_outputs = n_input
        self.n_iteration = n_iteration
        self.learning_rate = learning_rate
        self.n_iterations = n_iteration
        
        self.model_path = model_path
        
    def train(self,X_train):
        X = tf.placeholder(tf.float32, shape=[None, self.n_inputs])
        hidden = fully_connected(X, self.n_hidden, activation_fn=tf.nn.relu)
        outputs = fully_connected(hidden, self.n_outputs, activation_fn=tf.nn.relu)

        reconstruction_loss = tf.reduce_sum(tf.square(outputs - X)) # MSE

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        training_op = optimizer.minimize(reconstruction_loss)

        codings = hidden # the output of the hidden layer provides the codings
    
        init = tf.global_variables_initializer()
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            init.run()
            if os.path.isfile(self.model_path+".index"):
                saver.restore(sess, self.model_path)
            loss = []
            for iteration in range(self.n_iterations):
                iteration_loss = []
                for protein in X_train:
                    _, loss_val = sess.run([training_op, reconstruction_loss], feed_dict={X: [protein]})
                    iteration_loss.append(loss_val)
                loss.append(sum(iteration_loss)/float(len(X_train)))

            # Save model parameters
            save_path = saver.save(sess, self.model_path)
            print("Model saved in path: %s" % save_path)
        return loss
    
    def encode(self,samples):
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, shape=[None, self.n_inputs])
        hidden = fully_connected(X, self.n_hidden, activation_fn=None)
        codings = hidden # the output of the hidden layer provides the codings

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Later, launch the model, use the saver to restore variables from disk, and
        # do some work with the model.
        with tf.Session() as sess:
            # Restore variables from disk.
            saver.restore(sess, self.model_path)
            print("Model restored.")
            # Check the values of the variables
            encoding = codings.eval(feed_dict={X: samples},session=sess)
        return encoding
    
    
def LinearAutoencoder(X_train, n_input, n_hidden, n_iteration, learning_rate = 0.001):
    n_inputs = n_input # input is flatten version of input matrix
    n_hidden = n_hidden
    n_outputs = n_inputs

    learning_rate = learning_rate

    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.relu)
    outputs = fully_connected(hidden, n_outputs, activation_fn=tf.nn.relu)

    reconstruction_loss = tf.reduce_sum(tf.square(outputs - X)) # MSE

    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(reconstruction_loss)

    init = tf.global_variables_initializer()

    n_iterations = n_iteration # Number of iterations
    codings = hidden # the output of the hidden layer provides the codings

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()
        loss = []
        for iteration in range(n_iterations):
            _, loss_val = sess.run([training_op, reconstruction_loss], feed_dict={X: X_train}) # no labels (unsupervised)
            loss.append(loss_val)

        # Save model parameters
        save_path = saver.save(sess, "models/autoencoder.ckpt")
        print("Model saved in path: %s" % save_path)
        
        # Encode training samples
        codings_val = codings.eval(feed_dict={X: X_train})
    return codings_val, loss


def StackedAutoencoderWithTiedWeights(X_train, n_input, n_hidden1, n_hidden2, n_epochs, learning_rate = 0.001):
    n_inputs = n_input # for pair distance matrix
    n_hidden1 = n_hidden1
    n_hidden2 = n_hidden2 # codings
    n_hidden3 = n_hidden1
    n_outputs = n_inputs

    learning_rate = learning_rate
    l2_reg = 0.001

    activation = tf.nn.elu
    regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
    initializer = tf.contrib.layers.variance_scaling_initializer()

    X = tf.placeholder(tf.float32, shape=[None, n_inputs])

    weights1_init = initializer([n_inputs, n_hidden1])
    weights2_init = initializer([n_hidden1, n_hidden2])

    weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
    weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
    weights3 = tf.transpose(weights2, name="weights3") # tied weights
    weights4 = tf.transpose(weights1, name="weights4") # tied weights

    biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
    biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
    biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
    biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")

    hidden1 = activation(tf.matmul(X, weights1) + biases1)
    hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
    hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
    outputs = tf.matmul(hidden3, weights4) + biases4

    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
    reg_loss = regularizer(weights1) + regularizer(weights2)

    loss = reconstruction_loss + reg_loss

    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    n_epochs = n_epochs

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()
        loss_control = []
        for epoch in range(n_epochs):
            print("Iteration ",epoch)
            iteration_loss = []
            for protein in X_train:
                _, loss_val = sess.run([training_op, loss], feed_dict={X: [protein]})
                iteration_loss.append(loss_val)
            loss_control.append(sum(iteration_loss)/float(len(X_train)))
        # Test on the same protein
        codings_val = hidden2.eval(feed_dict={X: X_train})

    return codings_val, loss_control


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def train_autoencoder(X_train, n_neurons, n_epochs,
                      learning_rate = 0.01, l2_reg = 0.0005, seed=42,
                      hidden_activation=tf.nn.elu,
                      output_activation=tf.nn.elu):
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(seed)

        n_inputs = X_train.shape[1]

        X = tf.placeholder(tf.float32, shape=[None, n_inputs])

        my_dense_layer = partial(
            tf.layers.dense,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

        hidden = my_dense_layer(X, n_neurons, activation=hidden_activation, name="hidden")
        outputs = my_dense_layer(hidden, n_inputs, activation=output_activation, name="outputs")

        reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([reconstruction_loss] + reg_losses)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        init.run()
        losses = []
        for epoch in range(n_epochs):
            for protein in X_train:
                #print("\r{}%".format(100 * iteration // n_batches), end="")
                sys.stdout.flush()
                sess.run(training_op, feed_dict={X: [protein]})
            loss_train = reconstruction_loss.eval(feed_dict={X: [protein]})
            losses.append(loss_train)
        params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        hidden_val = hidden.eval(feed_dict={X: X_train})
        return hidden_val, params["hidden/kernel:0"], params["hidden/bias:0"], params["outputs/kernel:0"], params["outputs/bias:0"], losses
    
   
