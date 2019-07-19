==============================================

==============================================




------------
介绍
------------














--------------

--------------

















.. code:: python

       from tensorflow.examples.tutorials.mnist import input_data
       import tensorflow as tf
       mnist = input_data.read_data_sets("MNIST_data/", reshape=False, one_hot=False)
       data = input.provide_data(mnist)


The above code download and extract MNIST data in the MNIST\_data/



CNN classifier which takes images as input. If the one\_hot flag is set
to **True** it returns class labels as a one\_hot label. However, we set
the one\_hot flag to **False** for customized preprocessing and data
organization. The **input.provide\_data** function is provided to get








[number\_of\_training\_sample,28,28,1]. It is recommended to play around




--------------------

--------------------















   :align: center



   










.. code:: python

    import tensorflow as tf
    slim = tf.contrib.slim

    def net_architecture(images, num_classes=10, is_training=False,
                         dropout_keep_prob=0.5,
                         spatial_squeeze=True,
                         scope='Net'):


        end_points = {}

        with tf.variable_scope(scope, 'Net', [images, num_classes]) as sc:
            end_points_collection = sc.name + '_end_points'


            with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.max_pool2d], 
            outputs_collections=end_points_collection):
            

                net = tf.contrib.layers.conv2d(images, 32, [5,5], scope='conv1')
                net = tf.contrib.layers.max_pool2d(net, [2, 2], 2, scope='pool1')


                net = tf.contrib.layers.conv2d(net, 64, [5, 5], scope='conv2')
                net = tf.contrib.layers.max_pool2d(net, [2, 2], 2, scope='pool2')


                net = tf.contrib.layers.conv2d(net, 1024, [7, 7], padding='VALID', scope='fc3')
                net = tf.contrib.layers.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout3')


                logits = tf.contrib.layers.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='fc4')


                end_points = slim.utils.convert_collection_to_dict(end_points_collection)


                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='fc4/squeezed')
                    end_points[sc.name + '/fc4'] = logits
                return logits, end_points
 
    def net_arg_scope(weight_decay=0.0005):


        with tf.contrib.framework.arg_scope(
                [tf.contrib.layers.conv2d],
                padding='SAME',
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG',
                                                                                   uniform=False, seed=None,
                                                                                   dtype=tf.float32),
                activation_fn=tf.nn.relu) as sc:
            return sc

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function net\_arg\_scope is defined to share some attributes between





arg\_scope. So for this specific case the argument

layers default parameters (which are set by the arg\_scope) are as
defined in the arg\_scope. The is more work to use this useful
arg\_scope operation and it will be explained in the general TensorFlow

that all the parameters defined by arg\_scope, can be overwritten


default been set to **'SAME'** by the arg\_scope operation. Now it's the










layer\ `[reference] <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/variance_scaling_initializer>`__.









~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



net\_architecture panel in the above python script. It is worth noting























   :scale: 30 %
   :align: center
       
























~~~~~~~~~~~~~

~~~~~~~~~~~~~





network architecture and presented promising results. The dropout\_keep\_prob argument determines

disabled by the dropout layer. Moreover, the flag is\_training is



~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~


 [batch\_size, width, height, channel]. As a result, the embedding layer

So the dimension of [batch\_size, width, height, channel] becomes
[batch\_size, width x height x channel]. This


this layer has the dimensionality of [batch\_size, 1, 1, num\_classes].

is [batch\_size, num\_classes]. It is worth noting that the scope of the


--------------------

--------------------























   :scale: 30 %
   :align: center








.. code:: python
     
    graph = tf.Graph()
    with graph.as_default():


        global_step = tf.Variable(0, name="global_step", trainable=False)


        decay_steps = int(num_train_samples / FLAGS.batch_size *
                          FLAGS.num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   global_step,
                                                   decay_steps,
                                                   FLAGS.learning_rate_decay_factor,
                                                   staircase=True,
                                                   name='exponential_decay_learning_rate')



        image_place = tf.placeholder(tf.float32, shape=([None, height, width, num_channels]), name='image')
        label_place = tf.placeholder(tf.float32, shape=([None, FLAGS.num_classes]), name='gt')
        dropout_param = tf.placeholder(tf.float32)

     

        arg_scope = net.net_arg_scope(weight_decay=0.0005)
        with tf.contrib.framework.arg_scope(arg_scope):
            logits, end_points = net.net_architecture(image_place, num_classes=FLAGS.num_classes, dropout_keep_prob=dropout_param,
                                           is_training=FLAGS.is_training)


        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_place))


        with tf.name_scope('accuracy'):

            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_place, 1))


            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

     

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


        with tf.name_scope('train'):
            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

     
        arr = np.random.randint(data.train.images.shape[0], size=(3,))
        tf.summary.image('images', data.train.images[arr], max_outputs=3,
                         collections=['per_epoch_train'])


        for end_point in end_points:
            x = end_points[end_point]
            tf.summary.scalar('sparsity/' + end_point,
                              tf.nn.zero_fraction(x), collections=['train', 'test'])
            tf.summary.histogram('activations/' + end_point, x, collections=['per_epoch_train'])


        tf.summary.scalar("loss", loss, collections=['train', 'test'])
        tf.summary.scalar("accuracy", accuracy, collections=['train', 'test'])
        tf.summary.scalar("global_step", global_step, collections=['train'])
        tf.summary.scalar("learning_rate", learning_rate, collections=['train'])


        summary_train_op = tf.summary.merge_all('train')
        summary_test_op = tf.summary.merge_all('test')
        summary_epoch_train_op = tf.summary.merge_all('per_epoch_train')





~~~~~~~~~~~~~

~~~~~~~~~~~~~






~~~~~~~~~~

~~~~~~~~~~


global\_step is one of which. The reason behind
defining the global\_step is to have a track of where we are in the


The decay\_steps determines after how many steps

As can be seen **num\_epochs\_per\_decay** defines the decay factor
which is restricted to the number of passed epochs. The learning\_rate


idea of the *tf.train.exponential\_decay* layer. It is worth noting that
the *tf.train.exponential\_decay* layer takes *global\_step* as its


~~~~~~~~~~~~~

~~~~~~~~~~~~~







dimension is the *batch\_size* and is flexible.

The dropout\_param placeholder takes the probability of keeping a





~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~


**arg\_scope** operator. The
*tf.nn.softmax\_cross\_entropy\_with\_logits* is operating on the un-normalized



~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~





be added as the *train operations* to the graph. Basically 'train\_op'

execution of 'train\_op' is a training step. By passing 'global\_step'
to the optimizer, each time that the 'train\_op' is run, TensorFlow
update the 'global\_step' and increment it by one!

~~~~~~~~~

~~~~~~~~~


















needed in testing. We have a collection named 'per\_epoch\_train' too




--------

--------





~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




.. code:: python

     
    tensors_key = ['cost', 'accuracy', 'train_op', 'global_step', 'image_place', 'label_place', 'dropout_param',
                       'summary_train_op', 'summary_test_op', 'summary_epoch_train_op']
    tensors = [loss, accuracy, train_op, global_step, image_place, label_place, dropout_param, summary_train_op,
                   summary_test_op, summary_epoch_train_op]
    tensors_dictionary = dict(zip(tensors_key, tensors))


    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(graph=graph, config=session_conf)



dictionary to be used later by the corresponding keys. The allow\_soft\_placement



the TensorFlow. In this case, if the *allow\_soft\_placement* operator is



log\_device\_placement flag is to present which operations are set on





.. code:: python

     
    with sess.as_default():


        saver = tf.train.Saver(max_to_keep=FLAGS.max_num_checkpoint)


        sess.run(tf.global_variables_initializer())

        ###################################################

        ###################################################
        train_evaluation.train(sess, saver, tensors_dictionary, data,
                                 train_dir=FLAGS.train_dir,
                                 finetuning=FLAGS.fine_tuning,
                                 num_epochs=FLAGS.num_epochs, checkpoint_dir=FLAGS.checkpoint_dir,
                                 batch_size=FLAGS.batch_size)
                                     
        train_evaluation.evaluation(sess, saver, tensors_dictionary, data,
                               checkpoint_dir=FLAGS.checkpoint_dir)



operation to save and load the models. The **max\_to\_keep** flags



necessary. Finally, train\_evaluation function is


~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~



.. code:: python

     
    from __future__ import print_function
    import tensorflow as tf

    import progress_bar
    import os
    import sys

    def train(sess, saver, tensors, data, train_dir, finetuning,
                    num_epochs, checkpoint_dir, batch_size):
        """
        This function run the session whether in training or evaluation mode.
        :param sess: The default session.
        :param saver: The saver operator to save and load the model weights.
        :param tensors: The tensors dictionary defined by the graph.
        :param data: The data structure.
        :param train_dir: The training dir which is a reference for saving the logs and model checkpoints.
        :param finetuning: If fine tuning should be done or random initialization is needed.
        :param num_epochs: Number of epochs for training.
        :param checkpoint_dir: The directory of the checkpoints.
        :param batch_size: The training batch size.

        :return:
                 Run the session.
        """


        checkpoint_prefix = 'model'

        ###################################################################

        ###################################################################

        train_summary_dir = os.path.join(train_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir)
        train_summary_writer.add_graph(sess.graph)

        test_summary_dir = os.path.join(train_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir)
        test_summary_writer.add_graph(sess.graph)


        if finetuning:
            saver.restore(sess, os.path.join(checkpoint_dir, checkpoint_prefix))
            print("Model restored for fine-tuning...")

        ###################################################################

        ###################################################################
        for epoch in range(num_epochs):
            total_batch_training = int(data.train.images.shape[0] / batch_size)


            for batch_num in range(total_batch_training):
                #################################################

                #################################################

                start_idx = batch_num * batch_size
                end_idx = (batch_num + 1) * batch_size


                train_batch_data, train_batch_label = data.train.images[start_idx:end_idx], data.train.labels[
                                                                                            start_idx:end_idx]

                ########################################

                ########################################



                batch_loss, _, train_summaries, training_step = sess.run(
                    [tensors['cost'], tensors['train_op'], tensors['summary_train_op'],
                     tensors['global_step']],
                    feed_dict={tensors['image_place']: train_batch_data,
                               tensors['label_place']: train_batch_label,
                               tensors['dropout_param']: 0.5})

                ########################################

                ########################################


                train_summary_writer.add_summary(train_summaries, global_step=training_step)




                #################################################

                #################################################

                progress = float(batch_num + 1) / total_batch_training
                progress_bar.print_progress(progress, epoch_num=epoch + 1, loss=batch_loss)




            train_epoch_summaries = sess.run(tensors['summary_epoch_train_op'],
                                             feed_dict={tensors['image_place']: train_batch_data,
                                                        tensors['label_place']: train_batch_label,
                                                        tensors['dropout_param']: 0.5})


            train_summary_writer.add_summary(train_epoch_summaries, global_step=training_step)

            #####################################################

            #####################################################




            test_accuracy_epoch, test_summaries = sess.run([tensors['accuracy'], tensors['summary_test_op']],
                                                           feed_dict={tensors['image_place']: data.test.images,
                                                                      tensors[
                                                                          'label_place']: data.test.labels,
                                                                      tensors[
                                                                          'dropout_param']: 1.})
            print("Epoch " + str(epoch + 1) + ", Testing Accuracy= " + \
                  "{:.5f}".format(test_accuracy_epoch))

            ###########################################################

            ###########################################################


            current_step = tf.train.global_step(sess, tensors['global_step'])


            test_summary_writer.add_summary(test_summaries, global_step=current_step)

        ###########################################################

        ###########################################################




        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)


        save_path = saver.save(sess, os.path.join(checkpoint_dir, checkpoint_prefix))
        print("Model saved in file: %s" % save_path)


        ############################################################################

        ############################################################################
    def evaluation(sess, saver, tensors, data, checkpoint_dir):


            checkpoint_prefix = 'model'


            saver.restore(sess, os.path.join(checkpoint_dir, checkpoint_prefix))
            print("Model restored...")


            test_accuracy = 100 * sess.run(tensors['accuracy'], feed_dict={tensors['image_place']: data.test.images,
                                                                           tensors[
                                                                               'label_place']: data.test.labels,
                                                                           tensors[
                                                                               'dropout_param']: 1.})

            print("Final Test Accuracy is %% %.2f" % test_accuracy)











~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~








   :align: center
   








   :align: center
   










   :align: center
   







   :align: center










   :align: center
   


















-------

-------














