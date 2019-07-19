==============================================
以
==============================================

将
日

------------
介绍
------------


志
文
件
转
换
为
可
视
的
数
据

--------------
，
--------------

o
以
便
用
户
能
够
评
估
架
构
和
操
作
。

.. code:: python

       from tensorflow.examples.tutorials.mnist import input_data
       import tensorflow as tf
       mnist = input_data.read_data_sets("MNIST_data/", reshape=False, one_hot=False)
       data = input.provide_data(mnist)




储
这
些








的


`

`
路
径
`




定
义

--------------------
如
--------------------

下
：
#


保
存
日
志
文
件
的

路
s
:align: center

径

   
与
此
p
y
t
h
o
n
文

.. code:: python

import tensorflow as tf
    slim = tf.contrib.slim

    def net_architecture(images, num_classes=10, is_training=False,
                         dropout_keep_prob=0.5,
                         spatial_squeeze=True,
                         scope='Net'):

件
        end_points = {}

        with tf.variable_scope(scope, 'Net', [images, num_classes]) as sc:
            end_points_collection = sc.name + '_end_points'

所
            with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.max_pool2d], 
            outputs_collections=end_points_collection):
            
在
                net = tf.contrib.layers.conv2d(images, 32, [5,5], scope='conv1')
                net = tf.contrib.layers.max_pool2d(net, [2, 2], 2, scope='pool1')

的
                net = tf.contrib.layers.conv2d(net, 64, [5, 5], scope='conv2')
                net = tf.contrib.layers.max_pool2d(net, [2, 2], 2, scope='pool2')

文
                net = tf.contrib.layers.conv2d(net, 1024, [7, 7], padding='VALID', scope='fc3')
                net = tf.contrib.layers.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout3')

件
                logits = tf.contrib.layers.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='fc4')

夹
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

相
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='fc4/squeezed')
                    end_points[sc.name + '/fc4'] = logits
                return logits, end_points
 
    def net_arg_scope(weight_decay=0.0005):
同

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
#
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



将
所
有
元
素


储






A


结
构


`
`

o
s
.
p
a
t
h


d
i
r
n
a
m
n
e

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

o
s


p
a
t
h
.

a
b
s
p
a
t

h
(
_
_
f
i
l
e

_
:scale: 30 %
:align: center
       
_


)
)
`
`


用
来

获
取
当
前
p
y
t
h
o
n
文
件

~~~~~~~~~~~~~
的
~~~~~~~~~~~~~

目
录
名
。


`


f
.

~~~~~~~~~~~~~~~
a
~~~~~~~~~~~~~~~

p


.




a
g


.


L

--------------------
A
--------------------

G
S
`
`


指

向
所
有
使
用



`
`
F
L
A
G

S
:scale: 30 %
:align: center

`



`



.. code:: python
     
    graph = tf.Graph()
    with graph.as_default():

指
        global_step = tf.Variable(0, name="global_step", trainable=False)

示
        decay_steps = int(num_train_samples / FLAGS.batch_size *
                          FLAGS.num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   global_step,
                                                   decay_steps,
                                                   FLAGS.learning_rate_decay_factor,
                                                   staircase=True,
                                                   name='exponential_decay_learning_rate')


符
        image_place = tf.placeholder(tf.float32, shape=([None, height, width, num_channels]), name='image')
        label_place = tf.placeholder(tf.float32, shape=([None, FLAGS.num_classes]), name='gt')
        dropout_param = tf.placeholder(tf.float32)

     
定
        arg_scope = net.net_arg_scope(weight_decay=0.0005)
        with tf.contrib.framework.arg_scope(arg_scope):
            logits, end_points = net.net_architecture(image_place, num_classes=FLAGS.num_classes, dropout_keep_prob=dropout_param,
                                           is_training=FLAGS.is_training)

点
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_place))

的
        with tf.name_scope('accuracy'):
义
            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_place, 1))

绍
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

     
的
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

f
        with tf.name_scope('train'):
            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

     
        arr = np.random.randint(data.train.images.shape[0], size=(3,))
        tf.summary.image('images', data.train.images[arr], max_outputs=3,
                         collections=['per_epoch_train'])

l
        for end_point in end_points:
            x = end_points[end_point]
            tf.summary.scalar('sparsity/' + end_point,
                              tf.nn.zero_fraction(x), collections=['train', 'test'])
            tf.summary.histogram('activations/' + end_point, x, collections=['per_epoch_train'])

a
        tf.summary.scalar("loss", loss, collections=['train', 'test'])
        tf.summary.scalar("accuracy", accuracy, collections=['train', 'test'])
        tf.summary.scalar("global_step", global_step, collections=['train'])
        tf.summary.scalar("learning_rate", learning_rate, collections=['train'])

g
        summary_train_op = tf.summary.merge_all('train')
        summary_test_op = tf.summary.merge_all('test')
        summary_epoch_train_op = tf.summary.merge_all('per_epoch_train')


s
。

~~~~~~~~~~~~~
从
~~~~~~~~~~~~~

现
在
起
，

~~~~~~~~~~
你
~~~~~~~~~~

可




用




`




A
G




f

~~~~~~~~~~~~~
l
~~~~~~~~~~~~~

a
g
_
n
a
m





`


调
用

~~~~~~~~~~~~~~~~~~~~~~~~~~~~
f
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

l




s
。

~~~~~~~~~~~~~~~~
为
~~~~~~~~~~~~~~~~

方
便
起
见


仅







~~~~~~~~~


~~~~~~~~~

`
`
绝
对
路
径
`
`



是
很
有
用
的
。
通


使
用
以

--------
下
--------

代
码
脚

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

，
用

.. code:: python

     
    tensors_key = ['cost', 'accuracy', 'train_op', 'global_step', 'image_place', 'label_place', 'dropout_param',
                       'summary_train_op', 'summary_test_op', 'summary_epoch_train_op']
    tensors = [loss, accuracy, train_op, global_step, image_place, label_place, dropout_param, summary_train_op,
                   summary_test_op, summary_epoch_train_op]
    tensors_dictionary = dict(zip(tensors_key, tensors))

户
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(graph=graph, config=session_conf)


可


大
致
了


到
如
何


用
绝
对
路

.. code:: python

     
    with sess.as_default():
径
来
        saver = tf.train.Saver(max_to_keep=FLAGS.max_num_checkpoint)

表
        sess.run(tf.global_variables_initializer())

        ###################################################
示
        ###################################################
        train_evaluation.train(sess, saver, tensors_dictionary, data,
                                 train_dir=FLAGS.train_dir,
                                 finetuning=FLAGS.fine_tuning,
                                 num_epochs=FLAGS.num_epochs, checkpoint_dir=FLAGS.checkpoint_dir,
                                 batch_size=FLAGS.batch_size)
                                     
        train_evaluation.evaluation(sess, saver, tensors_dictionary, data,
                               checkpoint_dir=FLAGS.checkpoint_dir)






`
l
o


_

~~~~~~~~~~~~~~~~~~~
d
~~~~~~~~~~~~~~~~~~~

i

.. code:: python

     
    from __future__ import print_function
import tensorflow as tf
如
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

r
        checkpoint_prefix = 'model'

        ###################################################################
`
        ###################################################################

        train_summary_dir = os.path.join(train_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir)
        train_summary_writer.add_graph(sess.graph)

        test_summary_dir = os.path.join(train_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir)
        test_summary_writer.add_graph(sess.graph)

`
        if finetuning:
            saver.restore(sess, os.path.join(checkpoint_dir, checkpoint_prefix))
            print("Model restored for fine-tuning...")

        ###################################################################


        ###################################################################
        for epoch in range(num_epochs):
            total_batch_training = int(data.train.images.shape[0] / batch_size)

目
            for batch_num in range(total_batch_training):
                #################################################
录
                #################################################

                start_idx = batch_num * batch_size
                end_idx = (batch_num + 1) * batch_size

。
                train_batch_data, train_batch_label = data.train.images[start_idx:end_idx], data.train.labels[
                                                                                            start_idx:end_idx]

                ########################################
#
                ########################################



输
                batch_loss, _, train_summaries, training_step = sess.run(
                    [tensors['cost'], tensors['train_op'], tensors['summary_train_op'],
                     tensors['global_step']],
                    feed_dict={tensors['image_place']: train_batch_data,
                               tensors['label_place']: train_batch_label,
                               tensors['dropout_param']: 0.5})

                ########################################
入
                ########################################

绝
                train_summary_writer.add_summary(train_summaries, global_step=training_step)

对
路

                #################################################
径
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

#


o
            test_accuracy_epoch, test_summaries = sess.run([tensors['accuracy'], tensors['summary_test_op']],
                                                           feed_dict={tensors['image_place']: data.test.images,
                                                                      tensors[
                                                                          'label_place']: data.test.labels,
                                                                      tensors[
                                                                          'dropout_param']: 1.})
            print("Epoch " + str(epoch + 1) + ", Testing Accuracy= " + \
                  "{:.5f}".format(test_accuracy_epoch))

            ###########################################################
s
            ###########################################################

.
            current_step = tf.train.global_step(sess, tensors['global_step'])

p
            test_summary_writer.add_summary(test_summaries, global_step=current_step)

        ###########################################################
a
        ###########################################################

t

h
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

.
        save_path = saver.save(sess, os.path.join(checkpoint_dir, checkpoint_prefix))
        print("Model saved in file: %s" % save_path)


        ############################################################################
e
        ############################################################################
    def evaluation(sess, saver, tensors, data, checkpoint_dir):

x
            checkpoint_prefix = 'model'

p
            saver.restore(sess, os.path.join(checkpoint_dir, checkpoint_prefix))
            print("Model restored...")

a
            test_accuracy = 100 * sess.run(tensors['accuracy'], feed_dict={tensors['image_place']: data.test.images,
                                                                           tensors[
                                                                               'label_place']: data.test.labels,
                                                                           tensors[
                                                                               'dropout_param']: 1.})

            print("Final Test Accuracy is %% %.2f" % test_accuracy)


n
d
u
s
e
r
用
于

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
将
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



'
~
'



s
:align: center
   
符


号
转

换
s
:align: center
   
为


相
应
的
路

径
s
:align: center
   
指


示

符
s
:align: center

。





文



s
:align: center
   








#














示
例
：

-------
使
-------



'
~
/
l
o
g
s
'


等
于

