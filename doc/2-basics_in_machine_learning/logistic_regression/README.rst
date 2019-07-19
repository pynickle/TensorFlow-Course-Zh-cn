

说
~~~~~~~~

：
“
您
可
以

提
------------------------------------

供
了

介绍
------------

一
套
名
为
T
e




o
r

B
-------

o
a
r
d
的
可

视
-------------------

化
工
具
。
在
本


problem, obviously if the $P(x\\in\\{target\\\_class\\})$ = M, then
$P(x\\in\\{non\\\_target\\\_class\\}) = 1 - M$. So the hypothesis can be
包

$$P(y=1\|x)=h\_{W}(x)={{1}\\over{1+exp(-W^{T}x)}}=Sigmoid(W^{T}x) \\ \\
\\ (1)$$ $$P(y=0\|x)=1 - P(y=1\|x) = 1 - h\_{W}(x) \\ \\ \\ (2)$$

一
个


单
的
T
e
n

s
\\sum\_{i}{y^{(i)}log{1\\over{h\_{W}(x^{i})}}+(1-y^{(i)})log{1\\over{1-h\_{W}(x^{i})}}}$$

r
b
o

a
r
d

行
----------------------------------



安
装
及
启
用
。
*
*

的
---------------------

注
意

：
~~~~~~~~~~~~~~~




.. code:: python

e
    mnist = input_data.read_data_sets("MNIST_data/", reshape=True, one_hot=False)

    ########################
*
    ########################
*
    data={}

    data['train/image'] = mnist.train.images
    data['train/label'] = mnist.train.labels
    data['test/image'] = mnist.test.images
    data['test/label'] = mnist.test.labels

*
    index_list_train = []
    for sample_index in range(data['train/label'].shape[0]):
        label = data['train/label'][sample_index]
        if label == 1 or label == 0:
            index_list_train.append(sample_index)



    data['train/image'] = mnist.train.images[index_list_train]
    data['train/label'] = mnist.train.labels[index_list_train]


摘
    index_list_test = []
    for sample_index in range(data['test/label'].shape[0]):
        label = data['test/label'][sample_index]
        if label == 1 or label == 0:
            index_list_test.append(sample_index)

要
    data['test/image'] = mnist.test.images[index_list_test]
    data['test/label'] = mnist.test.labels[index_list_test]

:
T
e

n
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

s
o
r
n

.. code:: python

        ###############################################
b
        ###############################################
        image_place = tf.placeholder(tf.float32, shape=([None, num_features]), name='image')
        label_place = tf.placeholder(tf.int32, shape=([None,]), name='gt')
        label_one_hot = tf.one_hot(label_place, depth=FLAGS.num_classes, axis=-1)
        dropout_param = tf.placeholder(tf.float32)

        ##################################################
o
        ##################################################
a
        logits = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs = FLAGS.num_classes, scope='fc')

r
d
及
其
优

.. code:: python

点
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot))

的
        with tf.name_scope('accuracy'):
介
            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))

绍
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



出
了




程

使
-------

的
范
围
，
这
