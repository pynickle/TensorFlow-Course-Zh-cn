


~~~~~~~~








------------------------------------




介绍
------------













-------









-------------------




$y\\in\\{0,1\\}$ in which we use a different prediction process as



problem, obviously if the $P(x\\in\\{target\\\_class\\})$ = M, then
$P(x\\in\\{non\\\_target\\\_class\\}) = 1 - M$. So the hypothesis can be


$$P(y=1\|x)=h\_{W}(x)={{1}\\over{1+exp(-W^{T}x)}}=Sigmoid(W^{T}x) \\ \\
\\ (1)$$ $$P(y=0\|x)=1 - P(y=1\|x) = 1 - h\_{W}(x) \\ \\ \\ (2)$$











\\sum\_{i}{y^{(i)}log{1\\over{h\_{W}(x^{i})}}+(1-y^{(i)})log{1\\over{1-h\_{W}(x^{i})}}}$$










----------------------------------












---------------------





~~~~~~~~~~~~~~~



.. code:: python


    mnist = input_data.read_data_sets("MNIST_data/", reshape=True, one_hot=False)

    ########################

    ########################

    data={}

    data['train/image'] = mnist.train.images
    data['train/label'] = mnist.train.labels
    data['test/image'] = mnist.test.images
    data['test/label'] = mnist.test.labels


    index_list_train = []
    for sample_index in range(data['train/label'].shape[0]):
        label = data['train/label'][sample_index]
        if label == 1 or label == 0:
            index_list_train.append(sample_index)


    data['train/image'] = mnist.train.images[index_list_train]
    data['train/label'] = mnist.train.labels[index_list_train]



    index_list_test = []
    for sample_index in range(data['test/label'].shape[0]):
        label = data['test/label'][sample_index]
        if label == 1 or label == 0:
            index_list_test.append(sample_index)


    data['test/image'] = mnist.test.images[index_list_test]
    data['test/label'] = mnist.test.labels[index_list_test]






~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






.. code:: python

        ###############################################

        ###############################################
        image_place = tf.placeholder(tf.float32, shape=([None, num_features]), name='image')
        label_place = tf.placeholder(tf.int32, shape=([None,]), name='gt')
        label_one_hot = tf.one_hot(label_place, depth=FLAGS.num_classes, axis=-1)
        dropout_param = tf.placeholder(tf.float32)

        ##################################################

        ##################################################

        logits = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs = FLAGS.num_classes, scope='fc')







.. code:: python


        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot))


        with tf.name_scope('accuracy'):

            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))


            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

The tf.nn.softmax\_cross\_entropy\_with\_logits function does the work.



tf.nn.softmax\_cross\_entropy\_with\_logits function, for each class



-------






