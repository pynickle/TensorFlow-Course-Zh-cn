
说
~~~~~~~~

：
“
您
可
以

使
----------------------------------

用
T

e



介绍
------------

n
s
o
r
F
l
o
w
进

行
----------------------------------

像
训
练
一
个
大
规
模

的
---------------------

深
度

.. code:: python


神
经
    DATA_FILE = "data/fire_theft.xls"

网
    book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
    sheet = book.sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    num_samples = sheet.nrows - 1

    #######################
络
    #######################
    tf.app.flags.DEFINE_integer(
        'num_epochs', 50, 'The number of epochs for training the model. Default=50')
一
    FLAGS = tf.app.flags.FLAGS

样

.. code:: python

复
杂
    W = tf.Variable(0.0, name="weights")
    b = tf.Variable(0.0, name="bias")

而
混

.. code:: python

    def inputs():
        """
        Defining the place_holders.
        :return:
                Returning the data and label lace holders.
        """
        X = tf.placeholder(tf.float32, name="X")
        Y = tf.placeholder(tf.float32, name="Y")
        return X,Y

.. code:: python

    def inference():
        """
        Forward passing the X.
        :param X: Input.
        :return: X*W + b.
        """
        return X * W + b

.. code:: python

    def loss(X, Y):
        """
        compute the loss by comparing the predicted value to the actual label.
        :param X: The input.
        :param Y: The label.
        :return: The loss over the samples.
        """

乱
        Y_predicted = inference(X)
        return tf.squared_difference(Y, Y_predicted)

.. code:: python

的
    def train(loss):
        learning_rate = 0.0001
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

计
算

.. code:: python

with tf.Session() as sess:

。
        sess.run(tf.global_variables_initializer())

为
        X, Y = inputs()

了
        train_loss = loss(X, Y)
        train_op = train(train_loss)

更
        for epoch_num in range(FLAGS.num_epochs): # run 100 epochs
            for x, y in data:
              train_op = train(train_loss)

容
              loss_value,_ = sess.run([train_loss,train_op], feed_dict={X: x, Y: y})

易
            print('epoch %d, loss=%f' %(epoch_num+1, loss_value))

理
            wcoeff, bias = sess.run([W, b])







试
和
优

.. code:: python

    ###############################
化
    ###############################
    Input_values = data[:,0]
    Labels = data[:,1]
    Prediction_values = data[:,0] * wcoeff + bias
    plt.plot(Input_values, Labels, 'ro', label='main')
    plt.plot(Input_values, Prediction_values, label='Predicted')

T
    plt.legend()
    plt.savefig('plot.png')
    plt.close()

e

n
s
:align: center

o
r

F
l
o
w

使
-------

程
序
，
我
们
