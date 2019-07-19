

~~~~~~~~








----------------------------------








介绍
------------












----------------------------------











---------------------




.. code:: python




    DATA_FILE = "data/fire_theft.xls"


    book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
    sheet = book.sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    num_samples = sheet.nrows - 1

    #######################

    #######################
    tf.app.flags.DEFINE_integer(
        'num_epochs', 50, 'The number of epochs for training the model. Default=50')

    FLAGS = tf.app.flags.FLAGS



.. code:: python



    W = tf.Variable(0.0, name="weights")
    b = tf.Variable(0.0, name="bias")




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


        Y_predicted = inference(X)
        return tf.squared_difference(Y, Y_predicted)

.. code:: python


    def train(loss):
        learning_rate = 0.0001
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)




.. code:: python

    with tf.Session() as sess:


        sess.run(tf.global_variables_initializer())


        X, Y = inputs()


        train_loss = loss(X, Y)
        train_op = train(train_loss)


        for epoch_num in range(FLAGS.num_epochs): # run 100 epochs
            for x, y in data:
              train_op = train(train_loss)


              loss_value,_ = sess.run([train_loss,train_op], feed_dict={X: x, Y: y})


            print('epoch %d, loss=%f' %(epoch_num+1, loss_value))


            wcoeff, bias = sess.run([W, b])

In the above code, the sess.run(tf.global\_variables\_initializer())
initialize all the defined variables globally. The train\_op is built
upon the train\_loss and will be updated in each step. In the end, the




.. code:: python

    ###############################

    ###############################
    Input_values = data[:,0]
    Labels = data[:,1]
    Prediction_values = data[:,0] * wcoeff + bias
    plt.plot(Input_values, Labels, 'ro', label='main')
    plt.plot(Input_values, Prediction_values, label='Predicted')


    plt.legend()
    plt.savefig('plot.png')
    plt.close()





   :align: center










-------






