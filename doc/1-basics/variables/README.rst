TensorFlow变量简介：创建，初始化
--------------------------------------------------------------

本教程介绍如何定义和初始化TensorFlow变量。

介绍
------------

定义 ``变量`` 是必要的，因为它们包含参数。
如果没有参数，那么训练，更新，保存，恢复和任何
其他操作都无法执行。TensorFlow中定义的变量
只是具有特定形状和类型的张量。所以我们必须使用值来初始化这些张量
以使其有效。在本教程中，我们
将解释如何 ``定义`` 和 ``初始化`` 变量。
 `源
代码 <https://github.com/astorfi/TensorFlow-World/tree/master/codes/1-basics/variables>`__ 
可以在GitHub存储库中找到。

创建变量
------------------

对于一个变量的生成，我们将使用 tf.Variable() 类。当
当我们定义一个变量时，我们至少将一个 ``张量`` 和它的 ``值``
传递给图像。正常情况下会发生以下情况：

    -  包含一个值的 ``变量`` 张量将传递给
       图像。
通过使用tf.assign，变量初始化器设置初始值。

一些变量可以像如下定义：

.. code:: python

     
    import tensorflow as tf
    from tensorflow.python.framework import ops

    #######################################
    ############# 定义变量 ################
    #######################################

    # 使用默认值创建3个变量
    weights = tf.Variable(tf.random_normal([2, 3], stddev=0.1),
                          name="weights")
    biases = tf.Variable(tf.zeros([3]), name="biases")
    custom_variable = tf.Variable(tf.zeros([3]), name="custom")

    # 获取所有变量的张量并存储在列表中
    all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    

在上面的脚本中， ``ops.get_collection`` 
从定义的图像中获取所有已定义变量的列表。"name"键，为图表上的每个变量
定义了一个特定的名称。

初始化
--------------

必须在模型中的所有其他操作之前
运行变量的``变量初始化器``。作为类比，我们可以考虑
汽车的启动器。变量也可以从
已保存的模型，例如检查文件中“恢复”。变量
可以全局，或特别地初始化或是从其他变量初始化。我们
将在后续章节中研究不同的选择。

初始化特定变量
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By using tf.variables\_initializer, we can explicitly command the
TensorFlow仅初始化某个特定的变量。 脚本如下

.. code:: python
     
    # “variable_list_custom”是我们要初始化的变量列表。
    variable_list_custom = [weights, custom_variable]

    # 初始化器
    init_custom_op = tf.variables_initializer(var_list=variable_list_custom)

注意到自定义初始化并不意味着我们不需要
初始化其它变量！所有可以根据图表来完成计算的变量
都必须初始化或从已保存的变量中恢复。
这只允许我们去手动初始化
特定变量。

全局变量初始化
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

你可以使用 
tf.global\_variables\_initializer(). This op must be run after the model constructed. 
脚本如下：

.. code:: python
     
    # 方法-1
    # 添加一个op来初始化变量
    init_all_op = tf.global_variables_initializer()

    # 方法-2
    init_all_op = tf.variables_initializer(var_list=all_variables_list)

所有以上提供的方法都是相同的，我们只提供第二种来
证明``tf.global_variables_initializer()``什么都不是
但是当你在输入参数产生变量时，``tf.variables_initializer`` 

使用现有变量初始化变量
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New variables can be initialized using other existing variables' initial
values by taking the values using initialized\_value().

Initialization using predefined variables' values

.. code:: python

    # Create another variable with the same value as 'weights'.
    WeightsNew = tf.Variable(weights.initialized_value(), name="WeightsNew")

    # Now, the variable must be initialized.
    init_WeightsNew_op = tf.variables_initializer(var_list=[WeightsNew])

As it can be seen from the above script, the ``WeightsNew`` variable is
initialized with the values of the ``weights`` predefined value.

Running the session
-------------------

All we did so far was to define the initializers' ops and put them on the
graph. In order to truly initialize variables, the defined initializers'
ops must be run in the session. The script is as follows:

Running the session for initialization

.. code:: python

    with tf.Session() as sess:
        # Run the initializer operation.
        sess.run(init_all_op)
        sess.run(init_custom_op)
        sess.run(init_WeightsNew_op)

Each of the initializers has been run separated using a session.

Summary
-------

In this tutorial, we walked through the variable creation and
initialization. The global, custom and inherited variable initialization
have been investigated. In the future posts, we investigate how to save
and restore the variables. Restoring a variable eliminate the necessity
of its initialization.

