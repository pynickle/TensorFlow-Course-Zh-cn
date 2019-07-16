TensorFlow变量简介：创建，初始化
--------------------------------------------------------------

本教程介绍如何定义和初始化TensorFlow变量。

介绍
------------

定义 ``变量`` 是必要的，因为它们包含参数。如果没有参数，那么训练，更新，保存，恢复和任何其他操作都无法执行。TensorFlow中定义的变量只是具有特定形状和类型的张量。，所以我们必须使用值来初始化这些张量以使其有效。在本教程中，我们将解释如何 ``定义`` 和 ``初始化`` 变量。 `源代码 <https://github.com/astorfi/TensorFlow-World/tree/master/codes/1-basics/variables>`__ 可以在GitHub存储库中找到。

创建变量
------------------

对于一个变量的生成，我们将使用 tf.Variable() 类。当我们定义一个变量时，我们至少将一个 ``张量`` 和它的 ``值`` 传递给图像。正常情况下会发生以下情况：

    - 包含一个值的 ``变量`` 张量将传递给图像。
    - 通过使用tf.assign，变量初始化器设置初始值。

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

    # Get all the variables' tensors and store them in a list.
    all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    

In the above script, ``ops.get_collection`` gets the list of all defined variables
from the defined graph. The "name" key, define a specific name for each
variable on the graph

Initialization
--------------

``Initializers`` of the variables must be run before all other
operations in the model. For an analogy, we can consider the starter of
the car. Instead of running an initializer, variables can be
``restored`` too from saved models such as a checkpoint file. Variables
can be initialized globally, specifically, or from other variables. We
investigate different choices in the subsequent sections.

Initializing Specific Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By using tf.variables\_initializer, we can explicitly command the
TensorFlow to only initialize a certain variable. The script is as follows

.. code:: python
     
    # "variable_list_custom" is the list of variables that we want to initialize.
    variable_list_custom = [weights, custom_variable]

    # The initializer
    init_custom_op = tf.variables_initializer(var_list=variable_list_custom)

Noted that custom initialization does not mean that we don't need to
initialize other variables! All variables that some operations will be
done upon them over the graph, must be initialized or restored from
saved variables. This only allows us to realize how we can initialize
specific variables by hand.

Global variable initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All variables can be initialized at once using the
tf.global\_variables\_initializer(). This op must be run after the model constructed. 
The script is as below:

.. code:: python
     
    # Method-1
    # Add an op to initialize the variables.
    init_all_op = tf.global_variables_initializer()

    # Method-2
    init_all_op = tf.variables_initializer(var_list=all_variables_list)

Both the above methods are identical. We only provide the second one to
demonstrate that the ``tf.global_variables_initializer()`` is nothing
but ``tf.variables_initializer`` when you yield all the variables as the input argument.

Initialization of a variables using other existing variables
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

