============================
欢迎来到TensorFlow的世界！
============================

.. _this link: https://github.com/astorfi/TensorFlow-World/tree/master/codes/0-welcome

本节中的教程只是进入TensorFlow世界的一个开始。

我们使用Tensorboard来将结果可视化。TensorBoard是TensorFlow提供的图像可视化工具。使用谷歌的话说：“您可以使用TensorFlow进行像训练一个大规模的深度神经网络一样复杂而混乱的计算。为了更容易理解，调试和优化TensorFlow程序，我们提供了一套名为TensorBoard的可视化工具。在本教程中包含了一个简单的Tensorboard的安装及启用。

**注意：***
     
* 摘要:Tensorboard及其优点的介绍超出了本教程的范围，这将会在更高级的教程中介绍。


--------------------------
准备开发环境
--------------------------

首先，我们必须导入必要的数据库。

.. code:: python
    
       from __future__ import print_function
       import tensorflow as tf
       import os

由于我们的目标是使用Tensorboard，所以我们需要一个目录来存储信息（如果用户需要，可以记录操作及其相应的输出）。此信息由TensorFlow导出到 ``日志文件`` ，我们可以将日志文件转换为可视的数据，以便用户能够评估架构和操作。存储这些日志文件的 ``路径`` 定义如下：

.. code:: python
    
       # 保存日志文件的路径与此python文件所在的文件夹相同
       tf.app.flags.DEFINE_string(
       'log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
       'Directory where event logs are written to.')

       # 将所有元素存储在FLAG结构中
       FLAGS = tf.app.flags.FLAGS

``os.path.dirname(os.path.abspath(__file__))`` 用来获取当前python文件的目录名。``tf.app.flags.FLAGS`` 指向所有使用 ``FLAGS`` 指示符定义的flags。从现在起，你可以使用 ``FLAGS.flag_name`` 调用flags。

为方便起见，仅仅使用 ``绝对路径`` 是很有用的。通过使用以下代码脚本，用户可以大致了解到如何使用绝对路径来表示 ``log_dir`` 目录。

.. code:: python

    # 输入绝对路径
    # os.path.expanduser用于将 '~' 符号转换为相应的路径指示符。
    #       示例： '~/logs' 等于 '/home/username/logs'
    if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
        raise ValueError('You must assign absolute path for --log_dir')

--------
基础
--------

一些基础的数学运算可以被tensorflow定义：

.. code:: python

     # 定义一些常量
     a = tf.constant(5.0, name="a")
     b = tf.constant(10.0, name="b")

     # 一些基础的运算
     x = tf.add(a, b, name="add")
     y = tf.div(a, b, name="divide")
    
The ``tf.`` operator performs the specific operation and the output will be a ``Tensor``. The attribute ``name="some_name"`` is defined for better Tensorboard visualization as we see later in this tutorial.

-------------------
运行实验
-------------------

``session`` ，是运行的环境，执行如下： 

.. code:: python

    # 运行会话
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
        print("output: ", sess.run([a,b,x,y]))

    # 关闭写入者
    writer.close()
    sess.close()

定义 ``tf.summary.FileWriter`` 是为了将摘要写入日志文件。 ``sess.run()`` 命令必须用于评估所有Tensor，否则操作将不会执行。最后，通过使用 ``writer.close()`` ，摘要写入器将被关闭。
    
--------
Results
--------

在终端中运行的结果如下所示：

.. code:: shell

        [5.0, 10.0, 15.0, 0.5]


如果我们使用 ``tensorboard --logdir="absolute/path/to/log_dir"`` 来运行Tensorboard，我们在可视化 ``图像`` 时会得到如下结果：

.. figure:: https://github.com/astorfi/TensorFlow-World/blob/master/docs/_img/1-basics/basic_math_operations/graph-run.png
   :scale: 30 %
   :align: center

   **图1：** TensorFlow图像

