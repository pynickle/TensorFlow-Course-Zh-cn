==================================


==================================

'
/
h
o
m



e

/

u

s

------------------------
e
------------------------

r
 
n
a
m

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
e
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/

.. code:: bash

    sudo apt-get install python-numpy python-dev python-pip python-wheel python-virtualenv
    sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel python3-virtualenv
    
The second line is for ``python3`` installation.

~~~~~~~~~~~~~~~~~~~
l
~~~~~~~~~~~~~~~~~~~

o

g

.. code:: bash

    kernel version X does not match DSO version Y -- cannot find working devices in this configuration
    
For solving that error you may need to purge all NVIDIA drivers and install or update them again. Please refer to `CUDA Installation`_ for further detail.


    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'

开
幕
式

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
一
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

些

.. code:: bash

    sudo mkdir ~/virtualenvs

语

.. code:: bash

    sudo virtualenv --system-site-packages ~/virtualenvs/tensorflow

句

可

.. code:: bash

    source ~/virtualenvs/tensorflow/bin/activate

以

被

T

.. code:: bash

    echo 'alias tensorflow="source $HOME/virtualenvs/tensorflow/bin/activate" ' >> ~/.bash_aliases
    bash

e

.. code:: bash

    tensorflow
    
**check the ``~/.bash_aliases``**

n

.. code:: shell

    alias tensorflow="source $HO~/virtualenvs/tensorflow/bin/activate" 
    

s

o
 
.. code:: shell

    if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
    fi
 

    
---------------------------------
Configuration of the Installation
---------------------------------

r

.. code:: bash

     git clone https://github.com/tensorflow/tensorflow 

F

.. code:: bash

    cd tensorflow  # cd to the cloned directory

l

.. code:: bash

    $ ./configure
    Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python2.7
    Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:
    Do you wish to use jemalloc as the malloc implementation? [Y/n] Y
    jemalloc enabled
    Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] N
    No Google Cloud Platform support will be enabled for TensorFlow
    Do you wish to build TensorFlow with Hadoop File System support? [y/N] N
    No Hadoop File System support will be enabled for TensorFlow
    Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] N
    No XLA JIT support will be enabled for TensorFlow
    Found possible Python library paths:
      /usr/local/lib/python2.7/dist-packages
      /usr/lib/python2.7/dist-packages
    Please input the desired Python library path to use.  Default is [/usr/local/lib/python2.7/dist-packages]
    Using python library path: /usr/local/lib/python2.7/dist-packages
    Do you wish to build TensorFlow with OpenCL support? [y/N] N
    No OpenCL support will be enabled for TensorFlow
    Do you wish to build TensorFlow with CUDA support? [y/N] Y
    CUDA support will be enabled for TensorFlow
    Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
    Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave empty to use system default]: 8.0
    Please specify the location where CUDA 8.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
    Please specify the cuDNN version you want to use. [Leave empty to use system default]: 5.1.10
    Please specify the location where cuDNN 5 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
    Please specify a list of comma-separated Cuda compute capabilities you want to build with.
    You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
    Please note that each additional compute capability significantly increases your build time and binary size.
    [Default is: "3.5,5.2"]: "5.2"


o
w
定
义

：


     
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




.. code:: bash

    ./configure
    bazel test ...

---------------------


---------------------



    
#

.. code:: bash

    bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
    
The ``bazel build`` command builds a script named build_pip_package. Running the following script build a .whl file within the ~/tensorflow_package directory:

.. code:: bash

    bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tensorflow_package





-------------------------------


-------------------------------

定

~~~~~~~~~~~~~~~~~~~~~~~~~~~
义
~~~~~~~~~~~~~~~~~~~~~~~~~~~

一

.. code:: bash

    sudo pip install ~/tensorflow_package/file_name.whl
    

~~~~~~~~~~~~~~~~~~~~~~~~~~~
些
~~~~~~~~~~~~~~~~~~~~~~~~~~~

语

.. code:: bash
    
    pip install ~/tensorflow_package/file_name.whl

**WARNING**:
           * By using the virtual environment installation method, the sudo command should not be used anymore because if we use sudo, it points to native system packages and not the one available in the virtual environment.
           * Since ``sudo mkdir ~/virtualenvs`` is used for creating of the virtual environment, using the ``pip install`` returns ``permission error``. In this case, the root privilege of the environment directory must be changed using the ``sudo chmod -R 777 ~/virtualenvs`` command.
    
--------------------------
Validate the Installation
--------------------------

句

.. code:: bash

    python
    >> import tensorflow as tf
    >> hello = tf.constant('Hello, TensorFlow!')
    >> sess = tf.Session()
    >> print(sess.run(hello))

--------------------------
吧
--------------------------

！

运
行



--------------------------
使
--------------------------

验




