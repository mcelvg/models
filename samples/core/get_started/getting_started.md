
[Installing python3 on macOS](https://www.tensorflow.org/install/install_mac#install_tensorflow)

```sh
cd ~/src/github
git clone https://github.com/mcelvg/models.git
cd models
git config user.name "mcelvg"

mkdir venv
virtualenv --system-site-packages -p python3 venv
source venv/bin/activate
cd venv
easy_install -U pip
cd ..
pip3 install --upgrade tensorflow
```

[Verify installation](https://www.tensorflow.org/install/install_mac#ValidateYourInstallation)

```python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
exit()
```

```
$ python
Python 3.6.4 (default, Jan  6 2018, 11:51:15)
[GCC 4.2.1 Compatible Apple LLVM 9.0.0 (clang-900.0.39.2)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
2018-02-24 16:14:54.824355: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.2 AVX AVX2 FMA
>>> print(sess.run(hello))
b'Hello, TensorFlow!'
>>> exit()
```


[Getting Started with TensorFlow](https://www.tensorflow.org/get_started/premade_estimators)

* [premade_estimator.py](https://github.com/mcelvg/models/blob/master/samples/core/get_started/premade_estimator.py)  
  _An Example of a DNNClassifier for the Iris dataset._
* [premade_estimator_v2.py](https://github.com/mcelvg/models/blob/master/samples/core/get_started/premade_estimator_v2.py)  
  _Example using a custom checkpoint configuration._
* [Datasets Quick Start](https://www.tensorflow.org/get_started/datasets_quickstart)  
  _Example using a csv reader in the input function._  
  [premade_estimator_v3.py](https://github.com/mcelvg/models/blob/master/samples/core/get_started/premade_estimator_v3.py), [iris_data#csv_input_fn](https://github.com/mcelvg/models/blob/master/samples/core/get_started/iris_data.py#L82), and [iris_data#csv_eval_input_fn](https://github.com/mcelvg/models/blob/master/samples/core/get_started/iris_data.py#L96)  


[Creating Custom Estimators](https://www.tensorflow.org/get_started/custom_estimators)

* _Custom estimator mimicking a `DNNClassifier` for the Iris data set_ ([custom_estimator.py](https://github.com/mcelvg/models/blob/master/samples/core/get_started/custom_estimator.py))
* Walk through branching model function
* Minimal demo of tensorboard
