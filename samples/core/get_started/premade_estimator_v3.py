#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Example using a csv reader in the input function.
   https://www.tensorflow.org/get_started/datasets_quickstart"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import iris_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def main(argv):

    args = parser.parse_args(argv[1:])

    train_path, test_path = iris_data.maybe_download()

    feature_columns = [
        tf.feature_column.numeric_column(name)
        for name in iris_data.CSV_COLUMN_NAMES[:-1]]

    classifier = tf.estimator.LinearClassifier(feature_columns, n_classes=3)

    classifier.train(steps=1000,
                     input_fn=lambda: iris_data.csv_input_fn(train_path, args.batch_size))

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda: iris_data.csv_eval_input_fn(test_path, args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda: iris_data.eval_input_fn(predict_x,
                                                 labels=None,
                                                 batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = '\nPrediction is "{}" ({:.1f}%), expected "{}"'

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
