# Copyright (c) Roman Lutz. All rights reserved.
# The use and distribution terms for this software are covered by the
# Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
# which can be found in the file LICENSE.md at the root of this distribution.
# By using this software in any fashion, you are agreeing to be bound by
# the terms of this license.
# You must not remove this notice, or any other, from this software.


from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from metrics import mean_relative_error
import numpy as np
from get_data import test_players

# load data
# indices are
# 0: QB id
# 1: QB name
# 2: QB age
# 3: QB years pro
# 4-15: last game QB stats
# 16-27: last 10 games QB stats
# 28-31: last game defense stats
# 32-35: last 10 games defense stats
# 36-43: actual fantasy score = target
train = np.load('train.npy')
test = np.load('test.npy')

train_x = train[:, 2:36].astype(np.float)
train_y = train[:, 36].astype(np.float)  # 1 column
test_x = test[:, 2:36].astype(np.float)
test_y = test[:, 36].astype(np.float)

# determine indices in test set of 24 best players
indices = []
for index in range(len(test_x)):
    if test[index, 0] in test_players.keys():
        indices.append(index)

number_of_features = train_x.shape[1]

ds = SupervisedDataSet(number_of_features, 1)

ds.setField('input', train_x)
ds.setField('target', train_y.reshape((len(train_y), 1)))

file = open("neural_net_output.txt", "w")
file.write('RMSE(all), RMSE(24), MAE(all), MAE(24), MRE(all), MRE(24):\n')

for number_of_epochs in [10, 50, 100, 1000]:
    for number_of_hidden_units in [10, 25, 50, 100]:
        for hidden_class in [SigmoidLayer, TanhLayer]:
            if hidden_class == TanhLayer:
                hidden_class_name = 'Tanh Layer'
            elif hidden_class == LinearLayer:
                hidden_class_name = 'Linear Layer'
            else:
                hidden_class_name = 'Sigmoid Layer'

            #Build Neural Network
            net = buildNetwork(number_of_features, # input
                   number_of_hidden_units, # number of hidden units
                   1, # output neurons
                   bias = True,
                   hiddenclass = hidden_class,
                   outclass = LinearLayer
                   )

            trainer = BackpropTrainer(net, ds, learningrate=0.01, lrdecay=1.0, momentum=0.0, weightdecay=0.0, verbose=True)

            trainer.trainUntilConvergence(maxEpochs=number_of_epochs)

            predictions = []
            for x in test_x:
                predictions.append(net.activate(x)[0])

            predictions = np.array(predictions)

            # format output for LaTeX
            file.write('%d %d %s %f & %f &  %f & %f & %f & %f \\\\ \n' % (number_of_epochs,
                                                                          number_of_hidden_units,
                                                                          hidden_class_name,
                                                                          mean_squared_error(test_y, predictions)**0.5,
                                                                          mean_absolute_error(test_y, predictions),
                                                                          mean_relative_error(test_y, predictions),
                                                                          mean_squared_error(test_y[indices], predictions[indices])**0.5,
                                                                          mean_absolute_error(test_y[indices], predictions[indices]),
                                                                          mean_relative_error(test_y[indices], predictions[indices])))
            print zip(test_y[indices], predictions[indices])

file.close()