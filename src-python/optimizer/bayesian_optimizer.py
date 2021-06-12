from hyperopt import hp, tpe, fmin, STATUS_OK
from hyperopt.pyll.stochastic import sample
from hyperopt import Trials
from functools import partial
from trainer.training import UNetExperiment
from configuration import Config
import numpy as np
from utils.utils import LoadHippocampusData



#Create global and costants variables
global ITERATION
ITERATION = 0
MAX_EVALS = 20



#Create the objective function
def objective(params, c, data, split):
  global ITERATION
  ITERATION += 1

  params['batch_size'] = int(params['batch_size'])
  c.learning_rate =  params['learning_rate']
  c.batch_size = params['batch_size']
  print(c.learning_rate)
  print(c.batch_size)
  exp = UNetExperiment(c, split, data)
  dict_results = exp.run()
  final_loss = np.min(dict_results['val_loss'])

  #return a dictionary with information about the loss
  return {'loss':final_loss, 'params':params, 'iteration':ITERATION, 'status':STATUS_OK}


def bayes_optimization():
  #Initizalize config and retrieve data
  c = Config()
  data = LoadHippocampusData(c.root_dir, y_shape=c.patch_size, z_shape=c.patch_size)
  #Split the data
  train_val_keys = int(0.8*len(data))
  train_keys = int(0.8*train_val_keys)

  #shuffle_keys = torch.randperm(len(data))
  shuffle_keys = np.arange(len(data))
  train_indices = shuffle_keys[:train_keys]
  val_indices = shuffle_keys[train_keys:train_val_keys]
  test_indices = shuffle_keys[train_val_keys:]

  split = dict()
  split["train"] = train_indices
  split["val"] = val_indices
  split["test"] = test_indices


  fmin_objective = partial(objective, c=c, data=data, split=split)

  #Create a domain space
  params_space = {'learning_rate' : hp.loguniform('learning_rate', np.log(0.0001), np.log(0.001)),
                'batch_size' : hp.quniform('batch_size', 2, 33, 4)}

  #Create tpe algorithm for the surrogate function and for the choice of the next param set
  tpe_algorithm = tpe.suggest
  #Keep track of the results
  bayes_trials = Trials()

  #Run Optimization
  best = fmin(fn=fmin_objective, space=params_space, algo=tpe_algorithm, 
            max_evals=MAX_EVALS, trials=bayes_trials, 
            rstate=np.random.RandomState(50))
  
  return best