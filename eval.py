"""
Helpers for evaluating models.
"""
import numpy as np
import tqdm
import pandas as pd

from supervised_reptile.reptile import Reptile
from supervised_reptile.args import argument_parser, evaluate_kwargs

args = argument_parser().parse_args()
eval_kwargs = evaluate_kwargs(args)

# pylint: disable=R0913,R0914
def evaluate(dataset,
             model_state, 
             op_state,
             num_classes=5,
             num_shots=5,
             eval_inner_batch_size=5,
             eval_inner_iters=50,
             replacement=False,
             num_samples=10000,
             transductive=False,
             cuda=False):
    """
    Evaluate a model on a dataset.
    """
    reptile = Reptile(num_classes, model_state, op_state)
    
    accuracies = []
    for _ in tqdm.trange(1, num_samples+1):
        accuracy = reptile.evaluate(dataset=dataset, num_classes=num_classes, num_shots=num_shots,
                                       inner_batch_size=eval_inner_batch_size,
                                       inner_iters=eval_inner_iters, replacement=replacement, cuda=cuda)
        accuracies.append(accuracy)
        
    return np.mean(accuracies), np.std(accuracies)    


def do_evaluation(model_state, op_state, save_dir, val_set=None, test_set=None, train_set=None):
    eval_result = pd.DataFrame(columns=['mode','accuracy','std']) 
    
    for (dataset, mode) in [(val_set, 'Validation'), (test_set, 'Test'), (train_set, 'Train')]:
        if dataset is not None:
            print('\nEvaluating on {} Dataset'.format(mode))
            accuracy, std = evaluate(dataset, model_state, op_state, **eval_kwargs)
            print('\n{} accuracy: {:2.0%}, std: {:2.0%}'.format(mode, accuracy, std))
            eval_result.loc[len(eval_result)] = [mode, accuracy, std]
            
    eval_result.to_csv(os.path.join(save_dir, r'eval_result.csv'))