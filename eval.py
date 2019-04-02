"""
Helpers for evaluating models.
"""
import os
import numpy as np
import tqdm
import pandas as pd

from supervised_reptile.reptile import Reptile
from supervised_reptile.args import argument_parser, evaluate_kwargs
from supervised_reptile.util import mean_confidence_interval

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
             num_samples=10000,
             transductive=False,
             cuda=False,
             pin_memory=False,
             foml=False):
    """
    Evaluate a model on a dataset.
    """
    reptile_fn=FOML if foml else Reptile
    reptile = reptile_fn(num_classes, model_state, op_state, cuda, pin_memory)
    
    accuracies = []
    for _ in tqdm.trange(num_samples):
        accuracy = reptile.evaluate(dataset=dataset, num_classes=num_classes, num_shots=num_shots,
                                       inner_batch_size=eval_inner_batch_size,
                                       inner_iters=eval_inner_iters, transductive=transductive)
        accuracies.append(accuracy)
        
    return mean_confidence_interval(accuracies)  


def do_evaluation(model_state, op_state, save_dir, val_set=None, test_set=None, train_set=None):
    eval_result = pd.DataFrame(columns=['task','accuracy','me']) 
    
    for (dataset, task) in [(test_set, 'Test'), (train_set, 'Train'), (val_set, 'Validation')]:
        if dataset is not None:
            print('\nEvaluating on {} Dataset'.format(task))
            accuracy, me = evaluate(dataset, model_state, op_state, **eval_kwargs)
            print('\n{} accuracy: {:2.2%}, margin of error: {:2.2%}\n'.format(task, accuracy, me))
            eval_result.loc[len(eval_result)] = [task, accuracy, me]
            
    eval_result.to_csv(os.path.join(save_dir, r'eval_result.csv'))
    

