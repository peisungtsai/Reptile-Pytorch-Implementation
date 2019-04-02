"""
Train a model on miniImageNet.
"""

import random
import torch

from supervised_reptile.args import argument_parser, train_kwargs
from supervised_reptile.eval import do_evaluation
from supervised_reptile.image_loader import read_dataset
from supervised_reptile.train import train
from supervised_reptile.util import load_checkpoint

DATA_DIR='data/miniimagenet'   

def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    print(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)

    train_set, val_set, test_set = read_dataset(DATA_DIR)

    if not args.pretrained:
        print('Training...')
        train(train_set, test_set, args.checkpoint, **train_kwargs(args))
    else:
        print('Restoring from checkpoint...')
        model_state, op_state, meta_iteration, cur_meta_step_size, accuracy_tracking = load_checkpoint(args.checkpoint)
        train(train_set, test_set, args.checkpoint, model_state, op_state, **train_kwargs(args))            

    print('\nEvaluating...')
    model_state, op_state, meta_iteration, cur_meta_step_size, accuracy_tracking = load_checkpoint(args.checkpoint)
    do_evaluation(model_state, op_state, args.checkpoint, val_set, test_set, train_set)   

if __name__ == '__main__':
    main()

"""
--shots 5 --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-step-final 1 --meta-batch 5 --meta-iters 100000 --eval-batch 15 --eval-iters 50 
--learning-rate 0.001 --train-shots 16 --checkpoint ckpt_m55 --cuda --pin_memory --pretrained


python -u run_miniimagenet.py --shots 5 --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-step-final 0 --meta-batch 5 --meta-iters 100000 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --train-shots 16 --checkpoint ckpt_m55 --cuda --pin_memory --transductive --foml
"""