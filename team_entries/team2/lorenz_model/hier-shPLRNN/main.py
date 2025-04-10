#%%
import argparse
import torch
from bptt import BPTT
from dataset import MultiSubjectDataset

torch.set_num_threads(1)
torch.set_float32_matmul_precision('high')

# torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a hierarchical PLRNN model.")
    parser.add_argument('--data_path', type=str, default='./data/ml4de/lorenz_training2.pt', help='path to the data')

    # parser.add_argument('--eval_data_path', type=str, default='./data/ml4de/lorenz_truth2_LONG.pt', help='path to the evaluation data if available')
    parser.add_argument('--eval_data_path', type=str, default='./data/ml4de/lorenz_truth2.pt', help='path to the evaluation data if available')

    parser.add_argument('--obs_size', type=int, default=3, help='size of the observation space')
    parser.add_argument('--latent_size', type=int, default=None, help='size of the latent state')
    parser.add_argument('--obs_model', type=str, default='identity', help='observation model (either linear or identity)')
    parser.add_argument('--hidden_size', type=int, default=50, help='size of the hidden state')
    parser.add_argument('--forcing_size', type=int, default=None, help='number of latent dimensions that are teacher forced. Defaults to all.')

    parser.add_argument('--seq_len', type=int, default=30, help='length of the sequences during training')
    parser.add_argument('--train_set_size', type=int, default=1000, help='size of the data slice that is used for training.')

    parser.add_argument('--hierarchisation_scheme', type=str, default='projection', help='name of the hierarchisation scheme (either projection or baseline)')
    parser.add_argument('--num_individual_params', type=int, default=10, help='number of subject specific parameters')

    parser.add_argument('--tf_alpha_start', type=float, default=0.1, help='teacher forcing alpha at the start of training')
    parser.add_argument('--tf_alpha_end', type=float, default=None, help='teacher forcing alpha at the end of training.')

    parser.add_argument('--num_epochs', type=int, default=5000, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--batches_per_epoch', '-bpe', type=int, default=None, help='batches per epoch. Defaults to the entire dataset.')
    parser.add_argument('--subjects_per_batch', type=int, default=None, help='number of subjects per batch. Defaults to using all.')
    parser.add_argument('--num_workers', type=int, default=None, help='number of workers for the dataloader')

    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--individual_learning_rate', '-ilr', type=float, default=None, help='use a different learning rate for the shared parameters than for the individual ones. This specifies the lr for the latter.')
    parser.add_argument('--weight_decay', type=float, default=.0, help='L2 regularization strength (only for shared parameters)')

    # parser.add_argument('--save_path', type=str, default='./trained_models/ml4de', help='path to save the results')
    parser.add_argument('--save_path', type=str, default='./trained_models', help='path to save the results')
    parser.add_argument('--experiment', type=str, default='experiment', help='name of the experiment')
    parser.add_argument('--name', type=str, default='name', help='name of the model')
    parser.add_argument('--run', type=int, default=1, help='run number')

    parser.add_argument('--finetune', action='store_true', help='finetune a pre trained model.')
    parser.add_argument('--model_path', type=str, default=None, help='model path to pre trained model')
    parser.add_argument('--checkpoint', type=int, default=None, help='which checkpoint to load. If None, loads the latest.')

    parser.add_argument('--use_gpu', action='store_true', help='use GPU')
    parser.add_argument('--device_id', type=int, default=0, help='Which GPU to use. This will be handled automatically when using.')

    parser.add_argument('--compile', action='store_true', help='Whether to use TorchInductor for compilation. -> Speeds up inference but may break things')
    
    parser.add_argument('--metrics', nargs='*', default=['kl', 'pse'], help='Metrics to evaluate the model on. State space divergence (kl), power spectrum error (pse).')
    parser.add_argument('--kl_bins', type=int, default=30, help='Number of bins per dimension for the state space divergence metric. If 0 uses gmm variant.')
    parser.add_argument('--pse_smooth', type=int, default=15, help='Smoothing sigma for the power spectrum error metric.')
    parser.add_argument('--plots', nargs='*', default=['pow', 'hier'], help='Plots to generate. 3D, hovmoller, pow, hier. Defaults to pow and hier.')

    parser.add_argument('--lam', type=float, default=0.0, help='lambda for the hierarchical loss. defaults to 0.0')

    parser.add_argument('--clip_grad_norm', type=float, default=0.0, help='Clip gradient norm to this value. Defaults to 0, in which cases no clipping is done.')

    parser.add_argument('--clipped', action='store_true', help='Whether to use clipped shallow PLRNN.')

    parser.add_argument('--learn_noise_cov', action='store_true', help='Whether to learn the diagonal noise covariance matrix.')

    return parser.parse_args()

def get_dataset(args):
    dataset = MultiSubjectDataset(args.data_path, args.seq_len, args.train_set_size, args.subjects_per_batch, args.num_workers, args.device)
    return dataset

def get_device(args):
    args.device = 'cpu'
    if args.use_gpu:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        try:
            args.device = args.device + ':' + str(args.device_id)
        except AttributeError:
            print("Must specify device id when using cuda. This should be handeled by the task distribution.")
    print(f'Using device: {args.device}')
    return args

def handle_defaults(args):
    # set learning rates
    if args.individual_learning_rate is not None:
        args.learning_rate = (args.learning_rate, args.individual_learning_rate)
    else:
        args.learning_rate = (args.learning_rate, args.learning_rate)
    # set teacher forcing alpha
    if args.tf_alpha_end is None:
        args.tf_alpha_end = args.tf_alpha_start
    return args

def main():
    args = parse_args()
    args = get_device(args)
    args = handle_defaults(args)

    dataset = get_dataset(args)

    training_algorithm = BPTT(args, dataset)
    fun = training_algorithm.train if not args.finetune else training_algorithm.finetune
    if args.compile:
        torch.compile(fun)()
    else:
        fun()

if __name__ == '__main__':
    main()