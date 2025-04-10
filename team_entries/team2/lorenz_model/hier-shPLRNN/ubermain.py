from multitasking import *


def ubermain(n_runs):
    """
    Specify the argument choices you want to be tested here in list format:
    e.g. args.append(Argument('dim_z', [5, 6], add_to_name_as='z'))
    will test for dimensions 5 and 6 and save experiments under z5 and z6.
    Possible Arguments can be found in main.py
    
    When using GPU for training (i.e. Argument 'use_gpu 1')  it is generally
    not necessary to specify device ids, tasks will be distributed automatically.
    """
    args = []

    args.append(Argument('num_epochs', [5000]))
    args.append(Argument('batch_size', [1024]))
    args.append(Argument('batches_per_epoch', [50]))

    args.append(Argument('data_path', ['./data/lorenz63/3params64sub/noisy.pt']))
    args.append(Argument('eval_data_path', ['./data/lorenz63/3params64sub/full.pt']))
    args.append(Argument('train_set_size', [1000]))

    args.append(Argument('experiment', ['lorenz63']))
    args.append(Argument('run', list(range(1, 1 + n_runs))))

    args.append(Argument('hierarchisation_scheme', ['projection'], add_to_name_as=''))
    args.append(Argument('num_individual_params', [6], add_to_name_as='dp'))

    args.append(Argument('obs_model', ['identity']))
    args.append(Argument('obs_size', [3]))
    args.append(Argument('latent_size', [10]))
    args.append(Argument('forcing_size', [3]))
    args.append(Argument('hidden_size', [50], add_to_name_as='dh'))

    args.append(Argument('learning_rate', [1e-3]))
    args.append(Argument('individual_learning_rate', [1e-2]))

    args.append(Argument('tf_alpha_start', [.2]))
    args.append(Argument('tf_alpha_end', [.02]))

    args.append(Argument('seq_len', [30]))

    args.append(Argument('weight_decay', [0]))

    # args.append(Argument('compile', ['']))
    args.append(Argument('use_gpu', ['']))

    args.append(Argument('metrics', ['kl pse']))
    args.append(Argument('plots', ['pow hier 3D']))

    args.append(Argument('lam', [0]))

    # args.append(Argument('clip_grad_norm', [10]))

    args.append(Argument('learn_noise_cov', ['']))

    args.append(Argument('num_workers', [0]))

    return args


if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # number of runs for each experiment
    n_runs = 1
    # number of processes run parallel on a single GPU
    n_proc_per_gpu = 1
    # number of runs to run in parallel
    n_cpu = 1 * n_proc_per_gpu

    args = ubermain(n_runs)
    run_settings(*create_tasks_from_arguments(args, n_proc_per_gpu, n_cpu))