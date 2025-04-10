import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange
import os
from argparse import Namespace
from model import shallowPLRNN, nll_loss


def load_from_path(model, args, checkpoint=None):
    """Loads a pre trained model to finetune.
    Args:
        model: The "empty" model
        args: The parsed arguments
        checkpoint: The checkpoint to load. If None, the most recent model is loaded."""
    # handle specific model given
    path = args.model_path
    if checkpoint is None:
        # find most recent model
        cps = []
        for f in os.listdir(path):
            if f.split('.')[-1] == 'pt': cps.append(int(f.split('.')[0].split('_')[-1]))
        assert len(cps) > 0, 'No model files found in specified folder.'
        path = os.path.join(path, f'model_{max(cps)}.pt')
    else:
        path = os.path.join(path, f'model_{checkpoint}.pt')
    # load
    print(f'Loading model from {path}')
    # load state dict (on specified device)
    statedict = torch.load(path, map_location=model.device)
    # remove '_orig_mod.' from keys, which are added by the torch compile function
    # this way the model can be loaded without the need to use compile again if it was trained with it
    statedict = {k.replace('_orig_mod.', ''): v for k, v in statedict.items()}
    # when finetuning remove p_vector and noise_cov from the state dict
    if hasattr(args, "finetune") and args.finetune:
        statedict.pop('p_vector')
        statedict.pop('noise_cov')
    model.load_state_dict(statedict, strict=False)

def read_hypers(args):
    """Reads the hyperparams of the model at args.model_path from the hypers.txt"""
    new_args = Namespace()
    print(f"Frustratingm print thgis path: {args.model_path}")
    with open(os.path.join(args.model_path, 'hypers.txt')) as file:
        for line in file.readlines():
            line_ = line.split(': ')
            name, val = line_[0], line_[1]
            val = val.strip('\n')
            # find the correct type
            try:
                val = float(val)
                if val.is_integer():
                    val = int(val)
            except:
                try:
                    val = eval(val)
                except:
                    try:
                        parts = val.strip('()').split(',')
                        if len(parts) > 1:
                            val = tuple()
                    except:
                        pass
            setattr(new_args, name, val)
    return new_args

def edit_args(args, new_args):
    """Edits the args with the new_args.
    Args:
        args: The original args
        new_args: The new args"""
    for name in ['data_path', 'eval_data_path', 'experiment', 'name', 'run', 'device']:
        if getattr(args, name) is not None:
            setattr(new_args, name, getattr(args, name))
    return new_args

class BPTT:
    """Training class for backpropagation through time."""
    def __init__(self, args, dataset):
        """Args:
            model: model to train
            dataset: dataset to train on"""
        self.args = args
        # initialize model
        if args.model_path is not None: # load state dict from file if specified
            modelargs = read_hypers(args)
            modelargs = edit_args(args, modelargs)
            self.model = shallowPLRNN(modelargs, dataset)
            load_from_path(self.model, args)
        else:
            self.model = shallowPLRNN(args, dataset)
        # compile model if specified
        if args.compile:
            self.model = torch.compile(self.model)
        # initialize dataset
        self.dataset = dataset
        self.bpe = args.batches_per_epoch
        if self.bpe is None:
            self.bpe = len(dataset)//args.batch_size
        # initialize optimizers
        shared, individual = self.model.hierarchisation_scheme.grouped_parameters()
        self.shared_optimizer = Adam(shared, lr=args.learning_rate[0], weight_decay=args.weight_decay)
        self.individual_optimizer = Adam(individual, lr=args.learning_rate[1])
        # exponential LR schedule leads to compilation issues due to float multiplication (?)
        # therefore, we use a lambda function to decay the LR and do the multiplication with a tensor
        self.shared_scheduler = LambdaLR(self.shared_optimizer, lambda epoch: torch.tensor(0.999)**epoch)
        self.individual_scheduler = LambdaLR(self.individual_optimizer, lambda epoch: torch.tensor(0.999)**epoch)
        # initialize criterion
        self.criterion = nll_loss
        # move model to device
        self.model.to(args.device)

    def train(self):
        """Trains the model."""
        # initiate train mode
        self.model.train()
        # initialize progress bar
        pbar = trange(self.args.num_epochs)
        for e in pbar:
            self.model.hierarchisation_scheme.step = e+1
            epoch_losses = {'rnn': 0, 'hier': 0}
            dloader = self.dataset.get_dataloader(batch_size=self.args.batch_size, bpe=self.bpe)
            for data, target, subject in dloader:
                # move data to device
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                subject = subject.to(self.args.device)
                # forward pass
                prediction = self.model(data, subject)
                # reset gradients
                self.shared_optimizer.zero_grad()
                self.individual_optimizer.zero_grad()
                # calculate losses
                rnn_loss = self.criterion(prediction, target, self.model.noise_cov[subject])
                hier_loss = torch.tensor(0)
                if self.args.lam > 0:
                    hier_loss = self.args.lam*self.model.hierarchisation_scheme.loss()
                # backpropagate
                (rnn_loss + hier_loss).backward()
                # clip gradients
                if self.args.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                # update parameters
                self.shared_optimizer.step()
                self.individual_optimizer.step()
                # add loss to epoch loss
                epoch_losses['rnn'] += rnn_loss.item()
                epoch_losses['hier'] += hier_loss.item()
            # update progress bar
            for k, _ in epoch_losses.items():
                epoch_losses[k] /= len(dloader)
            pbar.set_postfix({'loss': sum(epoch_losses.values())})
            # update lr
            self.shared_scheduler.step()
            self.individual_scheduler.step()
            # update tf parameter
            self.model.tf_alpha *= self.model.tf_gamma
            self.model.saver.writer.add_scalar('tf_alpha', self.model.tf_alpha, e+1)
            # save model, stats and plots
            self.model.saver.save_loss(e+1, epoch_losses)

            save_expensive_every = 100   ## nSHould be 500
            save_cheap_every = 100   ## nShould be 100
            if (e+1)%save_expensive_every == 0:
                self.model.saver.save_expensive(e+1)
            elif (e+1)%save_cheap_every == 0:
                self.model.saver.save_cheap(e+1)

    def finetune(self):
        """Finetunes a model. Copied train method but with only the individual optimizer."""
        # initiate train mode
        self.model.train()
        # initialize progress bar
        pbar = trange(self.args.num_epochs)
        for e in pbar:
            self.model.hierarchisation_scheme.step = e+1
            epoch_losses = {'rnn': 0, 'hier': 0}
            dloader = self.dataset.get_dataloader(batch_size=self.args.batch_size, bpe=self.bpe)
            for data, target, subject in dloader:
                # if using subjects_per_batch, reshuffle the pool for next batch
                if self.args.subjects_per_batch is not None:
                    self.dataset.shuffle_subjects()
                # move data to device
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                subject = subject.to(self.args.device)
                # forward pass
                prediction = self.model(data, subject)
                # reset gradients
                self.individual_optimizer.zero_grad()
                # calculate losses
                rnn_loss = self.criterion(prediction, target, self.model.noise_cov[subject])
                hier_loss = torch.tensor(0)
                if self.args.lam > 0:
                    hier_loss = self.args.lam*self.model.hierarchisation_scheme.loss()
                # backpropagate
                (rnn_loss + hier_loss).backward()
                # clip gradients
                if self.args.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                # update parameters
                self.individual_optimizer.step()
                # add loss to epoch loss
                epoch_losses['rnn'] += rnn_loss.item()
                epoch_losses['hier'] += hier_loss.item()
            # update progress bar
            for k, _ in epoch_losses.items():
                epoch_losses[k] /= len(dloader)
            pbar.set_postfix({'loss': sum(epoch_losses.values())})
            # update lr
            self.individual_scheduler.step()
            # update tf parameter
            self.model.tf_alpha *= self.model.tf_gamma
            self.model.saver.writer.add_scalar('tf_alpha', self.model.tf_alpha, e+1)
            # save model, stats and plots
            self.model.saver.save_loss(e+1, epoch_losses)
            if (e+1)%500 == 0:
                self.model.saver.save_expensive(e+1)
            elif (e+1)%100 == 0:
                self.model.saver.save_cheap(e+1)