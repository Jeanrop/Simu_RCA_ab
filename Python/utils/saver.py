import os
import json
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter


def get_saver(trainer, path, resume=False):
    return Saver(trainer, path, resume)


class Saver(object):
    def __init__(self, trainer, path, resume):
        self.train_metrics = {}
        self.val_metrics = {}
        self.trainer = trainer
        self.epoch = 0
        self.best_scores = {}
        if resume:
            self.load_path = path
            path = os.path.join(self.load_path, 'retraining')

        self.set_run_dir(path)
        self.writer = SummaryWriter(self.save_path)

    def save_one_train_epoch(self, trainer, epoch, train_acc, train_loss):
        self.epoch = epoch
        self.trainer = trainer
        try:
            self.train_metrics['epoch'].append(epoch)
        except KeyError:
            self.train_metrics['epoch'] = [epoch]

        try:
            self.train_metrics['train_acc'].append(train_acc)
        except KeyError:
            self.train_metrics['train_acc'] = [train_acc]

        try:
            self.train_metrics['train_loss'].append(train_loss)
        except KeyError:
            self.train_metrics['train_loss'] = [train_loss]

        self.writer.add_scalar('Loss/train', train_loss, self.epoch)
        self.writer.add_scalar('Acc/train', train_acc, self.epoch)
        self.save_train_status()
        print('Train set - Accuracy: {:.4%}'.format(train_acc))
        print('Train set - Average loss: {:.4e}'.format(train_loss))

    def save_one_val_epoch(self, trainer, val_acc, val_loss):
        self.trainer = trainer
        try:
            self.val_metrics['epoch'].append(self.epoch)
        except KeyError:
            self.val_metrics['epoch'] = [self.epoch]

        try:
            self.val_metrics['val_acc'].append(val_acc)
        except KeyError:
            self.val_metrics['val_acc'] = [val_acc]

        try:
            self.val_metrics['val_loss'].append(val_loss)
        except KeyError:
            self.val_metrics['val_loss'] = [val_loss]

        self.save_best_model(val_acc, val_loss)
        self.save_train_status()
        self.writer.add_scalar('Loss/val', val_loss, self.epoch)
        self.writer.add_scalar('Acc/val', val_acc, self.epoch)

        print('Validation set - Average loss: {:.4e}'.format(val_loss))
        print('Validation set - Accuracy: {:.4%}'.format(val_acc))

    def save_train_status(self):
        # save train state (model, optimizer, learning rate etc)
        lr_state = self.trainer.scheduler.state_dict()
        optimizer_state = self.trainer.optimizer.state_dict()
        save_path = os.path.join(self.save_path, "training_state.pth.tar")
        training_state = {'epoch': self.epoch,
                          'state_dict': self.trainer.network.state_dict(),
                          'lr_scheduler': lr_state,
                          'optimizer': optimizer_state,
                          'best_scores': self.best_scores}
        torch.save(training_state, save_path)
        # save train metrics
        save_metric_path = os.path.join(self.save_path, 'train_metrics.json')
        with open(save_metric_path, 'w') as file:
            json.dump(self.train_metrics, file, indent=4)

        # save val metrics
        save_metric_path = os.path.join(self.save_path, 'val_metrics.json')
        with open(save_metric_path, 'w') as file:
            json.dump(self.val_metrics, file, indent=4)

    def save_best_model(self, val_acc, val_loss):
        if "val_acc" not in self.best_scores.keys():
            self.best_scores["val_acc"] = -1

        if "val_loss" not in self.best_scores.keys():
            self.best_scores["val_loss"] = 1

        if val_acc > self.best_scores["val_acc"]:
            self.best_scores["val_acc"] = val_acc
            network_weights = self.trainer.network.state_dict()
            save_path = os.path.join(self.save_path, 'best_val_acc.pth')
            torch.save(network_weights, save_path)

        if val_loss < self.best_scores["val_loss"]:
            self.best_scores["val_loss"] = val_loss
            network_weights = self.trainer.network.state_dict()
            save_path = os.path.join(self.save_path, 'best_val_loss.pth')
            torch.save(network_weights, save_path)

    def set_run_dir(self, path):
        time = datetime.datetime.today()
        log_id = '{}_{}h{}min'.format(time.date(), time.hour, time.minute)
        log_path = os.path.join(path, log_id)
        if not os.path.exists(path):
            os.mkdir(path)
        i = 0
        created = False
        while not created:
            try:
                os.mkdir(log_path)
                created = True
            except OSError:
                i += 1
                log_id = '{}_{}h{}min_{}'.format(time.date(),
                                                 time.hour,
                                                 time.minute,
                                                 i)
                log_path = os.path.join(path, log_id)
        self.save_path = log_path

    def save_config(self, config):
        with open(os.path.join(self.save_path, 'config.json'), 'w') as file:
            json.dump(config, file, indent=4)

    def get_training_state(self):
        state_dict_path = os.path.join(self.load_path,
                                       "training_state.pth.tar")
        training_state_dict = torch.load(state_dict_path)

        load_metric_path = os.path.join(self.load_path, 'val_metrics.json')
        with open(load_metric_path, 'r') as file:
            self.val_metrics = json.load(file)

        load_metric_path = os.path.join(self.load_path, 'train_metrics.json')
        with open(load_metric_path, 'r') as file:
            self.train_metrics = json.load(file)

        self.epoch = training_state_dict['epoch']
        self.best_scores = training_state_dict['best_scores']

        return training_state_dict
