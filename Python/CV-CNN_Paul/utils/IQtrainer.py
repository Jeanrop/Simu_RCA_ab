from time import time
from datetime import timedelta
import torch
from torch import nn
from utils.loss import get_loss
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler
from utils.saver import get_saver


class Trainer():
    def __init__(self, network, loss, optimizer,
                 scheduler, config, saver, device):
        self.device = device
        self.network = network.to(device)
        self.criterion = get_loss(**loss)
        self.optimizer = get_optimizer(self.network, **optimizer)
        self.scheduler = get_scheduler(self.optimizer, **scheduler)
        self.scheduler_type =  scheduler["scheduler_type"]
        self.config = config
        self.saver = get_saver(self, **saver)
        self.epoch = 0

    def train(self, train_loader):
        """
        Argument : training loader, network, optimizer,
        criterion, LR scheduler, parameters dictionnary, graphs matrix to save
        losses.
        Output : none

        Training process for one epoch : for every batch in the data loader,
        infer, compute loss, backward optimization, some display
        """
        start_time = time()
        self.network.train()
        self.epoch += 1
        lr = self.optimizer.param_groups[0]['lr']
        print('EPOCH {} - LR = {:.0E}'.format(self.epoch,
                                              lr))

        train_loss = 0
        acc = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs_r = batch['x_r'].to(self.device)
            inputs_i = batch['x_i'].to(self.device)
            targets_r = batch['y_r'].to(self.device)
            targets_i = batch['y_i'].to(self.device)

            self.optimizer.zero_grad()
            outputs_r, outputs_i = self.network(inputs_r, inputs_i)
            loss_r = self.criterion(outputs_r, targets_r)
            loss_i = self.criterion(outputs_i, targets_i)
            loss = loss_r+loss_i

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            with torch.no_grad():

                err = torch.abs((outputs_r - targets_r)/targets_r) + \
                                torch.abs((outputs_i - targets_i)/targets_i)
                err[err==float("Inf")]=0
                acc += err.float().mean()/2



            if batch_idx % 25 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]'
                      ' | Batch Loss: {:.3e}'.format(self.epoch,
                                                     batch_idx * len(inputs_r),
                                                     len(train_loader.dataset),
                                                     100. * batch_idx /
                                                     len(train_loader),
                                                     loss.item()))
        acc /= len(train_loader)
        acc = 1.0 - acc
        train_loss /= len(train_loader)
        epoch_summary = {'epoch': self.epoch,
                         'train_acc': acc.item(),
                         'train_loss': train_loss}
        self.saver.save_one_train_epoch(self, **epoch_summary)
        duration = timedelta(seconds=time()-start_time)

        if self.scheduler_type=="plateau":
            self.scheduler.step(train_loss)
        else:
            self.scheduler.step()

        print('Time for training : {}\n'.format(duration))

    def evaluate(self, val_loader):
        """
        Argument : training loader, current epoch number, network, optimizer,
        criterion, LR scheduler, parameters dictionnary, graphs matrix
        to save losses
        Output : none

        Evaluation process for one epoch : for every batch in the validation
        loader, infer, compute loss, save it, some display
        """
        self.network.eval()
        eval_loss = 0
        acc = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                inputs_r = batch['x_r'].to(self.device)
                inputs_i = batch['x_i'].to(self.device)
                targets_r = batch['y_r'].to(self.device)
                targets_i = batch['y_i'].to(self.device)

                outputs_r, outputs_i = self.network(inputs_r, inputs_i)
                eval_loss += self.criterion(outputs_r, targets_r).item() + \
                                    self.criterion(outputs_i, targets_i).item()

                err = torch.abs((outputs_r - targets_r)/targets_r) + \
                            torch.abs((outputs_i - targets_i)/targets_i)
                err[err==float("Inf")]=0
                acc += err.float().mean()/2


        eval_loss /= len(val_loader)
        acc /= len(val_loader)
        acc = 1.0 - acc
        epoch_summary = { 'val_acc': acc.item(),
                          'val_loss': eval_loss}
        self.saver.save_one_val_epoch(self, **epoch_summary)

    def save_config(self, config):
        self.saver.save_config(config)

    def resume_training(self):
        training_sate_dict = self.saver.get_training_state()
        self.network.load_state_dict(training_sate_dict['state_dict'])
        self.scheduler.load_state_dict(training_sate_dict['lr_scheduler'])
        self.optimizer.load_state_dict(training_sate_dict['optimizer'])
        self.epoch = training_sate_dict['epoch']
