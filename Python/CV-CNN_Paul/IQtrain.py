import argparse
import json

from models import get_model
from dataloaders import get_dataset, get_dataloader
from utils import utils
from utils.IQtrainer import Trainer


def main(raw_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True,
                        type=str, help='path to the config'
                        'file for the training')
    args = parser.parse_args(raw_args)

    with open(args.config_path) as json_file:
        config = json.load(json_file)
    n_epochs = config['epoch']
    # check and default the config file, keep it compliant
    # with the change in the config file
    config = utils.check_config(config)

    # create the network
    network = get_model(**config['model'])

    # create the trainer, it keeps track of the training state
    trainer = Trainer(network, **config['training'], config=config,
                      device='cuda:0')

    # save the config
    trainer.save_config(config)

    # create datasets and dataloaders
    train_dataset = get_dataset('train', **config["data"])
    val_dataset = get_dataset('val', **config["data"])
    train_loader, val_loader = get_dataloader(train_dataset,
                                              val_dataset,
                                              **config['data_loaders'])

    for epoch in range(n_epochs):
        trainer.train(train_loader)
        trainer.evaluate(val_loader)


if __name__ == "__main__":
    main()
