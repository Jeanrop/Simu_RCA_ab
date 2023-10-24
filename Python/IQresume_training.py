import os
import argparse
import json

from models import get_model
from dataloaders import get_dataset, get_dataloader
from utils import utils
from utils.IQtrainer import Trainer


def main(raw_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_dir', default=None,
                        type=str, help='path to the config'
                        'file for the training')
    args = parser.parse_args(raw_args)

    old_config_path = os.path.join(args.resume_dir, 'config.json')

    with open(old_config_path) as json_file:
        config = json.load(json_file)

    # check and default the config file, keep it compliant
    # with the change in the config file
    config = utils.check_config(config)

    n_epochs = config['epoch']
    config["training"]["saver"] = {
        "path": args.resume_dir,
        "resume": True
    }
    # create the network
    network = get_model(**config['model'])

    # create the trainer, it keeps track of the training state
    trainer = Trainer(network, **config['training'], config=config,
                      device='cuda:0')
    trainer.resume_training()
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
