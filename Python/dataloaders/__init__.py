from dataloaders import IQrealigned_dataset
from torch.utils.data import DataLoader


def get_dataset(split, data_format, params, dilate=False):
    if data_format == 'IQrealigned':
        return IQrealigned_dataset.ConcatIQrealigned_dataset(split,**params)
    else:
        raise NotImplementedError('available:I/Q')


def get_dataloader(train_dataset, val_dataset, train_params, val_params):
    train_loader = DataLoader(train_dataset, **train_params)
    val_loader = DataLoader(val_dataset, **val_params)
    return train_loader, val_loader
