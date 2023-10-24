from torch import optim


def get_scheduler(optimizer, scheduler_type, params):
    if scheduler_type == "multistep":
        return optim.lr_scheduler.MultiStepLR(optimizer, **params)
    elif scheduler_type == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer,**params)
    elif scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    else:
        raise NotImplementedError('available: multistep, exponential, plateau')
