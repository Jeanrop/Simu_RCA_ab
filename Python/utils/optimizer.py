from torch import optim


def get_optimizer(network, optimizer_type, initial_lr, weight_decay):
    if optimizer_type == 'adam':
        return optim.Adam(network.parameters(), lr=initial_lr,
                          weight_decay=weight_decay)
    elif optimizer_type == 'adamW':
        return optim.AdamW(network.parameters(), lr=initial_lr,
                          weight_decay=weight_decay)
    else:
        raise NotImplementedError('available: adam, adamW')
