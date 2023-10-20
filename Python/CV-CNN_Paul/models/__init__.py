from models import ComplexConvnet3d

def get_model(model_type, params=None):
    if model_type == 'IQInception':
                model = ComplexConvnet3d.IQInception_skip(**params)
    else:
        raise NotImplementedError('available: IQInception_skip')

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters of the network : {}'.format(total_params))
    return model
