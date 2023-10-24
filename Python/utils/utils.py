def check_config(config):
    err = 'training parameters unspecified'
    assert 'training' in config.keys(), err

    err = "loss unspecified"
    assert 'loss' in config['training'].keys(), err

    err = "optimizer unspecified"
    assert 'optimizer' in config['training'].keys(), err

    err = "scheduler unspecified"
    assert 'scheduler' in config['training'].keys(), err

    err = "saver unspecified"
    assert 'saver' in config['training'].keys(), err

    err = 'data parameters unspecified'
    assert 'data' in config.keys(), err

    err = 'data_loaders parameters unspecified'
    assert 'data_loaders' in config.keys(), err

    # defaulted undersampling and pass it to the model and dataset class
    if "undersampling" not in config.keys():
        config["undersampling"] = None
        print("undersampling is defaulted to None")
    #config["model"]["params"]["undersampling"] = config["undersampling"]
    #config["data"]["params"]["undersampling"] = config["undersampling"]

    return config
