import torch.optim as optim


def adam(model_params, config):
    return optim.Adam(model_params,
                      lr=config['lr'],
                      betas=(config['beta1'],
                             config['beta2']),
                      weight_decay=config['weight_decay'])


def sgd(model_params, config):
    return optim.SGD(model_params,
                     lr=config['lr'],
                     momentum=config['momentum'],
                     weight_decay=config['weight_decay'])

factory = {
    'adam': adam,
    'sgd': sgd
}
