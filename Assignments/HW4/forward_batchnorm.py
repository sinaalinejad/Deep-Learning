import numpy as np

def forward_batchnorm(Z, gamma, beta, eps, cache_dict, beta_avg, mode):
    """
    Performs the forward propagation through a BatchNorm layer.

    Arguments:
    Z -- input, with shape (num_examples, num_features)
    gamma -- vector, BN layer parameter
    beta -- vector, BN layer parameter
    eps -- scalar, BN layer hyperparameter
    beta_avg -- scalar, beta value to use for moving averages
    mode -- boolean, indicating whether used at 'train' or 'test' time

    Returns:
    out -- output, with shape (num_examples, num_features)
    """

    if mode == 'train':
        # Mean of Z across first dimension
        mu = np.mean(Z, axis=0)

        # Variance of Z across first dimension
        var = np.var(Z, axis=0)

        # Take moving average for cache_dict['mu']
        cache_dict['mu'] = beta_avg * cache_dict['mu'] + (1-beta_avg) * mu

        # Take moving average for cache_dict['var']
        cache_dict['var'] = beta_avg * cache_dict['var'] + (1-beta_avg) * var

    elif mode == 'test':
        # Load moving average of mu
        mu = cache_dict['mu']

        # Load moving average of var
        var = cache_dict['var']

    # Apply z_norm transformation
    Z_norm = (Z.T - mu) / np.sqrt(var + eps)
    Z_norm = Z_norm.T

    # Apply gamma and beta transformation to get Z tiled
    out = gamma * Z_norm + beta

    return out


## testing the function
input = np.random.randint(0, 5, 25)
input = np.resize(input, (5,5))
gamma = np.random.randint(0,5,5)
beta = np.random.randint(0,5,5)
cache_dict = {"mu":np.zeros((5,)), "var":np.zeros((5,))}
output = forward_batchnorm(input, gamma, beta, eps=0.01, cache_dict=cache_dict, beta_avg=0.8, mode="train")
print(output)
