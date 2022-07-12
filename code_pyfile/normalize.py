import numpy as np

# normaliza the data across the specified unit
# z-transfomed

notifyOn = False

def normalize(x, unit = 'dataset'):
    x_shape = x.shape
    if unit == 'dataset':
        xmean = np.mean(x)
        xstd = np.std(x)
        xnorm = (x-xmean)/xstd
    else:

        if len(x_shape) == 3: # raw EEG
            xnorm = normalize3(x, unit)
        else: # power, ISPC, wst
            xnorm = normalize4(x, unit)
    return xnorm


def normalize4(x, unit):
    # unit = 'chan', 'freq' ('scale'), 'trial',
    #  'chanfreq'('chanscale'), 'signal'

    if unit == 'chan':
        x = np.transpose(x, (3,1,2,0))
    elif unit == 'freq' or unit == 'scale':
        x = np.transpose(x, (1,0,2,3))
    elif unit == 'chanfreq' or unit == 'chanscale':
        x = np.transpose(x, (1,3,0,2))
    elif unit == 'signal':
        x = np.transpose(x, (0,1,3,2))

    x_shape = x.shape
    if np.in1d(unit, ['freq', 'chan', 'trial', 'scale'])[0]:
        xmean = np.mean(x, axis = (1,2,3))
        xstd = np.std(x, axis = (1,2,3))
        xmean = xmean.repeat(np.prod(x_shape[1:4])).reshape(x_shape)
        xstd = xstd.repeat(np.prod(x_shape[1:4])).reshape(x_shape)
    elif np.in1d(unit, ['chanfreq', 'chanscale'])[0]:
        xmean = np.mean(x, axis = (2,3))
        xstd = np.std(x, axis = (2,3))
        xmean = xmean.repeat(np.prod(x_shape[2:4])).reshape(x_shape)
        xstd = xstd.repeat(np.prod(x_shape[2:4])).reshape(x_shape)
    elif unit == 'signal':
        xmean = np.mean(x, axis = 3)
        xstd = np.std(x, axis = 3)
        xmean = xmean.repeat(x_shape[3]).reshape(x_shape)
        xstd = xstd.repeat(x_shape[3]).reshape(x_shape)
    else:
        xmean = 0
        xstd = 1
        print('Invalid UNIT. Normalization is OFF')

    x_norm = (x - xmean) / xstd

    if unit == 'chan':
        x_norm = np.transpose(x_norm, (3,1,2,0))
    elif unit == 'freq' or unit == 'scale':
        x_norm = np.transpose(x_norm, (1,0,2,3))
    elif unit == 'chanfreq' or unit == 'chanscale' :
        x_norm = np.transpose(x_norm, [2,0,3,1])
    elif unit == 'signal':
        x_norm = np.transpose(x_norm, [0,1,3,2])

    return x_norm



def normalize3(x, unit = 'signal'): # into 0 ~ 1 in each trial
    x_shape = x.shape

    if unit == 'trial':
        xmean = np.mean(x, axis = (1,2))
        xstd = np.std(x, axis = (1,2))
        xmean = xmean.repeat(x_shape[1]*x_shape[2])
        xstd = xstd.repeat(x_shape[1]*x_shape[2])
        xmean = xmean.reshape(x_shape)
        xstd = xstd.reshape(x_shape)

    elif unit == 'chan':
        xmean = np.mean(x, axis = (0,2))
        xstd = np.std(x, axis = (0,2))

        xstd = xstd.repeat(x_shape[2])
        xstd = xstd.reshape(x_shape[1:3])
        xstd = np.tile(xstd, (x_shape[0],1,1))

        xmean = xmean.repeat(x_shape[2])
        xmean = xmean.reshape(x_shape[1:3])
        xmean = np.tile(xmean, (x_shape[0],1,1))

    elif unit == 'signal':
        xmean = np.mean(x, axis = 2)
        xstd = np.std(x, axis = 2)
        xmean = xmean.repeat(x_shape[2]).reshape(x_shape)
        xstd = xstd.repeat(x_shape[2]).reshape(x_shape)

    else:
        xmean = 0
        xstd = 1
        print('Invalid UNIT. Normalization is OFF')

    x_norm = (x - xmean) / xstd
    return x_norm

