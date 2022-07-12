import numpy as np

notifyOn = False

def normalize(x, unit = 'dataset'):
    x_shape = x.shape
    if unit == 'dataset':
        xmin = np.amin(x)
        xmax = np.amax(x)
        xnorm = (x-xmin)/(xmax-xmin)
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
        xmin = np.amin(x, axis = (1,2,3))
        xmax = np.amax(x, axis = (1,2,3))
        xmin = xmin.repeat(np.prod(x_shape[1:4])).reshape(x_shape)
        xmax = xmax.repeat(np.prod(x_shape[1:4])).reshape(x_shape)
    elif np.in1d(unit, ['chanfreq', 'chanscale'])[0]:
        xmin = np.amin(x, axis = (2,3))
        xmax = np.amax(x, axis = (2,3))
        xmin = xmin.repeat(np.prod(x_shape[2:4])).reshape(x_shape)
        xmax = xmax.repeat(np.prod(x_shape[2:4])).reshape(x_shape)
    elif unit == 'signal':
        xmin = np.amin(x, axis = 3)
        xmax = np.amax(x, axis = 3)
        xmin = xmin.repeat(x_shape[3]).reshape(x_shape)
        xmax = xmax.repeat(x_shape[3]).reshape(x_shape)
    else:
        xmin = 0
        xmax = 1
        print('Invalid UNIT. Normalization is OFF')

    x = (x - xmin) / (xmax - xmin)

    if unit == 'chan':
        x = np.transpose(x, (3,1,2,0))
    elif unit == 'freq' or unit == 'scale':
        x = np.transpose(x, (1,0,2,3))
    elif unit == 'chanfreq' or unit == 'chanscale' :
        x = np.transpose(x, [2,0,3,1])
    elif unit == 'signal':
        x = np.transpose(x, [0,1,3,2])

    return x



def normalize3(x, unit = 'signal'): # into 0 ~ 1 in each trial
    x_shape = x.shape

    if unit == 'trial':
        xmin = np.amin(x, axis = (1,2))
        xmax = np.amax(x, axis = (1,2))
        xmin = xmin.repeat(x_shape[1]*x_shape[2])
        xmax = xmax.repeat(x_shape[1]*x_shape[2])
        xmin = xmin.reshape(x_shape)
        xmax = xmax.reshape(x_shape)

    elif unit == 'chan':
        xmin = np.amin(x, axis = (0,2))
        xmax = np.amax(x, axis = (0,2))

        xmin = xmin.repeat(x_shape[2])
        xmin = xmin.reshape(x_shape[1:3])
        xmin = np.tile(xmin, (x_shape[0],1,1))

        xmax = xmax.repeat(x_shape[2])
        xmax = xmax.reshape(x_shape[1:3])
        xmax = np.tile(xmax, (x_shape[0],1,1))

    elif unit == 'signal':
        xmin = np.amin(x, axis = 2)
        xmax = np.amax(x, axis = 2)
        xmin = xmin.repeat(x_shape[2]).reshape(x_shape)
        xmax = xmax.repeat(x_shape[2]).reshape(x_shape)

    else:
        xmin = 0
        xmax = 1
        print('Invalid UNIT. Normalization is OFF')

    x = (x - xmin) / (xmax - xmin)
    return x

