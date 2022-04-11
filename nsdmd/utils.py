import numpy as np
import colorednoise

#################### Random Functions

def cos_dist(a,b):
    '''
    Calculates the cosine distance between two arrays
    
    Parameters
    ----------
    a : first array
    b : second array
    
    Returns
    -------
    cos_dist : cosine distnace between two arrays
    '''
    cos_dist = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return(cos_dist)

def demean_mat(mat):
    '''
    Subtracts the mean out of the last dimension of a 3 dimensional matrix
    
    Parameters
    ----------
    mat : matrix with three dimensions
    
    Returns
    -------
    mat_m : de-meaned matrix
    '''
    mat_m = mat - np.mean(mat, axis=-1)[:,:,None]
    return(mat_m)

def moving_average_dim(ar, size, dim):
    """
    Calculates the moving average along dimension
    
    Parameters
    --------------
    ar: array to be averaged
    size: size of window
    dim: dimension to calculate over
    
    Returns
    --------------
    Moving average along dim
    """
    br = np.apply_along_axis(_moving_average, dim, ar, size)
    return(br)

def _moving_average(a, n):
    """
    Calculates the moving average of an array.
    Function taken from Jaime here:
    https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    
    Parameters
    --------------
    a: array to be averaged
    n: size of window
    
    Returns
    --------------
    Moving average
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

###################### For simulations

def make_network(freq, t_len, phi_amp, phi_phase, t_step=0.001, time_mod=0, coupling=0):
    '''
    Generates a network that oscillates at some frequency with the desired parameters
    
    Parameters
    ----------
    freq : frequency network oscillates at
        Can be int/float (if constant freq) or manual series of frequencies at every point
        E.g. cos(2 * pi * f(t) * t)
        If manual series, must be of length 3*t_len
    t_len : time length of network
    phi_amp : amplitude of channels
    phi_phase : phase of channels in number of time steps (must be length of phi_amp)
    t_step : time step
    time_mod : modulation of time series (multiplies by time series).
        Can be int/float (if freq) or manual series (must be of length t_len)
    coupling : modulation of individual freq. E.g. Freq coupling.
        Can be int/float (if freq) or manual series (must be of length 3*t_len)
        
    Returns
    -------
    net : the network
    f : the true global modulation
    c : the true coupling
    '''
    assert len(phi_amp)==len(phi_phase), 'Phi lengths not equal'
    
    if not (type(freq)==np.float64 or type(freq)==np.int64 or type(freq)==float or type(freq)==int):
        assert len(freq)==3*t_len, 'Length of frequency term not correct'
    
    if type(time_mod)==float or type(time_mod)==int:
        time_mod = 0.5*(1 + np.cos(time_mod * 2*np.pi * np.arange(0,t_len*t_step,t_step)))
        
    if type(coupling)==float or type(coupling)==int:
        coupling = 0.5*(1 + np.cos(coupling * 2*np.pi * np.arange(-t_len,2*t_len)*t_step))
        
    assert len(time_mod)==t_len, 'Length of time modulation not correct'
    assert len(coupling)==3*t_len, 'Length of coupling not correct'
    
    freq_term = np.cos(2*np.pi*freq*np.arange(-t_len,2*t_len)*t_step)
    t = freq_term * coupling
    phi_t = t[np.arange(t_len, 2*t_len) + phi_phase[:,None]]
    
    phi_amp = phi_amp / np.sum(phi_amp**2, axis=0)**0.5 #Normalize
    
    if len(phi_amp.shape) > 1:
        assert phi_amp.shape[1]==t_len
        net = phi_amp * phi_t * np.array(time_mod)[None,:]
    else:
        net = phi_amp[:,None] * phi_t * np.array(time_mod)[None,:]

    f = np.sum((phi_amp)**2, axis=0)**0.5 * np.array(time_mod)[None,:]*\
        coupling[np.arange(t_len, 2*t_len) + phi_phase[:,None]]
    
    c = coupling[t_len:2*t_len]
    
    return(net, f, c)

def create_decay(t_len, decay_len_start=1, decay_len_end=1):
    '''
    Creates a time series with cosine like decay
    Values of 1 indicate no decay at the start or end of array
    
    Parameters
    ----------
    t_len : length of array
    decay_len_start : length of decay at the beginning of the array
    decay_len_end : length of decay at the end of the array
    
    Returns
    -------
    decay : time series with desired decay
    '''
    decay_start = ((1+np.cos(0.5 / decay_len_start * (2*np.pi)*np.arange(decay_len_start)))/2)[::-1]
    decay_end   = ((1+np.cos(0.5 / decay_len_end   * (2*np.pi)*np.arange(decay_len_end)))  /2)
    decay = np.ones(t_len)
    decay[:decay_len_start] = decay_start
    decay[-decay_len_end:]  = decay_end
    return(decay)

def add_noise(x, std, freq_scaling=0):
    '''
    Adds colored noise to each array in x
    Assumes x is 2 dimensional
    
    Parameters
    ----------
    x : 2 dimensional matrix
    std : amplitude of noise
    freq_scaling : power of scaling. E.g. 0 is white noise, -2 is brown noise
    
    Returns
    -------
    x : original matrix with different noise for each array
    '''
    noise = np.empty(x.shape)
    for i in range(len(x)):
        noise[i] = std * colorednoise.powerlaw_psd_gaussian(-freq_scaling, x.shape[1])
    x += noise
    return x