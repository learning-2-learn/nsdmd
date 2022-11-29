import numpy as np
import colorednoise
from scipy.signal import butter
from scipy.signal import cheby1
from scipy.signal import filtfilt

#################### Random Functions

def diff_of_gauss(sig1, sig2, num_points):
    '''
    Calculates the difference of gaussians for the desired standard deviations
    Note that the width is calculated based on sig1 and sig2 and NOT on the sampling rate
    Also note that this always make sig1 have the smaller standard deviation
    
    Parameters
    ------------------
    sig1 : first standard deviation
    sig2 : second standard deviation
    num_points : number of points to include in array
    
    Returns
    ------------------
    diff : difference in Gaussians
    '''
    if sig1 < sig2:
        s1 = sig1
        s2 = sig2
    else:
        s1 = sig2
        s2 = sig1
    
    x = np.arange(-num_points/2+0.5, num_points/2+0.5, 1)
    a = np.exp(-x**2 / (2 * s1**2)) / s1
    b = np.exp(-x**2 / (2 * s2**2)) / s2
    
    diff = (a - b) / ((2*np.pi)**0.5)
    return diff

def cos_dist(a, b):
    """
    Calculates the cosine distance between two arrays

    Parameters
    ----------
    a : first array
    b : second array

    Returns
    -------
    cos_dist : cosine distnace between two arrays
    """
    if np.all(a == 0) or np.all(b == 0):
        cos_dist = 0
    else:
        cos_dist = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_dist


def demean_mat(mat):
    """
    Subtracts the mean out of the last dimension of a 3 dimensional matrix

    Parameters
    ----------
    mat : matrix with three dimensions

    Returns
    -------
    mat_m : de-meaned matrix
    """
    mat_m = mat - np.mean(mat, axis=-1)[:, :, None]
    return mat_m


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
    return br


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
    return ret[n - 1 :] / n


###################### For filtering


def butter_pass_filter(data, cutoff, fs, btype, order=5, axis=-1):
    """
    Butter pass filters a signal with a butter filter

    Parameters
    ----------
    data: the signal to filter
    cutoff: the cutoff frequency
    fs: sampling rate
    btype: either \'high\' or \'low\', determines low pass or high pass filter
    order: the order of the filter
    axis: axis to apply the filter

    Returns
    -------
    Either high or low pass filtered data
    """
    b, a = _butter_pass(cutoff, fs, btype, order=order)
    y = filtfilt(b, a, data, axis=axis)
    return y


def _butter_pass(cutoff, fs, btype, order=5):
    """
    Builds a butter pass filter

    Parameters
    ----------
    cutoff: the cutoff frequency
    fs: sampling rate
    btype: either \'high\' or \'low\', determines low pass or high pass filter
    order: the order of the filter

    Returns
    -------
    Either high or low pass filtered
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

def cheb_pass_filter(data, cutoff, fs, btype, order=5, db_amp=1, axis=-1):
    """
    Butter pass filters a signal with a butter filter

    Parameters
    ----------
    data: the signal to filter
    cutoff: the cutoff frequency
    fs: sampling rate
    btype: either \'high\' or \'low\', determines low pass or high pass filter
    order: the order of the filter
    db_amp: the amplitude of the passband ripple
    axis: axis to apply the filter

    Returns
    -------
    Either high or low pass filtered data
    """
    b, a = _cheb_pass(cutoff, fs, btype, db_amp=db_amp, order=order)
    y = filtfilt(b, a, data, axis=axis)
    return y

def _cheb_pass(cutoff, fs, btype, order=5, db_amp=1):
    """
    Builds a type 1 chebyshev filter

    Parameters
    ----------
    cutoff: the cutoff frequency
    fs: sampling rate
    btype: either \'high\' or \'low\', determines low pass or high pass filter
    order: the order of the filter
    db_amp: the amplitude of the passband ripple

    Returns
    -------
    Either high or low pass filtered
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = cheby1(order, db_amp, normal_cutoff, btype=btype, analog=False)
    return b, a


###################### For simulations


def make_network(freq, t_len, phi_amp, phi_phase, sr=1000, time_mod=0, coupling=0):
    """
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
        It can have shape (length of phi_amp, t_len)
    sr : sampling rate
    time_mod : modulation of time series (multiplies by time series).
        Can be int/float (if freq) or manual series (must be of length t_len)
    coupling : modulation of individual freq. E.g. Freq coupling.
        Can be int/float (if freq) or manual series (must be of length 3*t_len)

    Returns
    -------
    net : the network
    f : the true global modulation
    c : the true coupling
    """
    assert len(phi_amp) == len(phi_phase), "Phi lengths not equal"

    if not (
        type(freq) == np.float64
        or type(freq) == np.int64
        or type(freq) == float
        or type(freq) == int
    ):
        assert len(freq) == 3 * t_len, "Length of frequency term not correct"
        freq = np.insert(freq[:-1], 0, 0)
    else:
        freq = np.ones((3 * t_len) - 1) * freq
        freq = np.insert(freq, 0, 0)

    if type(time_mod) == float or type(time_mod) == int:
        time_mod = 0.5 * (
            1 + np.cos(time_mod * 2 * np.pi * np.arange(0, int(t_len / sr), 1 / sr))
        )

    if type(coupling) == float or type(coupling) == int:
        coupling = 0.5 * (
            1 + np.cos(coupling * 2 * np.pi * np.arange(-t_len, 2 * t_len) * 1 / sr)
        )

    assert len(time_mod) == t_len, "Length of time modulation not correct"
    assert len(coupling) == 3 * t_len, "Length of coupling not correct"

    phase_in = np.cumsum(freq / sr)  # Adds integral into cosine
    t = np.cos(2 * np.pi * phase_in) * coupling
    if len(phi_phase.shape)>1:
        assert phi_phase.shape[1] == t_len
        phi_t = t[np.arange(t_len, 2 * t_len) + phi_phase]
    else:
        phi_t = t[np.arange(t_len, 2 * t_len) + phi_phase[:, None]]

    phi_amp = phi_amp / np.sum(phi_amp**2, axis=0) ** 0.5  # Normalize

    if len(phi_amp.shape) > 1:
        assert phi_amp.shape[1] == t_len
        net = phi_amp * phi_t * np.array(time_mod)[None, :]
    else:
        net = phi_amp[:, None] * phi_t * np.array(time_mod)[None, :]

    if len(phi_phase.shape)>1:
        f = (
            np.sum((phi_amp) ** 2, axis=0) ** 0.5
            * np.array(time_mod)[None, :]
            * coupling[np.arange(t_len, 2 * t_len) + phi_phase]
        )
    else:
        f = (
            np.sum((phi_amp) ** 2, axis=0) ** 0.5
            * np.array(time_mod)[None, :]
            * coupling[np.arange(t_len, 2 * t_len) + phi_phase[:, None]]
        )

    c = coupling[t_len : 2 * t_len]

    return (net, f, c)


def create_decay(t_len, decay_len_start=1, decay_len_end=1):
    """
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
    """
    decay_start = (
        (1 + np.cos(0.5 / decay_len_start * (2 * np.pi) * np.arange(decay_len_start)))
        / 2
    )[::-1]
    decay_end = (
        1 + np.cos(0.5 / decay_len_end * (2 * np.pi) * np.arange(decay_len_end))
    ) / 2
    decay = np.ones(t_len)
    decay[:decay_len_start] = decay_start
    decay[-decay_len_end:] = decay_end
    return decay


def add_noise(x, std, freq_scaling=0):
    """
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
    """
    noise = np.empty(x.shape)
    for i in range(len(x)):
        noise[i] = std * colorednoise.powerlaw_psd_gaussian(-freq_scaling, x.shape[1])
    x += noise
    return x
