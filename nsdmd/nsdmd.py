import numpy as np
from nsdmd import optdmd
from nsdmd import utils

def opt_dmd_win(x, t, w_len, stride, rank, initial_freq_guess=None):
    '''
    Computes OPT-DMD for windows defined by the length and stride
    
    Parameters
    ----------
    x : 2 dimensional data matrix. 1st dimension is spatial, 2nd is time
    t : times associated with each snapshot in x
    w_len : window length of each run of opt-dmd
    stride : step of window
    rank : rank to run opt-dmd
    initial_freq_guess : guess for first window in frequency
    
    Returns
    -------
    freqs : frequencies of modes with shape (number of windows, rank)
    phis : complex spatial modes with shape (number of windows, number of channels, rank)
    windows : exact windows used, for testing purposes
    '''
    windows = np.array([np.arange(i,i+w_len) for i in np.arange(0, x.shape[-1]-w_len+1, stride)])
    freqs = np.empty((len(windows), rank))
    phis = np.empty((len(windows), len(x), rank), dtype=complex)
    
    for i, window in enumerate(windows):
        x_temp = x[:,window]
        t_temp = t[window]
        t_temp -= t_temp[0] #Forces the starting time to be 0
    
        if (i==0):
            if initial_freq_guess is None:
                guess = None
            else:
                assert len(initial_freq_guess)==rank, "Number of frequencies guessed isnt equal to the rank"
                guess = 0.0 + 1j * 2 * np.pi * initial_freq_guess
        else:
            guess = 0.0 + 1j * 2 * np.pi * freqs[i-1] #Setting real part to be 0
            
        dmd = optdmd.OptDMD(x_temp, t_temp, rank)
        dmd.fit(verbose=False, eigs_guess=guess)
        
        freqs[i] = np.array(dmd.eigs).imag / 2. / np.pi
        phis[i] = dmd.modes

    return(freqs, phis, windows)


def get_soln(freqs, phis, t, offsets):
    '''
    Gets the full solution from frequencies and phis
    
    Parameters
    ----------
    freqs : the frequencies with shape (number of windows x number of modes)
    phis : the phis with shape (number of windows x number of channels x number of modes)
    t : the time snapshots
    offsets : the temporal offset of each window
    
    Returns
    -------
    soln : the extended solutions with shape (number of windows x number of modes x number of channels x time)
    '''
    soln = np.empty((freqs.shape[0], freqs.shape[1], phis.shape[1], len(t)))
    for r in range(freqs.shape[-1]):
        for i in range(len(freqs)):
            temp = np.exp(2*np.pi*1j*((t-offsets[i]) * freqs[i,r]))
            temp2 = phis[i,:,r][:,None]*temp
            soln[i,r] = temp2.real
    
    return(soln)


def get_t_delay_from_soln(freqs, phis, t, t_step, offsets):
    '''
    Predicts temporal delays between channels from the solutions
    
    Parameters
    ----------
    freqs : the frequencies with shape (number of windows x number of modes)
    phis : the phis with shape (number of windows x number of channels x number of modes)
    t : the time snapshots
    t_step : the temporal difference between snapshots (or 1 over sampling rate)
    offsets : the temporal offset of each window
    
    Returns
    -------
    t_delay : the predicted time delays with shape (number of windows x number of modes x number of channels)
    '''
    t_delay = np.empty((freqs.shape[0], freqs.shape[1], phis.shape[1]), dtype=int)
    for r in range(freqs.shape[-1]):
        for i in range(len(freqs)):
            temp = np.exp(2*np.pi*1j*((t-offsets[i]) * freqs[i,r]))
            temp2 = phis[i,:,r][:,None]*temp
            temp3 = np.round(np.angle(temp2[:,0]) / (2*np.pi*freqs[i,r]) / t_step)
            t_delay[i,r] = np.array([int(ch) for ch in temp3])
    
    return(t_delay)


def group_by_similarity(freqs, phis, thresh_freq=0.2, thresh_phi_amp=0.95):
    '''
    Groups all modes based on frequencies and phi amplitudes.
    Note : does NOT look at phi phases (TODO)
    Note : modes are expected to be in order of pairs, where each pair represents the positive and negative frequency.
        This function cannot currently handle non-pairs
    Note : currently cannot control threshold of polarity
    
    Parameters
    ----------
    freqs : all frequencies with shape (number of windows x number of modes)
    phis : all phis with shape (number of windows x number of recordings x number of modes)
    thresh_freq : frequency threshold. Any pair of frequencies with a smaller difference is 'similar'
    thresh_phi_amp : phi_amp threshold. Any pair with larger value is 'similar'
        value is computed by cosine distance metric
        
    Returns
    -------
    groups : list of groups with length (number of modes)
        even modes (0,2,4,...) contain groups (lists) of consecutive similar solutions
        odd modes (1,3,5,...) contain a list of solutions (of paired modes) that are significantly different than counterparts
        
    '''
    groups = []
    for i in range(0,freqs.shape[1]):
        if i%2==0:
            groups.append(_group_by_freq_phi(freqs[:,i], np.abs(phis[:,:,i]), \
                                             thresh_freq=thresh_freq, thresh_phi_amp=thresh_phi_amp))
        else:
            temp1 = _group_by_polarity(freqs[:,i-1], freqs[:,i], 'freq_pol')
            temp2 = _group_by_polarity(np.abs(phis[:,:,i-1]), np.abs(phis[:,:,i]), 'phi_amp_pol')
            groups.append([[g] for g in np.unique(np.hstack((temp1,temp2)))])
    return(groups)




###################Implicit Functions

def _group_by_polarity(x1, x2, dtype, thresh=None):
    '''
    forms groups were the com (comparisons) are comparing objects with opposite polarity (e.g. +/- freq)
    '''
    if dtype=='freq_pol':
        com = np.abs(x1 + x2)
        if thresh==None:
            thresh = 0.05
    elif dtype=='phi_amp_pol':
        com = np.empty((len(x1)))
        for i in range(len(x1)):
            com[i] = 1 - utils.cos_dist(x1[i], x2[i])
        if thresh==None:
            thresh = 0.02
    elif dtype=='phi_angle_pol':
        print('todo')
        return([])
    return(np.argwhere(com>thresh)[:,0])

def _group_by_freq_phi(freq, phi_amp, thresh_freq=0.2, thresh_phi_amp=0.95):
    '''
    forms groups where the com (comparisons) are comparing consecutive things
    '''
    groups = []
    
    group_num = -1
    for i in range(len(freq)-1):
        if group_num == -1:
            if np.abs(freq[i+1] - freq[i]) < thresh_freq and utils.cos_dist(phi_amp[i], phi_amp[i+1]) > thresh_phi_amp:
                groups.append([i, i+1])
                in_group = True
            else:
                groups.append([i])
                in_group = False
            group_num += 1
        else:
            if np.abs(freq[i+1] - freq[i]) < thresh_freq and utils.cos_dist(phi_amp[i], phi_amp[i+1]) > thresh_phi_amp:
                if in_group:
                    groups[group_num].append(i+1)
                else:
                    groups.append([i, i+1])
                    group_num += 1
                    in_group = True
            else:
                if in_group:
                    in_group = False
                else:
                    groups.append([i])
                    group_num += 1
    if not in_group:
        groups.append([i+1])
    return(groups)