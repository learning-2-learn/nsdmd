import numpy as np
from nsdmd import optdmd
from nsdmd import utils

################## OPT-DMD

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

##################### Processing steps

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

###################### Exact Method

def exact_Bf(x, soln):
    '''
    Gets B and f approximate with the exact method approach
    
    Parameters
    ----------
    x : the data with shape (number of channels, time)
    soln : solutions with shape (number of modes, number of channels, time)
    
    Returns
    -------
    B : B matrix from exact method
    f : approximate f from exact method
    '''
    top = np.sum(soln*x[None,:,:], axis=1)
    bot = np.sum(soln**2, axis=1)
    f = top / bot
    B = np.sum(soln[:,None] * soln[None,:], axis=2) / bot[:,None]
    return(B, f)


def exact_f_from_Bf(B, f, variance_thresh=0.01):
    '''
    Gets the f_hat from the estimated f and B matrix in the exact method
    
    Parameters
    ----------
    B : B matrix from exact method
    f : approximate f from exact method
    
    Returns
    -------
    f_hat : f_hat from the exact method
    '''
    f_hat = np.empty(f.shape)
    for t in range(f.shape[1]):
        f_sub = f[:,t]
        B_sub = B[:,:,t].T
        u,s,vh = np.linalg.svd(B_sub)
        idx = s**2 / (s@s) > variance_thresh
        B_inv = vh.T[:,idx] @ np.diag(1./s[idx]) @ u.T[idx]
        f_hat[:,t] = B_inv @ f_sub
    return(f_hat)

def exact_f_greedy(B, f, soln, x, N, verbose=True):
    '''
    Computes f with a greedy forward elimination algorithm for every round of modes removed
    Uses the exact method
    
    Parameters
    ----------
    B : B matrix from exact method
    f : estimated f from exact method
    soln : solutions that are indexed the same as B and f
    x : original data matrix
    N : amount of averaging on either side of point interested in
    
    Returns
    -------
    idx_all : list of indicies, corresponding to each run of the greedy algorithm
    total_error : error for each run of the greedy algorithm
    '''
    idx_all = []

    f_hat = exact_f_from_Bf(B,f)
    f_hat[:,N:-N] = utils.moving_average_dim(f_hat,2*N+1,1)
    f_hat[f_hat<0] = 0
    x_rec = get_reconstruction(soln, f_hat)
    total_error = [get_reconstruction_error(x, x_rec)]

    for i in range(f.shape[0]):
        print(str(i) + '/' + str(f.shape[0]))
        if i==0:
            idx = np.arange(f.shape[0])
        else:
            idx = idx[np.arange(len(idx))!=np.argmax(error)]
        idx_all.append(idx)

        if(len(idx)>1):
            error = np.empty(len(idx))
            for j, r in enumerate(idx):
                idx_sub = idx[idx!=r]
                f_sub = f[idx_sub]
                B_sub = np.array([b[idx_sub] for b in B[idx_sub]])
                f_hat = exact_f_from_Bf(B_sub,f_sub)
                f_hat[:,N:-N] = utils.moving_average_dim(f_hat,2*N+1,1)
                f_hat[f_hat<0] = 0
                x_rec = get_reconstruction(soln[idx_sub], f_hat)
                error[j] = get_reconstruction_error(x, x_rec)
            total_error.append(np.max(error))
    return(idx_all, total_error)

###################### Gradient Descent

def grad_f_init(x, soln, beta, N):
    '''
    Finds the initial guess for f based on the gradient descent method
    Note : assumes beta is constant, unlike in paper (TODO)
    
    Parameters
    ----------
    x : original data matrix with shape (number of channels, time)
    soln : solutions with shape (number of modes, number of channels, time)
    beta : number indicating strength of temporal smoothing
    N : number of timepoints to smooth over
    
    Returns
    -------
    f_init : intial guess for f
    '''
    f_init = np.empty((soln.shape[0], soln.shape[-1]))
    for r in range(soln.shape[0]):
        for t in range(soln.shape[2]):
            top = (soln[r,:,t] @ x[:,t])
            bot = ((soln[r,:,t] @ soln[r,:,t])) + beta*N
            f_init[r,t] = top / bot
    return(f_init)

def grad_f_grad_loss(f, x, soln, alpha, beta, N):
    '''
    Finds the gradient of the loss function in the gradient descent method
    Note : assumes beta is constant, unlike in paper (TODO)
    Note : also doesn't do anything at the edge, maybe should implement reflection or something?? (TODO)
    
    Parameters
    ----------
    f : current guess of f with shape (number of modes, time)
    x : original data with shape (number of channels, time)
    soln : solutions with shape (number of modes, number of channels, time)
    alpha : number indicating strength of l1 regularization
    beta : number indicating strength of temporal smoothing
    N : number of timepoints to smooth over
    
    Returns
    -------
    dLdf : gradient of loss function
    '''
    Y = np.matmul(np.transpose(soln, [2,1,0]), np.transpose(f)[:,:,None])[:,:,0] #time, chan
    Y2 = Y - x.T
    l2_term = np.matmul(np.transpose(soln, [2,0,1]), Y2[:,:,None])[:,:,0].T
    
    alpha_term = np.ones((f.shape)) * alpha
    alpha_term[f<0] = -alpha_term[f<0]
    
    beta_term = np.zeros((f.shape))
    for i in range(1,N+1):
        beta_term[:,:-i] = beta_term[:,:-i] - beta*(f[:,i:]-f[:,:-i])
        beta_term[:,i:] =  beta_term[:,i:]  + beta*(f[:,i:]-f[:,:-i])
    
    dLdf = l2_term + alpha_term + beta_term
    return(dLdf)

def grad_f(x, soln, alpha, beta, N, lr, maxiter):
    '''
    Performs gradient descent to approximate f
    Note : assumes beta is constant, unlike in paper (TODO)
    
    Parameters
    ----------
    x : original data matrix with shape (number of channels, time)
    soln : solutions with shape (number of modes, number of channels, time)
    alpha : number indicating strength of l1 regularization
    beta : number indicating strength of temporal smoothing
    N : number of timepoints to smooth over
    lr : learning rate
    maxiter : total number of iterations
    
    Returns
    -------
    f : approximation of global modulation f
    '''
    f = grad_f_init(x, soln, beta, N)
    f[f<0]=0
    
    for i in range(maxiter):
        dLdf = grad_f_grad_loss(f, x, soln, alpha, beta, N)
        f = f - lr*dLdf
        f[f<0]=0
        f[:,N:-N] = utils.moving_average_dim(f,2*N+1,1)
        f[:,:N] = np.mean(f[:,:N],axis=1)[:,None]
        f[:,-N:] = np.mean(f[:,-N:],axis=1)[:,None]
            
    return(f)

def grad_f_amp(f, soln, x):
    '''
    Fixes overall amplitude of f hat in the gradient descent method
    
    Parameters
    ----------
    f : computed solution for f with shape (number of modes, time)
    soln : solutions with shape (number of modes, number of channels, time)
    x : data matrix with shape (number of channels, time)
    
    Returns
    -------
    f_hat : global modulation f with fixed amplitude
    '''
    norm,_,_,_ = np.linalg.lstsq((f[:,None,:] * soln).reshape((len(f),-1)).T, x.reshape((-1)),rcond=None)
    f_hat = f * norm[:,None]
    return(f_hat)

###################### Reconstruction

def get_reconstruction(soln, f):
    '''
    Reconstructs x from the S and f
    
    Parameters
    ----------
    soln : solutions with shape (number of modes, number of channels, time)
    f : global modulation f with shape (number of modes, time)
    
    Returns
    -------
    x_rec : reconstructed data matrix
    '''
    x_rec = np.zeros((soln.shape[1], soln.shape[2]))
    for r in range(len(soln)):
        temp = soln[r,:,:] * f[r][None,:]
        x_rec = x_rec + temp
    return(x_rec)

def get_reconstruction_error(x_true, x_rec):
    '''
    Calculates the reconstruction error from the original and reconstructed data matricies
    
    Parameters
    ----------
    x_true : true data matrix
    x_rec : reconstructed data matrix
    
    Returns
    -------
    error : error computed with cosine distance metric
    '''
    error = utils.cos_dist(x_rec.reshape((-1)), x_true.reshape((-1)))
    return(error)





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