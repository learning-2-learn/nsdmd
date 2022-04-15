import numpy as np
from nsdmd import optdmd
from nsdmd import utils

################## Classes

class NSDMD():
    def __init__(self, opt_win=500, opt_stride=100, opt_rank=20, \
                 sim_thresh_freq=0.2, sim_thresh_phi_amp=0.95, \
                 exact_var_thresh=0.01, exact_N=20,\
                 grad_alpha=0.1, grad_beta=0.1, grad_N=20, grad_lr=0.01, grad_maxiter=100,\
                 grad_fit_coupling=False, verbose=False):
        self.opt_win = opt_win
        self.opt_stride = opt_stride
        self.opt_rank = opt_rank
        self.sim_thresh_freq = sim_thresh_freq
        self.sim_thresh_phi_amp = sim_thresh_phi_amp
        self.exact_var_thresh = exact_var_thresh
        self.exact_N = exact_N
        self.grad_alpha = grad_alpha
        self.grad_beta = grad_beta
        self.grad_N = grad_N
        self.grad_lr = grad_lr
        self.grad_maxiter = grad_maxiter
        self.grad_fit_coupling = grad_fit_coupling
        self.verbose = verbose
        
    def fit_opt(self, x, t, initial_freq_guess=None):
        f, p, w = opt_dmd_win(x, t, self.opt_win, self.opt_stride, self.opt_rank, initial_freq_guess)
        self.freqs_ = f
        self.phis_ = p
        self.windows_ = w
        self.offsets_ = t[self.windows_[:,0]][:,None]*np.ones((self.freqs_.shape[1]), dtype=int)[None,:]
        return self
    
    def set_opt_values(self, freqs, phis, windows, offsets):
        self.freqs_ = freqs
        self.phis_ = phis
        self.windows_ = windows
        self.offsets_ = offsets
        return self

    def fit_reduction(self, x, t):
        group_idx = group_by_similarity(self.freqs_, self.phis_, self.sim_thresh_freq, self.sim_thresh_phi_amp)
        
        idx_init = get_red_init(group_idx)
        idx = tuple(idx_init.T)
        soln = get_soln(self.freqs_[idx], self.phis_[idx], t, self.offsets_[idx])
        
        B,f = exact_Bf(x, soln)
        idxs, self.errors_ = exact_f_greedy(B,f,soln,x,self.exact_N, self.exact_var_thresh, self.verbose)
        self.idx_red_ = [idx_init[idx] for idx in idxs]
        
        return self
    
    def fit_f(self, x, t, t_step, idx_num):
        self.idx_hat_ = self.idx_red_[idx_num]
        idx = tuple(self.idx_hat_.T)
        self.freq_hat_ = self.freqs_[idx]
        self.phi_hat_ = self.phis_[idx]
        self.offset_hat_ = self.offsets_[idx]
        self.delay_hat_ = get_t_delay_from_soln(self.freq_hat_, self.phi_hat_, t, t_step, self.offset_hat_)
        
        soln = get_soln(self.freq_hat_, self.phi_hat_, t, self.offset_hat_)
        f_hat = grad_f(x, soln, self.grad_alpha, self.grad_beta, \
                       self.grad_N, self.grad_lr, self.grad_maxiter, \
                       self.grad_fit_coupling, self.delay_hat_)
        self.f_hat_ = grad_f_amp(f_hat, soln, x)
        return self
    
    def transform(self, x, t):
        soln = get_soln(self.freq_hat_, self.phi_hat_, t, self.offset_hat_)
        x_rec = get_reconstruction(soln, self.f_hat_)
        return x_rec


################## OPT-DMD

def opt_dmd_win(x, t, w_len, stride, rank, initial_freq_guess=None):
    '''
    Computes OPT-DMD for windows defined by the length and stride
    Note : currently works with equally spaced time data
    
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
    phis : complex spatial modes with shape (number of windows, rank, number of channels)
    windows : exact windows used, for testing purposes
    '''
    windows = np.array([np.arange(i,i+w_len) for i in np.arange(0, x.shape[-1]-w_len+1, stride)])
    freqs = np.empty((len(windows), rank))
    phis = np.empty((len(windows), rank, len(x)), dtype=complex)
    
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
        phis[i] = dmd.modes.T

    return(freqs, phis, windows)

##################### Processing steps

def get_soln(freqs, phis, t, offsets):
    '''
    Gets the full solution from frequencies and phis
    
    Parameters
    ----------
    freqs : the frequencies with shape (number of modes)
    phis : the phis with shape (number of modes x number of channels)
    t : the time snapshots
    offsets : the temporal offset of each window
    
    Returns
    -------
    soln : the extended solutions with shape (number of modes x number of channels x time)
    '''
    soln = np.empty((freqs.shape[0], phis.shape[1], len(t)))
    for i in range(len(freqs)):
        temp = np.exp(2*np.pi*1j*((t-offsets[i]) * freqs[i]))
        temp2 = phis[i,:][:,None]*temp
        soln[i] = temp2.real
    
    return(soln)


def get_t_delay_from_soln(freqs, phis, t, t_step, offsets):
    '''
    Predicts temporal delays between channels from the solutions
    
    Parameters
    ----------
    freqs : the frequencies with shape (number of modes)
    phis : the phis with shape (number of modes x number of channels)
    t : the time snapshots
    t_step : the temporal difference between snapshots (or 1 over sampling rate)
    offsets : the temporal offset of each window
    
    Returns
    -------
    t_delay : the predicted time delays with shape (number of windows x number of modes x number of channels)
    '''
    t_delay = np.empty((freqs.shape[0], phis.shape[1]), dtype=int)
    for i in range(len(freqs)):
        temp = np.exp(2*np.pi*1j*((t-offsets[i]) * freqs[i]))
        temp2 = phis[i,:][:,None]*temp
        temp3 = np.round(np.angle(temp2[:,0]) / (2*np.pi*freqs[i]) / t_step)
        t_delay[i] = np.array([int(ch) for ch in temp3])
    
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
    freqs : all frequencies with shape (number of windows, number of modes)
    phis : all phis with shape (number of windows, number of modes, number of recordings)
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
            groups.append(_group_by_freq_phi(freqs[:,i], np.abs(phis[:,i,:]), \
                                             thresh_freq=thresh_freq, thresh_phi_amp=thresh_phi_amp))
        else:
            temp1 = _group_by_polarity(freqs[:,i-1], freqs[:,i], 'freq_pol')
            temp2 = _group_by_polarity(np.abs(phis[:,i-1,:]), np.abs(phis[:,i,:]), 'phi_amp_pol')
            groups.append([[g] for g in np.unique(np.hstack((temp1,temp2)))])
    return(groups)

def get_red_init(group_idx, random_seed=None):
    '''
    Gets the initial reduction of subselection of indicies from similarities
    
    Parameters
    ----------
    group_idx : output of group_by_similarity, list of groups of similar indicies
    
    Returns
    -------
    idx_red : list of reduced indicies where each pair has the window and mode respectively
    '''
    if random_seed is not None:
        np.random.seed(random_seed)
    idx_red = []
    for j, groups in enumerate(group_idx):
        for g in groups:
            idx_red.append([np.random.choice(g),j])
    idx_red = np.array(idx_red)
    return(idx_red)

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


def exact_f_from_Bf(B, f, var_thresh=0.01):
    '''
    Gets the f_hat from the estimated f and B matrix in the exact method
    
    Parameters
    ----------
    B : B matrix from exact method
    f : approximate f from exact method
    var_thresh : variance threshold of eigenvalues to not be considered noise
    
    Returns
    -------
    f_hat : f_hat from the exact method
    '''
    f_hat = np.empty(f.shape)
    for t in range(f.shape[1]):
        f_sub = f[:,t]
        B_sub = B[:,:,t].T
        u,s,vh = np.linalg.svd(B_sub)
        idx = s**2 / (s@s) > var_thresh
        B_inv = vh.T[:,idx] @ np.diag(1./s[idx]) @ u.T[idx]
        f_hat[:,t] = B_inv @ f_sub
    return(f_hat)

def exact_f_greedy(B, f, soln, x, N, var_thresh=0.01, verbose=True):
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
    var_thresh : variance threshold of eigenvalues to not be considered noise
    verbose : let's you know how far in the fitting process you are
    
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
        if verbose:
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
                f_hat = exact_f_from_Bf(B_sub,f_sub, var_thresh=var_thresh)
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
    f : current guess of f with shape (number of modes, time) OR (number of modes, number of channels, time)
    x : original data with shape (number of channels, time)
    soln : solutions with shape (number of modes, number of channels, time)
    alpha : number indicating strength of l1 regularization
    beta : number indicating strength of temporal smoothing
    N : number of timepoints to smooth over
    
    Returns
    -------
    dLdf : gradient of loss function
    '''
    if len(f.shape)==3:
        Y = np.matmul(np.transpose(soln, [2,1,0]), np.transpose(f, [2,0,1]))[:,:,0] #time, chan
        f_mean = np.mean(f, axis=1)
    else:
        Y = np.matmul(np.transpose(soln, [2,1,0]), np.transpose(f)[:,:,None])[:,:,0] #time, chan
        f_mean = f
    Y2 = Y - x.T
    l2_term = np.matmul(np.transpose(soln, [2,0,1]), Y2[:,:,None])[:,:,0].T
    
    alpha_term = np.ones((f_mean.shape)) * alpha
    alpha_term[f_mean<0] = -alpha_term[f_mean<0]
    
    beta_term = np.zeros((f_mean.shape))
    for i in range(1,N+1):
        beta_term[:,:-i] = beta_term[:,:-i] - beta*(f_mean[:,i:]-f_mean[:,:-i])
        beta_term[:,i:] =  beta_term[:,i:]  + beta*(f_mean[:,i:]-f_mean[:,:-i])
    
    dLdf = l2_term + alpha_term + beta_term
    return(dLdf)

def grad_f(x, soln, alpha, beta, N, lr, maxiter, fit_coupling=False, t_delay=None):
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
    fit_coupling : flag telling whether or not to fit time delays with individual channels
    t_delay : time delays of each channel with shape (number of modes, number of channels)
    
    Returns
    -------
    f : approximation of global modulation f
    '''
    f = grad_f_init(x, soln, beta, N)
    f[f<0]=0
    if fit_coupling:
        idx = t_delay[:,:,None] + np.arange(f.shape[1])[None,None,:]
        idx[idx<0]=0
        idx[idx>=f.shape[1]]=f.shape[1]-1
        f_3D = np.empty((soln.shape))
    
    for k in range(maxiter):
        if fit_coupling:
            for i in range(t_delay.shape[0]):
                for j in range(t_delay.shape[1]):
                    f_3D[i,j] = f[i,idx[i,j]]
            dLdf = grad_f_grad_loss(f_3D, x, soln, alpha, beta, N)
        else:
            dLdf = grad_f_grad_loss(f, x, soln, alpha, beta, N)
            
        f = f - lr*dLdf
        f[f<0]=0
        f[:,N:-N] = utils.moving_average_dim(f,2*N+1,-1)
        f[:,:N] = np.mean(f[:,:N],axis=-1)[:,None]
        f[:,-N:] = np.mean(f[:,-N:],axis=-1)[:,None]
    if fit_coupling:
        for i in range(t_delay.shape[0]):
            for j in range(t_delay.shape[1]):
                f_3D[i,j] = f[i,idx[i,j]]
        return(f_3D)
    else:
        return(f)

def grad_f_amp(f, soln, x):
    '''
    Fixes overall amplitude of f hat in the gradient descent method
    
    Parameters
    ----------
    f : computed solution for f with shape (number of modes, time) OR (number of modes, num of chans, time)
    soln : solutions with shape (number of modes, number of channels, time)
    x : data matrix with shape (number of channels, time)
    
    Returns
    -------
    f_hat : global modulation f with fixed amplitude
    '''
    if len(f.shape)==3:
        norm,_,_,_ = np.linalg.lstsq((f * soln).reshape((len(f),-1)).T, x.reshape((-1)),rcond=None)
        f_hat = f * norm[:,None,None]
    else:
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
    f : global modulation f with shape (number of modes, time) OR (num modes, num chans, time)
    
    Returns
    -------
    x_rec : reconstructed data matrix
    '''
    if len(f.shape)==3:
        x_rec = np.sum(soln * f, axis=0)
    else:
        x_rec = np.sum(soln * f[:,None,:], axis=0)
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


######################### Development/doesn't currently work

def find_lag(x, f, soln, t_delay, periods, N, edge_len):
    errors = np.empty((soln.shape[0], 2*N+1, soln.shape[1]))
    for i in range(soln.shape[0]):
        for j in np.arange(-N,N+1):
            f_temp = f.copy()
            for k in range(soln.shape[1]):
                idx = int(round(j*periods[i])) + t_delay[i,k] + np.arange(soln.shape[2])
                idx[idx<0]=0
                idx[idx>=soln.shape[2]]=soln.shape[2]-1
                f_temp[i,k] = f[i,k,idx]
            x_rec = nsdmd.get_reconstruction(soln[:,:,edge_len:-edge_len], f_temp[:,:,edge_len:-edge_len])
            errors[i,j] = np.mean((x_rec - x[:,edge_len:-edge_len])**2, axis=1)
    return errors