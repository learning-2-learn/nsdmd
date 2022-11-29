import numpy as np
from nsdmd import optdmd
from nsdmd import utils
from scipy.stats import circmean

################## Classes
# from attrs import define
# @define
# class NSDMD:
#     opt_win: int = 500
#     opt_stride: int = 100
#     opt_rank: int = 20
#     bandpass: list[list[int]] = None
#     bandpass_trim: int = 500
#     sim_thresh_freq: float = 0.2
#     sim_thresh_phi_amp: float = 0.95
#     sim_thresh_phi_phase: float = 0.05
#     drift_N: int = 51
#     exact_var_thresh: float = 0.01
#     feature_init: float, int = None
#     feature_N: int = 20
#     feature_seq_method: str = 'SBS'
#     feature_f_method: str = 'exact'
#     feature_maxiter: int = 5
#     feature_final_num: int = None
#     feature_maxiter_float: int = 1
#     grad_alpha: float = 0.1
#     grad_beta: float = 0.1
#     grad_N: int = 20
#     grad_lr: float = 0.01
#     grad_maxiter: int = 100
#     grad_fit_coupling: bool = False
#     verbose: bool = False


class NSDMD:
    def __init__(
        self,
        opt_win=500,
        opt_stride=100,
        opt_rank=20,
        bandpass=None,
        bandpass_trim=500,
        sim_thresh_freq=0.2,
        sim_thresh_phi_amp=0.95,
        sim_thresh_phi_phase=0.05,
        sim_group_size=1,
        drift_flag=True,
        drift_N=51,
        exact_var_thresh=0.01,
        feature_init=None,
        feature_N=20,
        feature_seq_method="SBS",
        feature_f_method="exact",
        feature_maxiter=5,
        feature_final_num=None,
        feature_maxiter_float=1,
        grad_alpha=0.1,
        grad_beta=0.1,
        grad_N=20,
        grad_lr=0.01,
        grad_momentum=0.9,
        grad_maxiter=100,
        grad_fit_coupling=False,
        grad_init_lowpass=2,
        verbose=False,
    ):
        self.opt_win = opt_win
        self.opt_stride = opt_stride
        self.opt_rank = opt_rank
        self.bandpass = bandpass
        self.bandpass_trim = bandpass_trim
        self.sim_thresh_freq = sim_thresh_freq
        self.sim_thresh_phi_amp = sim_thresh_phi_amp
        self.sim_thresh_phi_phase = sim_thresh_phi_phase
        self.sim_group_size = sim_group_size
        self.drift_flag = drift_flag
        self.drift_N = drift_N
        self.exact_var_thresh = exact_var_thresh
        self.feature_init = feature_init
        self.feature_N = feature_N
        self.feature_seq_method = feature_seq_method
        self.feature_f_method = feature_f_method
        self.feature_maxiter = feature_maxiter
        self.feature_final_num = feature_final_num
        self.feature_maxiter_float = feature_maxiter_float
        self.grad_alpha = grad_alpha
        self.grad_beta = grad_beta
        self.grad_N = grad_N
        self.grad_lr = grad_lr
        self.grad_momentum = grad_momentum
        self.grad_maxiter = grad_maxiter
        self.grad_fit_coupling = grad_fit_coupling
        self.grad_init_lowpass = grad_init_lowpass
        self.verbose = verbose

    def fit(self, x, t, sr, initial_freq_guess=None):
        self.fit_opt(x, t, sr, initial_freq_guess=initial_freq_guess)
        if self.bandpass is not None:
            x = x[:,self.bandpass_trim:self.bandpass_trim+self.windows_[-1,0]+self.opt_win]
            t = t[self.bandpass_trim:self.bandpass_trim+self.windows_[-1,0]+self.opt_win]
        self.fit_reduction(x, len(t), sr)
        guess = guess_best_fit_idx(self.num_modes_, self.errors_)
        if self.verbose:
            print("Guessing " + str(guess) + " number of modes")
        self.fit_f(x, len(t), sr, guess)
        return self

    def fit_opt(self, x, t, sr, initial_freq_guess=None):
        if self.verbose:
            print("Starting OPT-DMD...")
        if self.bandpass is None:
            f, p, w = opt_dmd_win(
                x, t, self.opt_win, self.opt_stride, self.opt_rank, initial_freq_guess
            )
            self.freqs_ = f
            self.phis_ = p
            self.windows_ = w
            self.offsets_ = (
                t[self.windows_[:, 0]][:, None]
                * np.ones((self.freqs_.shape[1]), dtype=int)[None, :]
            )
        else:
            f, p, w = opt_dmd_with_bandpass(
                x,
                t,
                sr,
                self.opt_win,
                self.opt_stride,
                self.opt_rank,
                self.bandpass,
                self.bandpass_trim,
                initial_freq_guess,
                self.verbose,
            )
            self.freqs_ = f
            self.phis_ = p
            self.windows_ = w
            self.offsets_ = (
                t[self.bandpass_trim : -self.bandpass_trim][self.windows_[:, 0]][
                    :, None
                ]
                * np.ones((self.freqs_.shape[1]), dtype=int)[None, :]
            )

        return self

    def set_opt_values(self, freqs, phis, windows, offsets):
        self.freqs_ = freqs
        self.phis_ = phis
        self.windows_ = windows
        self.offsets_ = offsets
        return self

    def fit_reduction(self, x, t_len, sr):
        if self.verbose:
            print("Gathering modes...")
        group_idx = group_by_similarity(
            self.freqs_, self.phis_, self.opt_stride/sr, self.sim_thresh_freq, self.sim_thresh_phi_amp, self.sim_thresh_phi_phase
        )

        idx_init = get_red_init(group_idx, len(self.windows_), self.sim_group_size)
        if not self.drift_flag:
            #Randomly picks solutions instead of drift
            for i in range(len(idx_init)):
                idx_init[i,0] = np.ones(idx_init.shape[2])*np.random.choice(np.unique(idx_init[i,0]))
        idx_init = idx_init[
            ~np.all(self.freqs_[tuple(np.transpose(idx_init, [1, 0, 2]))] == 0, axis=1)
        ]
        soln, freqs, phis = get_soln(
            self.freqs_, self.phis_, idx_init, t_len, self.windows_, self.drift_N, sr
        )

        if self.verbose:
            print("Found " + str(len(idx_init)) + " modes")

        if self.feature_init is not None:
            idx_feat_init = feature_init_remove(
                soln, np.mean(freqs, axis=1), x, sr, thresh=self.feature_init
            )
            idx_init = idx_init[idx_feat_init]
            soln = soln[idx_feat_init]
            freqs = freqs[idx_feat_init]
            phis = phis[idx_feat_init]

            if self.verbose:
                print("Initially reducing to " + str(len(idx_init)) + " modes\n")

        if self.grad_fit_coupling:
            if self.verbose:
                print("Calculating individual delays...\n")
            freq_mean = np.mean(freqs, axis=1)
            p = circmean(np.angle(phis), axis=1, high=np.pi, low=-np.pi)
            delay = np.array(
                np.round(p * sr / (2 * np.pi * freq_mean[:, None])), dtype=int
            )
        else:
            delay = None
            
        redux_class = reduction(
            sr=sr,
            bands=self.bandpass,
            band_trims=self.bandpass_trim,
            final_num=self.feature_final_num,
            seq_method=self.feature_seq_method,
            f_method=self.feature_f_method,
            maxiter_float=self.feature_maxiter_float,
            exact_var_thresh=self.exact_var_thresh,
            grad_alpha=self.grad_alpha,
            grad_beta=self.grad_beta,
            grad_lr=self.grad_lr,
            grad_momentum=self.grad_momentum,
            grad_init_lowpass=self.grad_init_lowpass,
            maxiter=self.feature_maxiter,
            grad_fit_coupling=self.grad_fit_coupling,
            grad_delay=delay,
            verbose=self.verbose
        )

        idxs, self.errors_, self.num_modes_ = redux_class.feature_selector(
            soln,
            x,
            self.feature_N
        )

        self.idx_red_ = [idx_init[idx] for idx in idxs]
        return self

    def fit_f(self, x, t_len, sr, idx_num):
        if np.any(self.num_modes_ == idx_num):
            self.idx_hat_ = self.idx_red_[
                np.argwhere(self.num_modes_ == idx_num)[-1, 0]
            ]
        else:
            print("Num modes requested has not been calculated, returning...")
            return self

        soln, freqs, phis = get_soln(
            self.freqs_,
            self.phis_,
            self.idx_hat_,
            t_len,
            self.windows_,
            self.drift_N,
            sr,
        )

        self.freq_mean_ = np.mean(freqs, axis=1)
        a = np.mean(np.abs(phis), axis=1)
        p = circmean(np.angle(phis), axis=1, high=np.pi, low=-np.pi)
        self.phi_mean_ = a * np.exp(1j * p)

        self.delay_hat_ = np.array(
            np.round(p * sr / (2 * np.pi * self.freq_mean_[:, None])), dtype=int
        )

        f_hat = grad_f(
            x,
            soln,
            self.grad_alpha,
            self.grad_beta,
            self.grad_N,
            self.grad_lr,
            self.grad_momentum,
            self.grad_maxiter,
            self.grad_init_lowpass,
            self.grad_fit_coupling,
            self.delay_hat_,
        )
        self.f_hat_ = grad_f_amp(f_hat, soln, x)
        return self

    def transform(self, x, t_len, sr):
        soln, _, _ = get_soln(
            self.freqs_,
            self.phis_,
            self.idx_hat_,
            t_len,
            self.windows_,
            self.drift_N,
            sr,
        )
        x_rec = get_reconstruction(soln, self.f_hat_)
        return x_rec

    def get_freq_and_phi(self, t_len, sr):
        _, freqs, phis = get_soln(
            self.freqs_,
            self.phis_,
            self.idx_hat_,
            t_len,
            self.windows_,
            self.drift_N,
            sr,
        )
        return (freqs, phis)


################## OPT-DMD


def opt_dmd_win(x, t, w_len, stride, rank, initial_freq_guess=None):
    """
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
    """
    windows = np.array(
        [np.arange(i, i + w_len) for i in np.arange(0, x.shape[-1] - w_len + 1, stride)]
    )
    freqs = np.empty((len(windows), rank))
    phis = np.empty((len(windows), rank, len(x)), dtype=complex)

    for i, window in enumerate(windows):
        x_temp = x[:, window]
        t_temp = t[window]
        t_temp -= t_temp[0]  # Forces the starting time to be 0

        if i == 0:
            if initial_freq_guess is None:
                guess = None
            else:
                assert (
                    len(initial_freq_guess) == rank
                ), "Number of frequencies guessed isnt equal to the rank"
                guess = 0.0 + 1j * 2 * np.pi * initial_freq_guess
        else:
            guess = 0.0 + 1j * 2 * np.pi * freqs[i - 1]  # Setting real part to be 0

        dmd = optdmd.OptDMD(x_temp, t_temp, rank)
        dmd.fit(verbose=False, eigs_guess=guess)

        freqs[i] = np.array(dmd.eigs).imag / 2.0 / np.pi
        phis[i] = dmd.modes.T

    return (freqs, phis, windows)


def opt_dmd_with_bandpass(
    x, t, sr, w_len, stride, rank, bp_ranges, trim, initial_guess=None, verbose=False
):
    """
    Runs opt_dmd_win for bandpassed regions

    Parameters
    ----------
    x : data matrix
    t : timepoints of x
    sr : sampling rate
    w_len : window length
    stride : stride for opt_dmd_win
    rank : rank of opt_dmd
    bp_ranges : list of bandpass ranges (e.g. [[1,4],[4,7],...])
    trim : amount of data matrix to exclude after bandpassing
    initial_guess : list of initial guesses for frequencies, corresponding to bp_ranges
    verbose : flag to say whether to show comments

    Returns
    -------
    freqs_ : non-trivial freqs
    phis_ : non-trivial phis
    w : corresponding windows
    """
    bp_ranges = np.array(bp_ranges)
    assert len(bp_ranges.shape) == 2, "Must be 2 dimensional"
    assert bp_ranges.shape[1] == 2, "Second dimension needs to be of length 2"
    if initial_guess is not None:
        initial_guess = np.array(initial_guess)
        assert len(bp_ranges) == len(
            initial_guess
        ), "Guess must be of same length as bp_ranges"
        assert (
            initial_guess.shape[1] == rank
        ), "Guess must have the same number of modes as rank"

    t = t.copy()[trim:-trim]
    freqs = []
    phis = []
    for i, bp in enumerate(bp_ranges):
        if verbose:
            print("Starting bandpass freq: " + str(bp[0]) + " - " + str(bp[1]) + " Hz")

        x_filt = _bandpass_x(x, sr, bp[0], bp[1], trim=trim)
        if initial_guess is not None:
            guess = _bandpass_guess(bp[0], bp[1], rank, initial_guess[i])
        else:
            guess = _bandpass_guess(bp[0], bp[1], rank, initial_guess)

        f, p, w = opt_dmd_win(x_filt, t, w_len, stride, rank, guess)
        f, p = _bandpass_exclude(f, p, bp[0], bp[1])

        freqs.append(f)
        phis.append(p)
    freqs_ = np.hstack(freqs)
    phis_ = np.hstack(phis)
    return (freqs_, phis_, w)


def _bandpass_x(x, sr, bp_low, bp_high, bp_filter='chebyshev', trim=None):
    """
    Bandpasses data to specified freq range

    Parameters
    ----------
    x : data matrix
    sr : sampling rate
    bp_low : lower bandpass range
    bp_high : upper bandpass range
    bp_filter : type of filter to use to bandpass the data.
        Can either be 'chebyshev' or 'butter'
    trim : how much of data to trim after bandpassing

    Returns
    -------
    x_filt : filtered data
    """
    assert bp_filter=='chebyshev' or bp_filter=='butter', 'wrong bandpass filter type'
    if bp_filter=='butter':   
        temp = utils.butter_pass_filter(x.copy(), bp_low, int(sr), "high")
        temp2 = utils.butter_pass_filter(temp, bp_high, int(sr), "low")
    else:
        temp = utils.cheb_pass_filter(x.copy(), bp_low, int(sr), "high")
        temp2 = utils.cheb_pass_filter(temp, bp_high, int(sr), "low")
    # x_filt = temp2 / np.std(temp2, axis=-1)[:, None] # Seems to mess things up??
    x_filt = temp2

    if trim is not None:
        x_filt = x_filt[:, trim:-trim]
    return x_filt


def _bandpass_guess(bp_low, bp_high, rank, initial_guess=None):
    """
    Finds the frequency guess of bandpassed data

    Parameters
    ----------
    bp_low : lower bound on frequency range
    bp_high : upper bound on frequency range
    rank : rank of guess
    initial_guess : initial guess of frequencies

    Returns
    -------
    guess : guess of desired rank and bandpass range
    """
    if initial_guess is None:
        guess = np.random.rand(int(np.ceil(rank / 2))) * (bp_high - bp_low) + bp_low
        guess = np.hstack([[g, -g] for g in guess])
        guess = guess[:rank]
    else:
        assert (
            len(initial_guess) == rank
        ), "Length of initial guess must be the same as rank"
        assert (
            np.all((np.abs(initial_guess)>bp_low) & (np.abs(initial_guess)<bp_high))
        )
        guess = initial_guess
    return guess


def _bandpass_exclude(freq, phi, bp_low, bp_high):
    """
    Excluded modes outside of bandpass range

    Parameters
    ----------
    freq : frequencies
    phi : spatial modes
    bp_low : lower bound on frequencies
    bp_high : upper bound on frequencies

    Returns
    -------
    freq : non-trivial frequencies outside of frequency range
    phi : non-trivial phis outside of frequency range
    """
    idx = (np.abs(freq) < bp_low) | (np.abs(freq) > bp_high)
    freq[idx] = 0
    phi[idx] = 0
    freq = freq[:, ~np.all(freq == 0, axis=0)]
    phi = phi[:, ~np.all(np.all(phi == 0, axis=0), axis=1)]

    return (freq, phi)


##################### Processing steps


def get_phi_init(freqs, phi, offsets, sr):
    """
    Gets the spatial mode, phi, at t=0

    Parameters
    ----------
    freqs : frequencies at every timepoint with shape (time)
    phi : phi at every measured window with shape (number of windows, number of channels)
    offsets : the temporal offsets (in number of timesteps) of each window with shape (number of windows)
    sr : the sampling rate

    Returns
    -------
    phi_init : phi at t=0 for each window with shape (number of windows, number of channels)
    """
    freqs = np.insert(freqs[:-1], 0, 0)
    phase_in = np.cumsum(freqs / sr)
    t_diff = np.exp(-2 * np.pi * 1j * phase_in[offsets])
    phi_init = phi * t_diff[:, None]
    return phi_init


def get_soln(freqs, phis, idxs, t_len, windows, N, sr):
    """
    Gets the solutions and calculates the frequencies, and phis over time

    Parameters
    ----------
    freqs : frequencies with shape (num windows, num modes)
    phis : phis with shape (num windows, num modes, num channels)
    idxs : list of rows (windows) and cols (modes)
        This list represents the sub groups of similar solutions
    t_len : length of entire region of interest
    windows : list of windows
    N : amount of temporal averaging of the frequecies
    sr : sampling rate

    Returns
    -------
    soln : solutions with shape (num modes, num channels, time)
    freq_all : frequencies across time with shape (num modes, time)
    phi_all : phis across time (except with a phase at t=0) with shape (num modes, num channels, time)
    """
    soln = np.empty((len(idxs), phis.shape[-1], t_len))

    loc = [[] for _ in range(t_len)]
    for i, win in enumerate(windows):
        for w in win:
            loc[w].append(i)
    loc_len = np.array([len(l) for l in loc])
    loc = np.array(loc, dtype=object)

    idx = tuple(np.transpose(idxs, [1, 0, 2]))

    freqs_sub = freqs[idx]
    freqs_all = np.empty((len(idxs), t_len))
    for i in np.unique(loc_len):
        for l in np.unique(loc[loc_len == i]):
            if type(l) == list:
                temp = np.array(l)
            else:
                temp = np.array([l])
            temp_i = np.argwhere(
                np.all(temp == np.array(list(loc[loc_len == i])), axis=1)
            )[:, 0]
            temp_ii = np.argwhere(loc_len == i)[:, 0][temp_i]
            temp_f = np.mean(freqs_sub[:, temp], axis=1)
            freqs_all[:, temp_ii] = temp_f[:, None]

    freqs_m = utils.moving_average_dim(freqs_all, N, 1)
    freqs_m = np.hstack(
        (
            freqs_m[:, 0][:, None]
            * np.ones(t_len - freqs_m.shape[1] - int(N / 2))[None, :],
            freqs_m,
            freqs_m[:, -1][:, None] * np.ones(int(N / 2))[None, :],
        )
    )

    phis_init = np.empty((len(idxs), len(windows), phis.shape[-1]), dtype=complex)
    for i in range(len(idxs)):
        phis_init[i] = get_phi_init(
            freqs_m[i], phis[tuple(idxs[i])], windows[idxs[i][0], 0], sr
        )

    phis_all = np.empty((len(idxs), t_len, phis.shape[-1]), dtype=complex)

    for i in np.unique(loc_len):
        for l in np.unique(loc[loc_len == i]):
            if type(l) == list:
                temp = np.array(l)
            else:
                temp = np.array([l])
            temp_i = np.argwhere(
                np.all(temp == np.array(list(loc[loc_len == i])), axis=1)
            )[:, 0]
            temp_ii = np.argwhere(loc_len == i)[:, 0][temp_i]
            temp_pa = np.mean(np.abs(phis_init[:, temp]), axis=1)
            temp_pp = circmean(np.angle(phis_init[:, temp]), axis=1)
            phis_all[:, temp_ii] = (temp_pa * np.exp(1j * temp_pp))[:, None, :]

    freqs_in = np.insert(freqs_m[:, :-1], 0, np.zeros(len(idxs)), axis=1)
    phase_in = np.cumsum(freqs_in / sr, axis=1)
    temp = np.exp(2 * np.pi * 1j * phase_in)
    soln = np.transpose((phis_all * temp[:, :, None]).real, [0, 2, 1])

    return (soln, freqs_all, phis_all)


def group_by_similarity(freqs, phis, deltaT, thresh_freq=0.2, thresh_phi_amp=0.95, thresh_phi_phase=0.05):
    """
    Groups all modes based on frequencies, phi amplitudes, and phi angles.
    Note : modes are expected to be in order of pairs, where each pair represents the positive and negative frequency.
        This function cannot currently handle non-pairs
    Note : currently cannot control threshold of polarity

    Parameters
    ----------
    freqs : all frequencies with shape (number of windows, number of modes)
    phis : all phis with shape (number of windows, number of modes, number of recordings)
    deltaT: the time difference between all pairs of snapshots in seconds. Assumes consistent sampling rate
    thresh_freq : frequency threshold. Any pair of frequencies with a smaller difference is 'similar'
    thresh_phi_amp : phi_amp threshold. Any pair with larger value is 'similar'
        value is computed by cosine distance metric
    thresh_phi_phase : phi_phase threshold. Any pair with smaller value is 'similar'
        value is MSE

    Returns
    -------
    groups : list of groups with length (number of modes)
        even modes (0,2,4,...) contain groups (lists) of consecutive similar solutions
        odd modes (1,3,5,...) contain a list of solutions (of paired modes) that are significantly different than counterparts

    """
    groups = []
    for i in range(0, freqs.shape[1]):
        if i % 2 == 0:
            temp = np.angle(phis[:,i,:])
            tempf = freqs[:,i]
            #Note below that we did 2 divided by 2. This aligns forward and backwards
            p_ahead = ((temp + (np.pi*tempf*deltaT)[:,None]) % (2*np.pi))[:-1]
            p_back = ((temp - (np.pi*tempf*deltaT)[:,None]) % (2*np.pi))[1:]
            p_diff = (p_ahead - p_back + np.pi)%(2*np.pi) - np.pi
            groups.append(
                _group_by_freq_phi(
                    freqs[:, i],
                    np.abs(phis[:, i, :]),
                    p_diff,
                    thresh_freq=thresh_freq,
                    thresh_phi_amp=thresh_phi_amp,
                    thresh_phi_phase=thresh_phi_phase
                )
            )
        else:
            temp1 = _group_by_polarity(freqs[:, i - 1], freqs[:, i], "freq_pol")
            temp2 = _group_by_polarity(
                np.abs(phis[:, i - 1, :]), np.abs(phis[:, i, :]), "phi_amp_pol"
            )
            temp3 = _group_by_polarity(
                np.angle(phis[:, i - 1, :]), -np.angle(phis[:, i, :]), "phi_angle_pol"
            )
            groups.append([[g] for g in np.unique(np.hstack((temp1, temp2, temp3)))])
    return groups


def get_red_init(group_idx, num_windows, min_group_size=1):
    """
    Gets the initial reduction of subselection of indicies from similarities

    Parameters
    ----------
    group_idx : output of group_by_similarity, list of groups of similar indicies
    num_windows : total number of windows
    min_group_size : keeps only groups with a groupsize equal or larger than the threshold

    Returns
    -------
    idx_red : list of rows (windows) and columns (modes)
    """
    idx_red = []
    for j, groups in enumerate(group_idx):
        for g in groups:
            if len(g)>=min_group_size:
                min_val = g[0]
                max_val = g[-1]
                temp = np.hstack(
                    (
                        np.ones((min_val), dtype=int) * min_val,
                        g,
                        np.ones((num_windows - max_val - 1), dtype=int) * max_val,
                    )
                )
                idx_red.append([temp, np.ones(len(temp)) * j])
    idx_red = np.array(idx_red, dtype=int)
    return idx_red


def feature_init_remove(soln, freqs, x, sr, thresh=0.2):
    """
    Initially keeps some number of solutions based on simple reconstruction error
    Assumes no global modulation

    Parameters
    ----------
    soln : solutions with shape (num modes, num channels, time)
    freqs : average frequency for each mode (num modes)
    x : original data with shape (num channels, time)
    sr : sampling rate
    threh : either integer with value >= 1 and < num modes (this keeps that many modes with best reconstruction)
        or float with value between 0 and 1 (this keeps all modes above this cosine distance)

    Returns
    -------
    idxs : list of indicies that surpass threshold
    """
    freqs = np.abs(freqs)
    f = np.ones((1, soln.shape[-1]))
    # f = grad_f_init(x,soln,0,2)
    
    errors = np.empty((len(soln)))
    for i in range(len(soln)):
        soln_sub = soln[[i]]
        x_sub = _bandpass_x(x, sr, np.max([freqs[i] - 1, 0.1]), freqs[i] + 1)

        f_sub = grad_f_amp(f, soln_sub, x_sub)
        # f_sub = np.array([f[i]])
        x_rec = get_reconstruction(soln_sub, f_sub)
        errors[i] = get_reconstruction_error(x_rec, x_sub)

    if thresh > 0 and thresh < 1:
        idxs = np.argwhere(errors > thresh)[:, 0]
    elif thresh >= 1 and thresh < len(soln):
        idxs = np.argsort(errors)[::-1][: int(thresh)]
    else:
        print(
            "Incorrect threshold, should be either value between 0 and 1 or integer above 1"
        )
    return idxs


###################### Exact Method


def exact_Bf(x, soln):
    """
    Gets B and f approximate with the exact method approach

    Parameters
    ----------
    x : the data with shape (number of channels, time)
    soln : solutions with shape (number of modes, number of channels, time)

    Returns
    -------
    B : B matrix from exact method
    f : approximate f from exact method
    """
    top = np.sum(soln * x[None, :, :], axis=1)
    bot = np.sum(soln**2, axis=1)
    f = top / bot
    B = np.sum(soln[:, None] * soln[None, :], axis=2) / bot[:, None]
    return (B, f)


def exact_f_from_Bf(B, f, N, var_thresh=0.01):
    """
    Gets the f_hat from the estimated f and B matrix in the exact method

    Parameters
    ----------
    B : B matrix from exact method
    f : approximate f from exact method
    N : amount to average over
    var_thresh : variance threshold of eigenvalues to not be considered noise

    Returns
    -------
    f_hat : f_hat from the exact method
    """
    f_hat = np.empty(f.shape)
    for t in range(f.shape[1]):
        f_sub = f[:, t]
        B_sub = B[:, :, t].T
        u, s, vh = np.linalg.svd(B_sub)
        idx = s**2 / (s @ s) > var_thresh
        B_inv = vh.T[:, idx] @ np.diag(1.0 / s[idx]) @ u.T[idx]
        f_hat[:, t] = B_inv @ f_sub

    f_hat[:, N:-N] = utils.moving_average_dim(f_hat, 2 * N + 1, 1)
    f_hat[f_hat < 0] = 0
    return f_hat


#################### Reduction Method


class reduction:
    """
    Class for reducing the number of modes used in NS-DMD
    Sequential reduction methods include SBS, SFS, SBFS, and SFFS, described here:
    http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/

    Global modulation methods include exact, and grad, described in the paper
    
    Attributes
    ----------
    self.sr : sampling rate of dataset,
    self.bands : bands to evaluate feature selection over,
    self.band_trims : amount of data to throw away after bandpassing
    self.final_num : end number of modes to stop computing feature selection
    self.seq_method : method of sequential feature selection
    self.floating : bool indicating whether to include floating methods in sequential feature selection
    self.f_method : method of exact or gradient descent
    self.maxiter_float : parameter to  stop sequential selector from repeating too many times
    self.exact_var_thresh : threshold for exact method
    self.grad_alpha : alpha for gradient descent
    self.grad_beta : beta for gradient descent
    self.grad_lr : lr for gradient descent
    self.grad_momentum : momentum for gradient descent
    self.grad_init_lowpass : value of initial guess lowpass (or None if no low pass)
    self.maxiter : how many runs for quick gradient descent
    self.grad_fit_coupling : whether to fit coupling for gradient descent
    self.grad_delay : coupling delay for gradient descent
    self.verbose : flag to show comments as it's processing
    
    Methods
    -------
    feature_selector(soln, x, feature_N)
        Computes feature selection on dataset x
    """
    def __init__(
        self,
        sr=1000,
        bands=None,
        band_trims=None,
        final_num=None,
        seq_method="SBS",
        f_method="exact",
        maxiter_float=1,
        exact_var_thresh=0.01,
        grad_alpha=0.1,
        grad_beta=0.1,
        grad_lr=0.01,
        grad_momentum=0.9,
        grad_init_lowpass=None,
        maxiter=5,
        grad_fit_coupling=False,
        grad_delay=None,
        verbose=True
    ):
        """
        Initializes reduction class
        
        Parameters
        ----------
        sr : sampling rate of dataset
        bands : bands to evaluate feature selection over
        band_trims : amount of data to throw away after bandpassing
        final_num : end number of modes to stop computing feature selection
        seq_method : method of sequential feature selection
        f_method : method of exact or gradient descent
        maxiter_float : parameter to  stop sequential selector from repeating too many times
        exact_var_thresh : threshold for exact method
        grad_alpha : alpha for gradient descent
        grad_beta : beta for gradient descent
        grad_lr : lr for gradient descent
        grad_momentum : momentum for gradient descent
        grad_init_lowpass : value of initial guess lowpass (or None if no low pass)
        maxiter : how many runs for quick gradient descent
        grad_fit_coupling : whether to fit coupling for gradient descent
        grad_delay : coupling delay for gradient descent
        verbose : flag to show comments as it's processing
        """
        self.sr=sr
        self.bands=bands
        self.band_trims=band_trims
        self.final_num=final_num
        self.seq_method=seq_method
        self.f_method=f_method
        self.maxiter_float=maxiter_float
        self.exact_var_thresh=exact_var_thresh
        self.grad_alpha=grad_alpha
        self.grad_beta=grad_beta
        self.grad_lr=grad_lr
        self.grad_momentum=grad_momentum
        self.grad_init_lowpass=grad_init_lowpass
        self.maxiter=maxiter
        self.grad_fit_coupling=grad_fit_coupling
        self.grad_delay=grad_delay
        self.verbose=verbose

    def feature_selector(
        self,
        soln,
        x,
        feature_N
    ):
        """
        Wrapper to compute feature selection.
        Sequential methods include SBS, SFS, SBFS, and SFFS, described here:
        http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/

        Global modulation methods include exact, and grad, described in the paper

        Parameters
        ----------
        soln : solutions with shape (num modes, num chan, time)
        x : true data with shape (num chan, time)
        feature_N : amount of averaging to do on global modulation f

        Returns
        -------
        idxs : list of idxs used in feature selection
        errors : list of errors for each step
        num_modes : number of modes for each step
        """
        if self.f_method == "exact" or self.f_method == "grad":
            if self.seq_method == "SBS":
                self.floating = False
                idxs, errors, num_modes, = self._SBS(
                    soln,
                    x,
                    feature_N
                )
            elif self.seq_method == "SFS":
                self.floating = False
                idxs, errors, num_modes, = self._SFS(
                    soln,
                    x,
                    feature_N
                )
            elif self.seq_method == "SBFS":
                self.floating = True
                idxs, errors, num_modes, = self._SBS(
                    soln,
                    x,
                    feature_N
                )
            elif self.seq_method == "SFFS":
                self.floating = True
                idxs, errors, num_modes, = self._SFS(
                    soln,
                    x,
                    feature_N
                )
            else:
                idxs = []
                errors = []
                num_modes = 0
                print("Incorrect sequential method, must be SBS, SFS, SBFS, or SFFS")
        else:
            idxs = []
            errors = []
            num_modes = 0
            print("Incorrect global modulation method("+str(self.f_method)+"), must be exact or grad")

        return idxs, errors, num_modes


    def _SBS(
        self,
        soln,
        x,
        N
    ):
        """
        Sequential methods: SBS and SBFS, described here:
        http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/

        Global modulation methods include exact, and grad, described in the paper

        Parameters
        ----------
        soln : solutions with shape (num modes, num chan, time)
        x : true data with shape (num chan, time)
        N : amount of averaging to do on global modulation f

        Returns
        -------
        idx_all : list of idxs used in feature selection
        total_error : list of errors for each step
        num_modes : number of modes for each step
        """
        if self.final_num is None or self.final_num <= 0:
            self.final_num = 0
        elif self.final_num >= soln.shape[0]:
            self.final_num = soln.shape[0] - 1

        if self.verbose:
            print(
                "Number of modes: "
                + str(soln.shape[0])
                + "/"
                + str(soln.shape[0])
                + ", stopping at "
                + str(self.final_num)
            )
        if self.f_method == "grad":
            f_hat = grad_f(
                x,
                soln,
                self.grad_alpha,
                self.grad_beta,
                N,
                self.grad_lr,
                self.grad_momentum,
                self.maxiter,
                self.grad_init_lowpass,
                self.grad_fit_coupling,
                self.grad_delay,
            )
            f_hat = grad_f_amp(f_hat, soln, x)
        else:
            B, f = exact_Bf(x, soln)
            f_hat = exact_f_from_Bf(B, f, N, var_thresh=self.exact_var_thresh)

        x_rec = get_reconstruction(soln, f_hat)
        total_error = [get_reconstruction_error(x, x_rec, self.sr, self.bands, self.band_trims)]

        idx = np.arange(soln.shape[0])
        idx_excluded = []
        idx_all = [idx]
        num_modes = [soln.shape[0]]

        i = soln.shape[0] - 1
        i_last = soln.shape[0]

        while self.final_num != i:
            if self.verbose:
                print(
                    "Number of modes: "
                    + str(i)
                    + "/"
                    + str(soln.shape[0])
                    + ", stopping at "
                    + str(self.final_num)
                )
            if i_last == i + 1:
                num_float = 0
                i_last = i

            error = np.empty(len(idx))
            for j, r in enumerate(idx):
                idx_sub = idx[idx != r]
                if self.f_method == "grad":
                    f_hat = grad_f(
                        x,
                        soln[idx_sub],
                        self.grad_alpha,
                        self.grad_beta,
                        N,
                        self.grad_lr,
                        self.grad_momentum,
                        self.maxiter,
                        self.grad_init_lowpass,
                        self.grad_fit_coupling,
                        self.grad_delay,
                    )
                    f_hat = grad_f_amp(f_hat, soln[idx_sub], x)
                else:
                    f_sub = f[idx_sub]
                    B_sub = np.array([b[idx_sub] for b in B[idx_sub]])
                    f_hat = exact_f_from_Bf(B_sub, f_sub, N, var_thresh=self.exact_var_thresh)

                x_rec = get_reconstruction(soln[idx_sub], f_hat)
                error[j] = get_reconstruction_error(x, x_rec, self.sr, self.bands, self.band_trims)

            total_error.append(np.max(error))
            idx_excluded.append(idx[np.argmax(error)])
            idx = idx[np.arange(len(idx)) != np.argmax(error)]
            idx_all.append(idx)
            num_modes.append(i)

            flag_float = True
            while (
                self.floating
                and flag_float
                and i < soln.shape[0] - 1
                and num_float != self.maxiter_float
            ):
                error = np.empty(len(idx_excluded))
                for j, r in enumerate(np.array(idx_excluded)):
                    idx_sub = np.array(list(idx) + [r])
                    if self.f_method == "grad":
                        f_hat = grad_f(
                            x,
                            soln[idx_sub],
                            self.grad_alpha,
                            self.grad_beta,
                            N,
                            self.grad_lr,
                            self.grad_momentum,
                            self.maxiter,
                            self.grad_init_lowpass,
                            self.grad_fit_coupling,
                            self.grad_delay,
                        )
                        f_hat = grad_f_amp(f_hat, soln[idx_sub], x)
                    else:
                        f_sub = f[idx_sub]
                        B_sub = np.array([b[idx_sub] for b in B[idx_sub]])
                        f_hat = exact_f_from_Bf(
                            B_sub, f_sub, N, var_thresh=self.exact_var_thresh
                        )

                    x_rec = get_reconstruction(soln[idx_sub], f_hat)
                    error[j] = get_reconstruction_error(x, x_rec, self.sr, self.bands, self.band_trims)

                if np.max(error) > total_error[-1]:
                    total_error.append(np.max(error))
                    idx = np.insert(idx, 0, np.array(idx_excluded)[np.argmax(error)])
                    idx_excluded = np.array(idx_excluded)[
                        np.arange(len(idx_excluded)) != np.argmax(error)
                    ]
                    idx_all.append(idx)
                    idx_excluded = list(idx_excluded)
                    i = i + 1
                    num_modes.append(i)
                else:
                    flag_float = False
                num_float += 1
            i -= 1

        num_modes = np.array(num_modes)
        return (idx_all, total_error, num_modes)


    def _SFS(
        self,
        soln,
        x,
        N
    ):
        """
        Sequential methods: SFS and SFFS, described here:
        http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/

        Global modulation methods include exact, and grad, described in the paper

        Parameters
        ----------
        soln : solutions with shape (num modes, num chan, time)
        x : true data with shape (num chan, time)
        N : amount of averaging to do on global modulation f

        Returns
        -------
        idx_all : list of idxs used in feature selection
        total_error : list of errors for each step
        num_modes : number of modes for each step
        """
        if self.final_num is None or self.final_num > soln.shape[0] or self.final_num <= 0:
            self.final_num = soln.shape[0]

        idx = []
        idx_used = np.arange(soln.shape[0])
        total_error = []
        idx_all = []
        num_modes = []

        termination_num = self.final_num + 1
        i = 1
        i_last = 0

        while termination_num != i:
            if self.verbose:
                print(
                    "Number of modes: "
                    + str(i)
                    + "/"
                    + str(soln.shape[0])
                    + ", stopping at "
                    + str(self.final_num)
                )
            if i_last == i - 1:
                num_float = 0
                i_last = i

            error = np.empty(len(idx_used))
            for j, r in enumerate(idx_used):
                idx_sub = np.array(idx + [r])
                if self.f_method == "grad":
                    f_hat = grad_f(
                        x,
                        soln[idx_sub],
                        self.grad_alpha,
                        self.grad_beta,
                        N,
                        self.grad_lr,
                        self.grad_momentum,
                        self.maxiter,
                        self.grad_init_lowpass,
                        self.grad_fit_coupling,
                        self.grad_delay,
                    )
                    f_hat = grad_f_amp(f_hat, soln[idx_sub], x)
                else:
                    B_sub, f_sub = exact_Bf(x, soln[idx_sub])
                    f_hat = exact_f_from_Bf(B_sub, f_sub, N, var_thresh=self.exact_var_thresh)

                x_rec = get_reconstruction(soln[idx_sub], f_hat)
                error[j] = get_reconstruction_error(x, x_rec, self.sr, self.bands, self.band_trims)

            total_error.append(np.max(error))
            idx = idx + [idx_used[np.argmax(error)]]
            idx_used = idx_used[np.arange(len(idx_used)) != np.argmax(error)]
            idx_all.append(np.array(idx))
            num_modes.append(i)

            flag_float = True
            while self.floating and flag_float and i > 1 and num_float != self.maxiter_float:
                error = np.empty(len(idx))
                for j, r in enumerate(np.array(idx)):
                    idx_sub = np.array(idx)[np.array(idx) != r]
                    if self.f_method == "grad":
                        f_hat = grad_f(
                            x,
                            soln[idx_sub],
                            self.grad_alpha,
                            self.grad_beta,
                            N,
                            self.grad_lr,
                            self.grad_momentum,
                            self.maxiter,
                            self.grad_init_lowpass,
                            self.grad_fit_coupling,
                            self.grad_delay,
                        )
                        f_hat = grad_f_amp(f_hat, soln[idx_sub], x)
                    else:
                        B_sub, f_sub = exact_Bf(x, soln[idx_sub])
                        f_hat = exact_f_from_Bf(
                            B_sub, f_sub, N, var_thresh=self.exact_var_thresh
                        )

                    x_rec = get_reconstruction(soln[idx_sub], f_hat)
                    error[j] = get_reconstruction_error(x, x_rec, self.sr, self.bands, self.band_trims)

                if np.max(error) > total_error[-1]:
                    total_error.append(np.max(error))
                    idx_used = np.insert(idx_used, 0, idx[np.argmax(error)])
                    idx = np.array(idx)[np.arange(len(idx)) != np.argmax(error)]
                    idx_all.append(idx)
                    idx = list(idx)
                    i = i - 1
                    num_modes.append(i)
                else:
                    flag_float = False
                num_float += 1
            i += 1

        num_modes = np.array(num_modes)
        return (idx_all, total_error, num_modes)


def guess_best_fit_idx(num_modes, errors, alpha=0.0000001):
    """
    Tries to guess the best number of modes based on cosine distance
    Adds a penalty to larger number of modes

    Parameters
    ----------
    num_modes : number of modes for each error
    errors : errors
    alpha : amount of penalty for large number of modes

    Returns
    -------
    num : the best number of modes, as chosen by this method
    """
    errors_reg = 1 - np.array(errors) + np.array([alpha * i for i in num_modes])
    num = num_modes[np.argmin(errors_reg)]
    return num


###################### Gradient Descent


def grad_f_init(x, soln, beta, low_pass=None):
    """
    Finds the initial guess for f based on the gradient descent method

    Parameters
    ----------
    x : original data matrix with shape (number of channels, time)
    soln : solutions with shape (number of modes, number of channels, time)
    beta : array indicating strength of temporal smoothing
    low_pass : either None (indicating no lowpass of initial guess) or a number greater than 0 indicating
               at what value to low pass filter the initial guess.

    Returns
    -------
    f_init : intial guess for f
    """
    f_init = np.empty((soln.shape[0], soln.shape[-1]))
    for r in range(soln.shape[0]):
        for t in range(soln.shape[2]):
            top = soln[r, :, t] @ x[:, t]
            bot = ((soln[r, :, t] @ soln[r, :, t])) + np.sum(beta)
            f_init[r, t] = top / bot
    
    if low_pass is not None:
        assert low_pass>0, 'Initial low_pass is not greater than 0'
        # Mirrors data before low pass filtering data
        f_init_long = np.empty((f_init.shape[0], 3*f_init.shape[1]))
        f_init_long[:,:f_init.shape[1]] = f_init[:,::-1]
        f_init_long[:,f_init.shape[1]:2*f_init.shape[1]] = f_init[:,::1]
        f_init_long[:,2*f_init.shape[1]:] = f_init[:,::-1]
        f_init = utils.butter_pass_filter(f_init_long, low_pass, 1000, 'low')[:,f_init.shape[1]:2*f_init.shape[1]]
        f_init = grad_f_amp(f_init, soln, x)
            
    return f_init


def grad_f_grad_loss(f, x, soln, alpha, beta, N):
    """
    Finds the gradient of the loss function in the gradient descent method

    Parameters
    ----------
    f : current guess of f with shape (number of modes, time) OR (number of modes, number of channels, time)
    x : original data with shape (number of channels, time)
    soln : solutions with shape (number of modes, number of channels, time)
    alpha : number indicating strength of l1 regularization
    beta : array of length 2N+1 indicating strength of temporal smoothing
        reflection is used to mitigate boundary effects
    N : number of timepoints to smooth over

    Returns
    -------
    dLdf : gradient of loss function
    """
    if len(f.shape) == 3:
        Y = np.einsum("ijk,ijk->jk", soln, f)
        f_mean = np.mean(f, axis=1)
    else:
        Y = np.einsum("ijk,ik->jk", soln, f)
        f_mean = f
    Y2 = Y - x
    l2_term = np.einsum("ijk,jk->ik", soln, Y2)

    alpha_term = np.ones((f_mean.shape)) * alpha
    alpha_term[f_mean < 0] = -alpha_term[f_mean < 0]
        
    beta_term = f_mean * np.sum(beta)
    f_mean_ex = np.empty((f_mean.shape[0], f_mean.shape[1]+2*N))
    f_mean_ex[:,:N] = f_mean[:,N:0:-1]
    f_mean_ex[:,N:-N] = f_mean
    f_mean_ex[:,-N:] = f_mean[:,-1:-N-1:-1]
    
    for i in range(len(beta_term)):
        beta_term[i] = beta_term[i] - np.convolve(beta, f_mean_ex[i], mode='valid')

    dLdf = l2_term + alpha_term + beta_term
    return dLdf

def grad_f(x, soln, alpha, beta, N, lr, momentum, maxiter, init_lowpass=None, fit_coupling=False, t_delay=None):
    """
    Performs gradient descent to approximate f

    Parameters
    ----------
    x : original data matrix with shape (number of channels, time)
    soln : solutions with shape (number of modes, number of channels, time)
    alpha : number indicating strength of l1 regularization
    beta : number or array with length 2N+1 indicating strength of temporal smoothing
        if single number, beta will turn into array with length 2N+1
    N : number of timepoints to smooth over
    lr : learning rate
    momentum : amount of momentum to include
    maxiter : total number of iterations
    init_lowpass : value of initial guess lowpass (or None if no low pass)
    fit_coupling : flag telling whether or not to fit time delays with individual channels
    t_delay : time delays of each channel with shape (number of modes, number of channels)

    Returns
    -------
    f : approximation of global modulation f
    """
    if not (
        type(beta) == np.float64
        or type(beta) == np.int64
        or type(beta) == float
        or type(beta) == int
    ):
        assert len(beta) == 2*N+1, "Length of beta term not correct"
    else:
        beta = np.ones(2*N+1) * beta
    
    f = grad_f_init(x, soln, beta, init_lowpass)
    f[f < 0] = 0
    if fit_coupling:
        idx = t_delay[:, :, None] + np.arange(f.shape[1])[None, None, :]
        idx[idx < 0] = 0
        idx[idx >= f.shape[1]] = f.shape[1] - 1
        f_3D = np.empty((soln.shape))

    dLdf_old = 0
    for k in range(maxiter):
        if fit_coupling:
            for i in range(t_delay.shape[0]):
                for j in range(t_delay.shape[1]):
                    f_3D[i, j] = f[i, idx[i, j]]
            dLdf = grad_f_grad_loss(f_3D, x, soln, alpha, beta, N)
        else:
            dLdf = grad_f_grad_loss(f, x, soln, alpha, beta, N)

        f = f - lr * (dLdf + momentum * dLdf_old)
        f[f < 0] = 0
        f[:, N:-N] = utils.moving_average_dim(f, 2 * N + 1, -1)
        f[:, :N] = np.mean(f[:, :N], axis=-1)[:, None]
        f[:, -N:] = np.mean(f[:, -N:], axis=-1)[:, None]
        dLdf_old = dLdf
        
    if fit_coupling:
        for i in range(t_delay.shape[0]):
            for j in range(t_delay.shape[1]):
                f_3D[i, j] = f[i, idx[i, j]]
        return f_3D
    else:
        return f


def grad_f_amp(f, soln, x):
    """
    Fixes overall amplitude of f hat in the gradient descent method

    Parameters
    ----------
    f : computed solution for f with shape (number of modes, time) OR (number of modes, num of chans, time)
    soln : solutions with shape (number of modes, number of channels, time)
    x : data matrix with shape (number of channels, time)

    Returns
    -------
    f_hat : global modulation f with fixed amplitude
    """
    if len(f.shape) == 3:
        norm, _, _, _ = np.linalg.lstsq(
            (f * soln).reshape((len(f), -1)).T, x.reshape((-1)), rcond=None
        )
        f_hat = f * norm[:, None, None]
    else:
        norm, _, _, _ = np.linalg.lstsq(
            (f[:, None, :] * soln).reshape((len(f), -1)).T, x.reshape((-1)), rcond=None
        )
        f_hat = f * norm[:, None]
    f_hat[f_hat < 0] = 0
    return f_hat


###################### Reconstruction


def get_reconstruction(soln, f):
    """
    Reconstructs x from the S and f

    Parameters
    ----------
    soln : solutions with shape (number of modes, number of channels, time)
    f : global modulation f with shape (number of modes, time) OR (num modes, num chans, time)

    Returns
    -------
    x_rec : reconstructed data matrix
    """
    if len(f.shape) == 3:
        x_rec = np.sum(soln * f, axis=0)
    else:
        x_rec = np.sum(soln * f[:, None, :], axis=0)
    return x_rec


def get_reconstruction_error(x_true, x_rec, sr=1000, bands=None, trim=None):
    """
    Calculates the reconstruction error from the original and reconstructed data matricies

    Parameters
    ----------
    x_true : true data matrix
    x_rec : reconstructed data matrix
    sr : sampling rate of data
    bands : if list of bands, will calculate the cosine distance for each individual band and take the mean
    trim : amount of data to trim off after bandpassing the data

    Returns
    -------
    error : error computed with cosine distance metric
    """
    # error = utils.cos_dist(x_rec.reshape((-1)), x_true.reshape((-1)))
    if bands is None:
        error = utils.cos_dist(x_rec.reshape((-1)), x_true.reshape((-1)))
    else:
        bands = np.array(bands)
        assert len(bands.shape)==2
        assert bands.shape[1]==2
        
        error_b = np.empty(len(bands))
        for i,bp in enumerate(bands):
            x_true_b = _bandpass_x(x_true.copy(), sr, bp[0], bp[1], bp_filter='chebyshev', trim=None)
            x_rec_b = _bandpass_x(x_rec.copy(), sr, bp[0], bp[1], bp_filter='chebyshev', trim=None)
            error_b[i] = utils.cos_dist(x_rec_b.reshape((-1)), x_true_b.reshape((-1)))
        error = np.mean(error_b)
    return error


###################Implicit Functions


def _group_by_polarity(x1, x2, dtype, thresh=None):
    """
    forms groups were the com (comparisons) are comparing objects with opposite polarity (e.g. +/- freq)
    """
    if dtype == "freq_pol":
        com = np.abs(x1 + x2)
        if thresh == None:
            thresh = 0.05
    elif dtype == "phi_amp_pol":
        com = np.empty((len(x1)))
        for i in range(len(x1)):
            com[i] = 1 - utils.cos_dist(x1[i], x2[i])
        if thresh == None:
            thresh = 0.02
    elif dtype == "phi_angle_pol":
        diff = (x1 - x2 + np.pi)%(2*np.pi) - np.pi
        com = np.sqrt(np.sum(diff**2, axis=1)) / diff.shape[1]
        if thresh == None:
            thresh = 0.05
    return np.argwhere(com > thresh)[:, 0]


def _group_by_freq_phi(freq, phi_amp, phi_phase, thresh_freq=0.2, thresh_phi_amp=0.95, thresh_phi_phase=0.05):
    """
    forms groups where the com (comparisons) are comparing consecutive things
    """
    groups = []

    group_num = -1
    for i in range(len(freq) - 1):
        if group_num == -1:
            if (
                np.abs(freq[i + 1] - freq[i]) < thresh_freq
                and utils.cos_dist(phi_amp[i], phi_amp[i + 1]) > thresh_phi_amp
                and np.sqrt(np.sum(phi_phase[i]**2))/len(phi_phase[i]) < thresh_phi_phase
            ):
                groups.append([i, i + 1])
                in_group = True
            else:
                groups.append([i])
                in_group = False
            group_num += 1
        else:
            if (
                np.abs(freq[i + 1] - freq[i]) < thresh_freq
                and utils.cos_dist(phi_amp[i], phi_amp[i + 1]) > thresh_phi_amp
                and np.sqrt(np.sum(phi_phase[i]**2))/len(phi_phase[i]) < thresh_phi_phase
            ):
                if in_group:
                    groups[group_num].append(i + 1)
                else:
                    groups.append([i, i + 1])
                    group_num += 1
                    in_group = True
            else:
                if in_group:
                    in_group = False
                else:
                    groups.append([i])
                    group_num += 1
    if not in_group:
        groups.append([i + 1])
    return groups


######################### Development/doesn't currently work


def find_lag(x, f, soln, t_delay, periods, N, edge_len):
    errors = np.empty((soln.shape[0], 2 * N + 1, soln.shape[1]))
    for i in range(soln.shape[0]):
        for j in np.arange(-N, N + 1):
            f_temp = f.copy()
            for k in range(soln.shape[1]):
                idx = (
                    int(round(j * periods[i]))
                    + t_delay[i, k]
                    + np.arange(soln.shape[2])
                )
                idx[idx < 0] = 0
                idx[idx >= soln.shape[2]] = soln.shape[2] - 1
                f_temp[i, k] = f[i, k, idx]
            x_rec = nsdmd.get_reconstruction(
                soln[:, :, edge_len:-edge_len], f_temp[:, :, edge_len:-edge_len]
            )
            errors[i, j] = np.mean((x_rec - x[:, edge_len:-edge_len]) ** 2, axis=1)
    return errors
