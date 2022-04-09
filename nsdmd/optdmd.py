import numpy as np
import scipy.linalg
import scipy.sparse

# This section was taken from https://github.com/shervinsahba/dmdz
# Copyright 2020 Shervin Sahba

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

def varpro2_expfun(alpha, t):
    A = t[:,np.newaxis] @ alpha[:,np.newaxis].T
    return np.exp(A)


def varpro2_dexpfun(alpha, t, j):
    # computes d/d(alpha_i) where we begin indexing at 0
    if (j < 0) or (j >= len(alpha)):
        raise ValueError("varpro2_dexpfun: cannot compute %sth derivative. Index j for d/d(alpha_j) out of range."%j)
    t = t.reshape((-1, 1))
    A = scipy.sparse.lil_matrix((t.size, alpha.size), dtype=complex)
    A[:, j] = t * np.exp(alpha[j] * t)
    return scipy.sparse.csc_matrix(A)


def varpro2_opts(set_options_dict=None):
    options = {
        "lambda0": 1.0,
        "max_lambda": 52,
        "lambda_up": 2.0,
        "lambda_down": 3.0,
        "use_marquardt_scaling": True,
        "max_iterations": 30,
        "tolerance": 1.0e-1,
        "eps_stall": 1.0e-12,
        "compute_full_jacobian": True,
        "verbose": True,
        "ptf": 1
    }
    optionsmin = {
        "lambda0": 0.0,
        "max_lambda": 0,
        "lambda_up": 1.0,
        "lambda_down": 1.0,
        "use_marquardt_scaling": False,
        "max_iterations": 0,
        "tolerance": 0.0,
        "eps_stall": -np.finfo(np.float64).min,
        "compute_full_jacobian": False,
        "verbose": False,
        "ptf": 0
    }
    optionsmax = {
        "lambda0": 1.0e16,
        "max_lambda": 200,
        "lambda_up": 1.0e16,
        "lambda_down": 1.0e16,
        "use_marquardt_scaling": True,
        "max_iterations": 1.0e12,
        "tolerance": 1.0e16,
        "eps_stall": 1.0,
        "compute_full_jacobian": True,
        "verbose": True,
        "ptf": 2147483647                 # sys.maxsize() for int datatype
    }
    if not set_options_dict:
        print("Default varpro2 options used.")
    else:
        for key in set_options_dict:
            if key in options:
                if optionsmin[key] <= set_options_dict[key] <= optionsmax[key]:
                    options[key] = set_options_dict[key]
                else:
                    warnings.warn("Value %s = %s is not in valid range (%s,%s)" %
                                  (key, set_options_dict[key], optionsmin[key], optionsmax[key]), Warning)
            else:
                warnings.warn("Key %s not in options" % key, Warning)
    return options


def varpro2(y, t, phi_function, dphi_function, alpha_init,
            linear_constraint=False,
            tikhonov_regularization=0,
            prox_operator=False,
            options=None):
    """
    :param y: data matrix
    :param t: vector of sample times
    :param phi: function phi(alpha,t) that takes matrices of size (m,n)
    :param dphi: function dphi(alpha,t,i) returning the d(phi)/d(alpha)
    :param alpha_init: initial guess for vector alpha
    :param use_tikhonov_regularization: Sets L2 regularization. Zero or False will have no L2 regularization.
    Can use a scalar (gamma) or matrix: min|y - phi*b|_F^2 + |gamma alpha|_2^2
    :param prox_operator: prox operator that can be applied to vector alpha at each step
    :param options: options for varpro2
    """

    def update_alpha(alpha, rjac, rhs, djacobian_pivot, prox_operator):
        # update eigenvalues
        delta0 = scipy.linalg.lstsq(rjac, rhs)[0]
        delta0 = delta0[djacobian_pivot]
        alpha0 = alpha + delta0
        if prox_operator:
            alpha0 = prox_operator(alpha0)
            delta0 = alpha0 - alpha
        return alpha0, delta0

    def varpro2_solve(phi, y, gamma, alpha):
        # least squares solution for mode amplitudes phi @ b = y, residual, and error
        b = scipy.linalg.lstsq(phi, y)[0]
        residual = y - phi@b
        if len(alpha) == 1 or np.isscalar(alpha):
            alpha = np.ravel(alpha).item()*np.eye(*gamma.shape)
        error_last = 0.5*(np.linalg.norm(residual, 'fro')**2 + np.linalg.norm(gamma@alpha)**2)
        return b, residual, error_last

    def varpro2_svd(phi, tolrank):
        # rank truncated svd where rank is scaled by a tolerance
        U, s, Vh = np.linalg.svd(phi, full_matrices=False)
        rank = np.sum(s > tolrank*s[0])
        U = U[:, :rank]
        s = s[:rank]
        V = Vh[:rank, :].conj().T
        return U, s, V

    t = np.ravel(t)
    n_data_cols = y.shape[1]
    n_t = len(t)
    n_alpha = len(alpha_init)

    options = varpro2_opts(set_options_dict=options)
    lambda0 = options['lambda0']

    if linear_constraint:
        # TODO linear constraints functionality
        raise Exception("linear constraint functionality not yet coded!")

    if tikhonov_regularization:
        if np.isscalar(tikhonov_regularization):
            gamma = tikhonov_regularization*np.eye(n_alpha)
    else:
        gamma = np.zeros((n_alpha, n_alpha))

    if prox_operator:
        alpha_init = prox_operator(alpha_init)

    # Initialize values
    alpha = np.copy(np.asarray(alpha_init, dtype=complex))
    alphas = np.zeros((n_alpha, options['max_iterations']), dtype=complex)
    if tikhonov_regularization:
        djacobian = np.zeros((n_t*n_data_cols + n_alpha, n_alpha), dtype=complex)
        rhs_temp = np.zeros(n_t*n_data_cols + n_alpha, dtype=complex)
        raise Exception("Tikhonov part not coded")
    else:
        djacobian = np.zeros((n_t*n_data_cols, n_alpha), dtype=complex)
        rhs_temp = np.zeros(n_t*n_data_cols, dtype=complex)
    error = np.zeros(options['max_iterations'])
    # res_scale = np.linalg.norm(y, 'fro')      # TODO res_scale unused in Askham's MATLAB code. Ditch it?
    scales = np.zeros(n_alpha)
    rjac = np.zeros((2*n_alpha, n_alpha), dtype=complex)

    phi = phi_function(alpha, t)
    tolrank = n_t*np.finfo(float).eps
    U, s, V = varpro2_svd(phi, tolrank)
    b, residual, error_last = varpro2_solve(phi, y, gamma, alpha)

    for iteration in range(options['max_iterations']):
        # build jacobian matrix by looping over alpha indices
        for j in range(n_alpha):
            dphi_temp = dphi_function(alpha, t, j)  # d/(dalpha_j) of phi. sparse output.
            sp_U = scipy.sparse.csc_matrix(U)
            djacobian_a = (dphi_temp - sp_U @ (sp_U.conj().T @ dphi_temp)).todense() @ b
            if options['compute_full_jacobian']:
                djacobian_b = U@scipy.linalg.lstsq(np.diag(s), V.conj().T @ dphi_temp.conj().T.todense() @ residual)[0]
                djacobian[:n_t*n_data_cols, j] = djacobian_a.ravel(order='F') + djacobian_b.ravel(order='F')
            else:
                djacobian[:n_t*n_data_cols, j] = djacobian_a.A.ravel(order='F')  # approximate Jacobian
            if options['use_marquardt_scaling']:
                scales[j] = min(np.linalg.norm(djacobian[:n_t*n_data_cols, j]), 1.0)
                scales[j] = max(scales[j], 1e-6)
            else:
                scales[j] = 1.0

        if tikhonov_regularization:
            print("using tikhonov regularization")
            djacobian[n_t*n_data_cols + 1:, :] = gamma

        # loop to determine lambda for the levenberg part
        # precompute components that don't depend on step-size parameter lambda
        # get pivots and lapack style qr for jacobian matrix
        rhs_temp[:n_t*n_data_cols] = residual.ravel(order='F')

        if tikhonov_regularization:
            rhs_temp[n_t*n_data_cols:] = -gamma@alpha

        g = djacobian.conj().T@rhs_temp

        djacobian_Q, djacobian_R, djacobian_pivot = scipy.linalg.qr(djacobian, mode='economic',
                                                                    pivoting=True)  # TODO do i need householder reflections?
        rjac[:n_alpha, :] = np.triu(djacobian_R[:n_alpha, :])
        rhs_top = djacobian_Q.conj().T@rhs_temp
        rhs = np.concatenate((rhs_top[:n_alpha], np.zeros(n_alpha)), axis=0)

        scales_pivot = scales[djacobian_pivot]
        rjac[n_alpha:2*n_alpha, :] = lambda0*np.diag(scales_pivot)

        alpha0, delta0 = update_alpha(alpha, rjac, rhs, djacobian_pivot, prox_operator)
        phi = phi_function(alpha0, t)
        b0, residual0, error0 = varpro2_solve(phi, y, gamma, alpha0)

        # update rule
        actual_improvement = error_last - error0
        predicted_improvement = np.real(0.5*delta0.conj().T@g)
        improvement_ratio = actual_improvement/predicted_improvement

        descent = " "  # marker that indicates in output whether the algorithm needed to enter the descent loop
        if error0 < error_last:
            # rescale lambda based on actual vs pred improvement
            lambda0 = lambda0*max(1/options['lambda_down'], 1 - (2*improvement_ratio - 1)**3)
            alpha, error_last, b, residual = (alpha0, error0, b0, residual0)
        else:
            # increase lambda until something works. kinda like gradient descent
            descent = "*"
            for j in range(options['max_lambda']):
                lambda0 = lambda0*options['lambda_up']
                rjac[n_alpha:2*n_alpha, :] = lambda0*np.diag(scales_pivot)

                alpha0, delta0 = update_alpha(alpha, rjac, rhs, djacobian_pivot, prox_operator)
                phi = phi_function(alpha0, t)
                b0, residual0, error0 = varpro2_solve(phi, y, gamma, alpha0)
                if error0 < error_last:
                    alpha, error_last, b, residual = (alpha0, error0, b0, residual0)
                    break

            if error0 > error_last:
                error[iteration] = error_last
                convergence_message = "Failed to find appropriate step length at iteration %d. Residual %s. Lambda %s"%(
                iteration, error_last, lambda0)
                if options['verbose']:
                    warnings.warn(convergence_message, Warning)
                return b, alpha, alphas, error, iteration, (False, convergence_message)

        # update and status print
        alphas[:, iteration] = alpha
        error[iteration] = error_last
        if options['verbose'] and (iteration%options['ptf'] == 0):
            print("step %02d%s error %.5e lambda %.5e"%(iteration, descent, error_last, lambda0))

        if error_last < options['tolerance']:
            convergence_message = "Tolerance %s met"%options['tolerance']
            return b, alpha, alphas, error, iteration, (True, convergence_message)

        if iteration > 0:
            if error[iteration - 1] - error[iteration] < options['eps_stall']*error[iteration - 1]:
                convergence_message = "Stall detected. Residual reduced by less than %s times previous residual."%(
                options['eps_stall'])
                if options['verbose']:
                    print(convergence_message)
                return b, alpha, alphas, error, iteration, (True, convergence_message)
            pass

        phi = phi_function(alpha, t)
        U, s, V = varpro2_svd(phi, tolrank)

    convergence_message = "Failed to reach tolerance %s after maximal %d iterations. Residual %s"%(
    options['tolerance'], iteration, error_last)
    if options['verbose']:
        warnings.warn(convergence_message, Warning)
    return b, alpha, alphas, error, iteration, (False, convergence_message)


class SVD(object):

    def __init__(self, svd_rank=0):
        self.X = None
        self.U = None
        self.s = None
        self.V = None
        self.svd_rank = 0

    @staticmethod
    def cumulative_energy(s, normalize=True):
        cumulative_energy = np.cumsum(s)
        if normalize:
            cumulative_energy = cumulative_energy/s.sum()
        return cumulative_energy

    @staticmethod
    def gavish_donoho_rank(X, s, energy_threshold=0.999999):
        """
        Returns matrix rank for Gavish-Donoho singular value thresholding.
        Reference: https://arxiv.org/pdf/1305.5870.pdf
        """
        beta = X.shape[0]/X.shape[1]
        omega = 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43
        cutoff = np.searchsorted(SVD.cumulative_energy(s), energy_threshold)
        rank = np.sum(s > omega*np.median(s[:cutoff]))
        print("Gavish-Donoho rank is {}, computed on {} of {} "
              "singular values such that cumulative energy is {}.".format(rank, cutoff, len(s), energy_threshold))
        return rank

    @staticmethod
    def svd(X, svd_rank=0, full_matrices=False, verbose=False, **kwargs):
        """
        Computes the SVD of matrix X. Defaults to economic SVD.
        :param svd_rank: 0 for Gavish-Donoho threshold, -1 for no truncation, and
            integers [1,infinty) to attempt that truncation.
        :param full_matrices: False is the economic default.
        :return: U, s, V - note that this is V not Vh!
        See documentation for numpy.linalg.svd for more information.
        """
        U, s, V = np.linalg.svd(X, full_matrices=full_matrices, **kwargs)
        V = V.conj().T

        if svd_rank == 0:
            truncation_decision = "Gavish-Donoho"
            rank = SVD.gavish_donoho_rank(X, s)
        elif svd_rank >= 1:
            truncation_decision = "manual"
            if svd_rank < U.shape[1]:
                rank = svd_rank
            else:
                rank = U.shape[1]
                warnings.warn("svd_rank {} exceeds the {} columns of U. "
                              "Using latter value instead".format(svd_rank, U.shape[1]))
        elif svd_rank == -1:
            truncation_decision="no"
            rank = X.shape[1]

        if verbose:
            print("SVD performed with {} truncation, rank {}.".format(truncation_decision, rank))

        return U[:, :rank], s[:rank], V[:, :rank]

    def fit(self, full_matrices=False, **kwargs):
        if self.X is None:
            raise ValueError('SVD instance has no data X for SVD.X')
        else:
            self.U, self.s, self.V = self.svd(self.X, svd_rank=self.svd_rank,
                                              full_matrices=full_matrices, **kwargs)
        print("Computed SVD using svd_rank={}".format(self.svd_rank))
        

class OptDMD(object):

    def __init__(self, X, timesteps, rank, optimized_b=False):
        self.svd_X = SVD.svd(X, -1, verbose=False)  # TODO check
        self.X = X
        self.timesteps = timesteps
        self.rank = rank  # rank of varpro2 fit, i.e. number of exponentials

        self.optimized_b = optimized_b

        self.eigs = None        # DMD continuous-time eigenvalues
        self.modes = None       # DMD eigenvectors
        self.amplitudes = None  # DMD mode amplitude vector

    @property
    def amplitudes_mod(self):
        return np.abs(self.amplitudes)

    @property
    def omega(self):
        """
        Returns the continuous-time DMD eigenvalues.
        """
        return self.eigs

    @property
    def temporaldynamics(self):
        """
        :return: matrix that contains temporal dynamics of each mode, stored by row
        """
        return np.exp(np.outer(self.omega, self.timesteps - self.timesteps[0])) * self.amplitudes[:, None]

    @property
    def reconstruction(self):
        """
        Reconstruction of data matrix X and the mean square error
        """
        reconstruction = (self.modes @ self.temporaldynamics)
        abs_error = np.abs(self.X - reconstruction)
        print("X_dmd MSE {}".format(np.mean(abs_error**2)))
        return reconstruction, abs_error

    @staticmethod
    def compute_amplitudes(X, modes, optimized_b):
        if optimized_b:
            # Jovanovic et al. 2014, Sparsity-promoting dynamic mode decomposition,
            # https://hal-polytechnique.archives-ouvertes.fr/hal-00995141/document
            # TODO. For now, it will return the non-optimized code.
            b = scipy.linalg.lstsq(modes, X.T[0])[0]
        else:
            b = scipy.linalg.lstsq(modes, X.T[0])[0]
        return b

    @staticmethod
    def optdmd(X, t, r, projected=True, eigs_guess=None, U=None, verbose=True):
        if projected:
            if U is None:
                U, _, _ = np.linalg.svd(X, full_matrices=False)
                U = U[:, :r]
                if verbose:
                    print('data projection: U_r\'X')
            else:
                if verbose:
                    print('data projection: U_provided\'X')
            varpro_X = (U.conj().T@X).T
        else:
            if verbose:
                print('data projection: none, X')
            varpro_X = X.T

        if eigs_guess is None:
            def generate_eigs_guess(U, X, t, r):
                UtX = U.conj().T@X
                UtX1 = UtX[:, :-1]
                UtX2 = UtX[:, 1:]

                dt = np.ravel(t)[1:] - np.ravel(t)[:-1]
                dX = (UtX2 - UtX1)/dt
                Xin = (UtX2 + UtX1)/2

                U1, s1, Vh1 = np.linalg.svd(Xin, full_matrices=False)
                U1 = U1[:, :r]
                V1 = Vh1.conj().T[:, :r]
                s1 = s1[:r]
                Atilde = U1.conj().T@dX@V1/s1

                eigs_guess = np.linalg.eig(Atilde)[0]
                return eigs_guess

            eigs_guess = generate_eigs_guess(U, X, t, r)
            if verbose:
                print("eigs_guess: generated eigs seed for varpro2.")
        else:
            if verbose:
                print("eigs_guess: user provided eigs seed for varpro2.")

        if verbose:
            options = {"verbose" : True}
        else:
            options = {"verbose" : False}
        modes, eigs, eig_array, error, iteration, convergence_status = varpro2(varpro_X, t, varpro2_expfun,
                                                                               varpro2_dexpfun, eigs_guess, options=options)
        modes = modes.T

        # normalize
        b = np.sqrt(np.sum(np.abs(modes)**2, axis=0)).T
        indices_small = np.abs(b) < 10*10e-16*max(b)
        b[indices_small] = 1.0
        modes = modes/b
        modes[:, indices_small] = 0.0
        b[indices_small] = 0.0

        if projected:
            modes = U @ modes

        return eigs, modes, b

    def fit(self, projected=True, eigs_guess=None, U=None, verbose=True):
        if verbose:
            print("Computing optDMD on X, shape {} by {}.".format(*self.X.shape))
        self.eigs, self.modes, self.amplitudes = OptDMD.optdmd(self.X, self.timesteps, self.rank,
                                                               projected=projected, eigs_guess=eigs_guess, U=U, verbose=verbose)
        return self

    def sort_by(self, mode="eigs"):
        """
        Sorts DMD analysis results for eigenvalues, eigenvectors (modes, Phi), and amplitudes_mod (b)
        in order of decreasing magnitude, either by "eigs" or "b".
        """
        if mode == "mod_eigs" or mode == "eigs":
            indices = np.abs(self.eigs).argsort()[::-1]
        elif mode == "amplitudes_mod" or mode == "b" or mode == "amps":
            indices = np.abs(self.amplitudes_mod).argsort()[::-1]
        else:
            mode = "default"
            indices = np.arange(len(self.eigs))
        self.eigs = self.eigs[indices]
        self.modes = self.modes[:, indices]
        self.amplitudes = self.amplitudes[indices]
        print("Sorted DMD analysis by {}.".format(mode))