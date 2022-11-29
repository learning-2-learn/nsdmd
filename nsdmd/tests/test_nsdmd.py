import numpy as np
from scipy.signal import welch
from nsdmd.nsdmd import opt_dmd_win
from nsdmd.nsdmd import opt_dmd_with_bandpass
from nsdmd.nsdmd import _bandpass_x
from nsdmd.nsdmd import _bandpass_guess
from nsdmd.nsdmd import _bandpass_exclude
from nsdmd.nsdmd import group_by_similarity
from nsdmd.nsdmd import get_red_init
from nsdmd.nsdmd import feature_init_remove
from nsdmd.nsdmd import get_phi_init
from nsdmd.nsdmd import get_soln
from nsdmd.nsdmd import exact_Bf
from nsdmd.nsdmd import exact_f_from_Bf
from nsdmd.nsdmd import get_reconstruction
from nsdmd.nsdmd import get_reconstruction_error
from nsdmd.nsdmd import grad_f_init
from nsdmd.nsdmd import grad_f_grad_loss
from nsdmd.nsdmd import grad_f
from nsdmd.nsdmd import grad_f_amp
from nsdmd.nsdmd import reduction
# from nsdmd.nsdmd import feature_selector
# from nsdmd.nsdmd import _SBS
# from nsdmd.nsdmd import _SFS
from nsdmd.nsdmd import guess_best_fit_idx


def test_opt_dmd_win():
    # freqs are 1,-1,2,-2
    # phi_phase is all the same
    # phi_amp in norm*np.arange(1,100)
    # windows are
    sig1 = (
        np.arange(1, 100)[:, None]
        * np.cos(np.arange(1000) * 0.001 * 2 * np.pi)[None, :]
    )  # 1 Hz
    sig2 = (
        np.arange(1, 100)[:, None]
        * np.cos(np.arange(1000) * 0.001 * 2 * np.pi * 2)[None, :]
    )  # 2 Hz
    sig = np.hstack((sig1, sig2))

    res_f, res_p, res_w = opt_dmd_win(sig, np.arange(2000) * 0.001, 500, 1000, 2)
    ans_f = np.array([[1, -1], [2, -2]])
    ans_p = np.arange(1, 100) / (np.sum(np.arange(1, 100) ** 2)) ** 0.5
    ans_w = np.array([np.arange(500), np.arange(1000, 1500)])

    assert np.allclose(res_f, ans_f, 0.001)  # freq
    assert np.allclose(
        np.abs(res_p)[:, 0, :], np.abs(res_p)[:, 1, :]
    )  # check polarity of phi
    assert np.allclose(
        np.angle(res_p)[0, 0, :], np.ones(99) * np.angle(res_p)[0, 0, 0]
    )  # phase
    assert np.allclose(np.abs(res_p)[0, 0, :], ans_p, 0.0001)  # amplitude
    assert np.allclose(res_w, ans_w)  # windows


def test_opt_dmd_with_bandpass():
    assert False, "TODO"


def test__bandpass_x():
    x = np.random.normal(0, 1, size=(2,10000))
    x_b = _bandpass_x(x, 1000, 100, 200, trim=100)
    assert np.allclose(np.array(x_b.shape), np.array([2,9800]))
    
    f_b, p_b = welch(x_b[0], 1000)
    res_l = p_b[np.argwhere(f_b<80)[:,0]]
    res_h = p_b[np.argwhere(f_b>220)[:,0]]
    assert np.all(res_h < 0.001)
    assert np.all(res_l < 0.001)


def test__bandpass_guess():
    res = _bandpass_guess(5,10,1000)
    assert np.allclose(res[::2], -res[1::2])
    assert np.all((res[::2]>5) & (res[::2]<10))
    
    res = _bandpass_guess(5,10,2, [7,-7])
    assert np.allclose(res, np.array([7,-7]))


def test__bandpass_exclude():
    freq = 5*np.ones((4,2))
    freq[:,1] = 10
    freq[0,0] = 10

    phi = np.ones((4,2,6))
    res_f, res_p = _bandpass_exclude(freq, phi, 4, 6)
    
    ans_f = 5*np.ones((4,1))
    ans_f[0] = 0
    
    ans_p = np.ones((4,1,6))
    ans_p[0] = 0
    
    assert np.allclose(ans_f, res_f)
    assert np.allclose(ans_p, res_p)


def test_group_by_similarity():
    freqs = np.array([[1, -1], [2, -2], [2.1, -2.1], [1, -1]])
    phis = np.array(
        [
            [[1, 2, 3], [1, 2, 3]],
            [[2, 2, 2], [2, 2, 2]],
            [[3.1, 3.1, 3.1], [3.1, 3.1, 3.1]],
            [[1, 2, 3], [1, 2, 3]],
        ]
    )

    res_1 = group_by_similarity(freqs, phis, thresh_freq=4, thresh_phi_amp=0.1)
    ans_1 = [[[0, 1, 2, 3]], []]
    assert res_1 == ans_1, "Case where everything is similar"

    res_2 = group_by_similarity(freqs, phis, thresh_freq=0.5, thresh_phi_amp=0.1)
    ans_2 = [[[0], [1, 2], [3]], []]
    assert res_2 == ans_2, "Case where freq is different"

    res_3 = group_by_similarity(freqs, phis, thresh_freq=4, thresh_phi_amp=0.95)
    ans_3 = [[[0], [1, 2], [3]], []]
    assert res_3 == ans_3, "Case where phi amp is different"

    freqs = np.array([[1, -1], [1, -3]])
    phis = np.array([[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]])
    res_4 = group_by_similarity(freqs, phis, thresh_freq=4, thresh_phi_amp=0.1)
    ans_4 = [[[0, 1]], [[1]]]
    assert res_4 == ans_4, "Case where freq polarity is different"

    freqs = np.array([[1, -1], [1, -1]])
    phis = np.array([[[1, 1, 1], [2, 5, 2]], [[1, 1, 1], [2, 5, 2]]])
    res_5 = group_by_similarity(freqs, phis, thresh_freq=4, thresh_phi_amp=0.1)
    ans_5 = [[[0, 1]], [[0], [1]]]
    assert res_5 == ans_5, "Case where phi amp polarity is different"


def test_get_red_init():
    group_idx = [[[0, 1, 2], [3]], [[1]]]
    res = get_red_init(group_idx, 4)
    ans = np.array(
        [
            [np.array([0, 1, 2, 2]), np.zeros(4)],
            [np.array([3, 3, 3, 3]), np.zeros(4)],
            [np.array([1, 1, 1, 1]), np.ones(4)],
        ]
    )
    assert np.allclose(res, ans)


def test_feature_init_remove():
    assert False, "TODO"


def test_get_phi_init():
    f = np.hstack((4 * np.ones((500)), 2 * np.ones((500))))
    sr = 1000
    phi_a = np.array([1, 2, 3])
    phi = np.vstack((phi_a, phi_a, -phi_a))
    offsets = np.array((0, 250, 750))

    res = get_phi_init(f, phi, offsets, sr)
    ans = np.vstack((phi_a, phi_a, phi_a))
    assert np.allclose(res, ans)


def test_get_soln():
    freqs = np.array([[4, -4], [4, -4], [2, -2], [2, -2]])
    phi = np.array(
        [
            [[1, 2, 3], [1, 2, 3]],
            [[1, 2, 3], [1, 2, 3]],
            [[1, 2, 3], [1, 2, 3]],
            [[-1, -2, -3], [-1, -2, -3]],
        ]
    )
    idxs = np.array(
        [[np.arange(4), np.zeros(4)], [np.arange(4), np.ones(4)]], dtype=int
    )
    t_len = 1000
    N = 1
    sr = 1000
    windows = np.array(
        [np.arange(250), np.arange(250, 500), np.arange(500, 750), np.arange(750, 1000)]
    )

    res = get_soln(freqs, phi, idxs, t_len, windows, N, sr)
    ans = np.array(
        [
            [[1.0, 1.0, -1.0], [2.0, 2.0, -2.0], [3.0, 3.0, -3.0]],
            [[1.0, 1.0, -1.0], [2.0, 2.0, -2.0], [3.0, 3.0, -3.0]],
        ]
    )

    np.allclose(ans, res[0][:, :, np.array([0, 250, 750])])

    ans = np.ones((2, 1000))
    ans[0, :500] = 4
    ans[0, 500:] = 2
    ans[1, :500] = -4
    ans[1, 500:] = -2
    assert np.allclose(res[1], ans)

    ans = np.ones((2, 1000, 3))
    ans[:, :, 1] = 2
    ans[:, :, 2] = 3
    assert np.allclose(res[2], ans)


def test_exact_Bf():
    x1 = np.arange(1, 2)[:, None] * np.cos(np.arange(4) * 0.001 * 2 * np.pi)[None, :]
    x2 = (
        np.arange(2, 1, -1)[:, None] * np.cos(np.arange(4) * 0.001 * 2 * np.pi)[None, :]
    )
    x = x1 + x2
    s = np.vstack((x1[None, :, :], x2[None, :, :]))

    res_B, res_f = exact_Bf(x, s)
    ans_B = np.array(
        [[[1, 1, 1, 1], [2, 2, 2, 2]], [[0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1]]]
    )
    ans_f = np.array([[3, 3, 3, 3], [1.5, 1.5, 1.5, 1.5]])

    assert np.allclose(res_B, ans_B), "B is wrong"
    assert np.allclose(res_f, ans_f), "f is wrong"


def test_exact_f_from_Bf():
    B = np.array([[[2], [1]], [[1], [2]]])
    f = np.array([[4], [5]])
    N = 1
    ans = np.array([[1], [2]])

    res = exact_f_from_Bf(B, f, N)
    assert np.allclose(ans, res)


def test_get_reconstruction():
    x1 = np.arange(1, 3)[:, None] * np.cos(np.arange(400) * 0.001 * 2 * np.pi)[None, :]
    x2 = (
        np.arange(3, 1, -1)[:, None]
        * np.cos(np.arange(400) * 0.001 * 2 * np.pi)[None, :]
    )
    x = x1 + x2
    s = np.vstack((x1[None, :, :], x2[None, :, :]))
    f = np.ones((2, 400))

    ans = x
    res = get_reconstruction(s, f)
    assert np.allclose(ans, res), "f is 2 dim"

    f = np.ones((2, 2, 400))

    ans = x
    res = get_reconstruction(s, f)
    assert np.allclose(ans, res), "f is 3 dim"


def test_get_reconstruction_error():
    x = np.array([[1, 1], [1, 1]])
    y = np.array([[2, 2], [2, 2]])

    ans = 1
    res = get_reconstruction_error(x, y)
    assert ans == res


def test_grad_f_init():
    x1 = np.arange(1, 2)[:, None] * np.ones(4)[None, :]
    x2 = np.arange(2, 1, -1)[:, None] * np.ones(4)[None, :]
    x = x1 + x2
    s = np.vstack((x1[None, :, :], x2[None, :, :]))
    beta = 1
    N = 1

    res = grad_f_init(x, s, beta, N)
    ans = np.array([[1.5, 1.5, 1.5, 1.5], [1.2, 1.2, 1.2, 1.2]])

    assert np.allclose(res, ans)


def test_grad_f_grad_loss():
    x1 = np.arange(1, 2)[:, None] * np.ones(4)[None, :]
    x2 = np.arange(2, 1, -1)[:, None] * np.ones(4)[None, :]
    x = x1 + x2
    s = np.vstack((x1[None, :, :], x2[None, :, :]))
    f = np.ones((2, 4)) / 2

    res_l2 = grad_f_grad_loss(f, x, s, 0, 0, 1)
    ans_l2 = np.array([[-1.5, -1.5, -1.5, -1.5], [-3, -3, -3, -3]])
    assert np.allclose(res_l2, ans_l2), "L2 problem"

    f = np.ones((2, 1, 4)) / 2

    res_l2 = grad_f_grad_loss(f, x, s, 0, 0, 1)
    ans_l2 = np.array([[-1.5, -1.5, -1.5, -1.5], [-3, -3, -3, -3]])
    assert np.allclose(res_l2, ans_l2), "L2 problem (3 dimensional f)"

    x1 = np.arange(1, 2)[:, None] * np.ones(4)[None, :]
    x2 = np.arange(2, 1, -1)[:, None] * np.ones(4)[None, :]
    x = x1 + x2
    s = np.vstack((x1[None, :, :], x2[None, :, :]))
    f = np.ones((2, 4))

    res_alpha = grad_f_grad_loss(f, x, s, 1, 0, 1)
    ans_alpha = np.ones((2, 4))
    assert np.allclose(res_alpha, ans_alpha), "alpha problem, may also be L2 problem"

    x1 = np.arange(1, 2)[:, None] * np.ones(6)[None, :]
    x2 = np.arange(2, 1, -1)[:, None] * np.ones(6)[None, :]
    x = (x1 + x2) * np.arange(6)[None, :]
    s = np.vstack((x1[None, :, :], x2[None, :, :]))
    f = np.ones((2, 6)) * np.arange(6)[None, :]

    res_beta1 = grad_f_grad_loss(f, x, s, 0, 1, 0)
    res_beta2 = grad_f_grad_loss(f, x, s, 0, 1, 2)
    ans_beta1 = np.zeros((2, 6))
    ans_beta2 = np.array([[-3, -2, 0, 0, 2, 3], [-3, -2, 0, 0, 2, 3]])
    assert np.allclose(
        res_beta1, ans_beta1
    ), "trivial beta problem, may also be L2 problem"
    assert np.allclose(res_beta2, ans_beta2), "beta problem, may also be L2 problem"


def test_grad_f():
    x1 = np.arange(1, 3)[:, None] * np.cos(np.arange(400) * 0.001 * 2 * np.pi)[None, :]
    x2 = (
        np.arange(3, 1, -1)[:, None]
        * np.cos(np.arange(400) * 0.001 * 2 * np.pi * 2)[None, :]
    )
    x = x1 + x2
    s = np.vstack((x1[None, :, :], x2[None, :, :]))

    res = grad_f(x, s, 0, 0.1, 20, 0.1, 1000)
    ans = np.ones((2, 400))
    assert np.allclose(res, ans, 0.001), "Problem with gradient descent"

    delay = np.array([[0, 0], [0, 0]])
    res = grad_f(x, s, 0, 0.1, 20, 0.1, 1000, True, delay)
    ans = np.ones((2, 400))
    assert np.allclose(res, ans, 0.001), "Problem with gradient descent with delays"


def test_grad_f_amp():
    x1 = np.arange(1, 3)[:, None] * np.cos(np.arange(400) * 0.001 * 2 * np.pi)[None, :]
    x2 = (
        np.arange(3, 1, -1)[:, None]
        * np.cos(np.arange(400) * 0.001 * 2 * np.pi * 2)[None, :]
    )
    x = x1 + x2
    s = np.vstack((x1[None, :, :], x2[None, :, :]))
    f = np.ones((2, 400)) / 2

    res = grad_f_amp(f, s, x)
    ans = np.ones((2, 400))
    assert np.allclose(res, ans), "Problem with amplitude fixes (2 dim)"

    f = np.ones((2, 2, 400)) / 2

    res = grad_f_amp(f, s, x)
    ans = np.ones((2, 2, 400))
    assert np.allclose(res, ans), "Problem with amplitude fixes (3 dim)"


def test_feature_selector():
    assert False, "TODO"


def test__SBS():
    x1 = np.arange(1, 2)[:, None] * np.cos(np.arange(4) * 0.001 * 2 * np.pi)[None, :]
    x2 = (
        np.arange(2, 1, -1)[:, None] * np.cos(np.arange(4) * 0.001 * 2 * np.pi)[None, :]
    )
    x = x1 + x2
    s = np.vstack((x1[None, :, :], x2[None, :, :]))

    res_i, res_e, res_n = reduction._SBS(s, x, "exact", 1, verbose=False)
    ans_n = np.array([2, 1])

    assert np.allclose(res_i[0], np.array([0, 1])), "indexing issue"
    assert len(res_i[1]) == 1, "indexing issue"
    assert res_i[1][0] == 0 or res_i[1][0] == 1, "indexing issue"
    assert np.allclose(res_e, np.ones((2)), 0.0001), "error issue"
    assert np.allclose(ans_n, res_n)

    assert False, "TODO floating test"


def test__SFS():

    assert False, "TODO"


def test_guess_best_fit_idx():
    assert False, "TODO"
