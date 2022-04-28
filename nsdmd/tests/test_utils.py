import numpy as np
import math
from scipy.signal import welch
from scipy.stats import linregress
from nsdmd.utils import cos_dist
from nsdmd.utils import demean_mat
from nsdmd.utils import make_network
from nsdmd.utils import create_decay
from nsdmd.utils import add_noise
from nsdmd.utils import moving_average_dim
from nsdmd.utils import butter_pass_filter
from nsdmd.utils import _butter_pass


def test_butter_pass_filter():
    x = np.random.normal(0, 1, size=10000)
    x_h = butter_pass_filter(x, 100, 1000, 'high')
    x_l = butter_pass_filter(x, 100, 1000, 'low')
    f_h, p_h = welch(x_h, 1000)
    f_l, p_l = welch(x_l, 1000)
    res_h = p_h[np.argwhere(f_h<80)[:,0]]
    ans_h = np.zeros(np.sum(f_h<80))
    res_l = p_l[np.argwhere(f_l>120)[:,0]]
    ans_l = np.zeros(np.sum(f_l>120))
    assert np.all(res_h < 0.001)
    assert np.all(res_l < 0.001)

def test__butter_pass():
    bh, ah = _butter_pass(250,1000, 'high', order=1)
    bl, al = _butter_pass(250,1000, 'low', order=1)
    ans_bh = np.array([0.5,-0.5])
    ans_bl = np.array([0.5,0.5])
    ans_ah = np.array([1,0])
    ans_al = np.array([1,0])
    assert np.allclose(np.array([bh, ah, bl, al]), np.array([ans_bh, ans_ah, ans_bl, ans_al]))

def test_cos_dist():
    ans_1 = 0.0
    res_1 = cos_dist(np.array([0, 0, 1]), np.array([0, 1, 0]))
    assert ans_1 == res_1

    ans_2 = 1.0
    res_2 = cos_dist(np.array([1, 1, 2]), np.array([2, 2, 4]))
    assert math.isclose(ans_2, res_2)


def test_demean_mat():
    mat = (
        np.arange(0, 10, 1)[:, None, None]
        + np.arange(0, 100, 20)[None, :, None]
        + np.ones(3)[None, None, :]
    )
    ans = np.zeros((mat.shape))
    res = demean_mat(mat)
    assert np.all(ans == res)


def test_make_network():
    # All tests are either with default settings or modified default settings
    f = 4
    t_len = 1000
    sr = 1000
    phi_a = np.array([1, 2, 3])
    phi_p = np.array([0, 30, -30])

    res_n, res_f, res_c = make_network(f, t_len, phi_a, phi_p, sr)
    assert np.allclose(
        res_n[0, np.array([0, 250, 500, 750])], np.ones((4)) / (14**0.5)
    ), "regular idx 0"
    assert np.allclose(
        res_n[1, np.array([-30, 220, 470, 720])], 2 * np.ones((4)) / (14**0.5)
    ), "regular idx 1"
    assert np.allclose(
        res_n[2, np.array([30, 280, 530, 780])], 3 * np.ones((4)) / (14**0.5)
    ), "regular idx 2"

    sr = 500

    res_n, res_f, res_c = make_network(f, t_len, phi_a, phi_p, sr)
    assert np.allclose(
        res_n[0, np.array([0, 125, 250, 375])], np.ones((4)) / (14**0.5)
    ), "different time step"

    sr = 1000
    f = np.hstack((4 * np.ones(1500), 2 * np.ones(1500)))

    res_n, res_f, res_c = make_network(f, t_len, phi_a, phi_p, sr)
    assert np.allclose(
        res_n[0, np.array([0, 250, 500])], np.ones((3)) / (14**0.5)
    ), "freq array"
    assert np.allclose(
        res_n[0, np.array([750])], -np.ones((1)) / (14**0.5)
    ), "freq array"

    f = 4
    time_mod = np.arange(1000)

    res_n, res_f, res_c = make_network(f, t_len, phi_a, phi_p, sr, time_mod=time_mod)
    assert np.allclose(
        res_f[:, np.array([0, 250, 500, 750])],
        np.ones(3)[:, None] * np.array([0, 250, 500, 750])[None, :],
    ), "time_mod"

    time_mod = 0
    coupling = np.arange(3000)

    res_n, res_f, res_c = make_network(f, t_len, phi_a, phi_p, sr, coupling=coupling)
    assert np.allclose(
        res_f[:, np.array([0, 250, 500, 750])],
        np.array(
            [
                [1000, 1250, 1500, 1750],
                [1030, 1280, 1530, 1780],
                [970, 1220, 1470, 1720],
            ]
        ),
    ), "coupling, global f"
    assert np.allclose(
        res_c[np.array([0, 250, 500, 750])], np.array([1000, 1250, 1500, 1750])
    ), "coupling, c"


def test_create_decay():
    ans_1 = np.ones(10)
    res_1 = create_decay(10)
    assert np.all(ans_1 == res_1)

    ans_2 = np.array([0.5, 1.0, 1.0, 0.75, 0.25])
    res_2 = create_decay(5, 2, 3)
    assert np.all(np.isclose(ans_2, res_2))


def test_add_noise():
    x = np.ones((1, 100000))
    std = 0.1
    power = 0

    y = add_noise(x, std, power)
    f, p = welch(y[0], 1000)
    reg = linregress(np.log(f[1:-1]), np.log(p[1:-1]))
    res = reg[0]
    ans = 0
    assert np.abs(res - ans) < 0.1, "Problem with noise at power = 0"

    x = np.ones((1, 100000))
    std = 0.001
    power = -2

    y = add_noise(x, std, power)
    f, p = welch(y[0], 1000)
    reg = linregress(np.log(f[1:-1]), np.log(p[1:-1]))
    res = reg[0]
    ans = -2
    assert np.abs(res - ans) < 0.1, "Problem with noise at power = -2"


def test_moving_average_dim():
    x = np.ones(3)[:, None] * np.arange(10)[None, :]

    ans = np.ones(3)[:, None] * np.arange(1, 9)[None, :]
    res = moving_average_dim(x, 3, 1)

    assert np.allclose(ans, res)
