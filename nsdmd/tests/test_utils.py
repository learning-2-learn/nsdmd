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

def test_cos_dist():
    ans_1 = 0.0
    res_1 = cos_dist(np.array([0,0,1]), np.array([0,1,0]))
    assert ans_1==res_1
    
    ans_2 = 1.0
    res_2 = cos_dist(np.array([1,1,2]), np.array([2,2,4]))
    assert math.isclose(ans_2, res_2)
    
def test_demean_mat():
    mat = np.arange(0,10,1)[:,None,None] + np.arange(0,100,20)[None,:,None] + np.ones(3)[None,None,:]
    ans = np.zeros((mat.shape))
    res = demean_mat(mat)
    assert np.all(ans==res)
    
def test_make_network():
    #TODO
    assert False, "TODO"
    
def test_create_decay():
    ans_1 = np.ones(10)
    res_1 = create_decay(10)
    assert np.all(ans_1==res_1)
    
    ans_2 = np.array([0.5,1.,1.,0.75,0.25])
    res_2 = create_decay(5, 2, 3)
    assert np.all(np.isclose(ans_2, res_2))
    
def test_add_noise():
    x = np.ones((1,100000))
    std = 0.1
    power = 0

    y = add_noise(x, std, power)
    f, p = welch(y[0], 1000)
    reg = linregress(np.log(f[1:-1]), np.log(p[1:-1]))
    res = reg[0]
    ans = 0
    assert np.abs(res - ans) < 0.1, "Problem with noise at power = 0"
    
    x = np.ones((1,100000))
    std = 0.001
    power = -2

    y = add_noise(x, std, power)
    f, p = welch(y[0], 1000)
    reg = linregress(np.log(f[1:-1]), np.log(p[1:-1]))
    res = reg[0]
    ans = -2
    assert np.abs(res - ans) < 0.1, "Problem with noise at power = -2"
    
def test_moving_average_dim():
    x = np.ones(3)[:,None] * np.arange(10)[None,:]
    
    ans = np.ones(3)[:,None] * np.arange(1,9)[None,:]
    res = moving_average_dim(x, 3, 1)
    
    assert np.allclose(ans, res)
    