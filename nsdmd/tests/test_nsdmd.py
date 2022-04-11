import numpy as np
from nsdmd.nsdmd import opt_dmd_win

def test_opt_dmd_win():
    #freqs are 1,-1,2,-2
    #phi_phase is all the same
    #phi_amp in norm*np.arange(1,100)
    #windows are
    sig1 = np.arange(1,100)[:,None] * np.cos(np.arange(1000)*0.001 * 2 * np.pi)[None,:] # 1 Hz
    sig2 = np.arange(1,100)[:,None] * np.cos(np.arange(1000)*0.001 * 2 * np.pi * 2)[None,:] # 2 Hz
    sig = np.hstack((sig1, sig2))
    
    res_f, res_p, res_w = opt_dmd_win(sig, np.arange(2000)*0.001,500,1000,2)
    ans_f = np.array([[1,-1],[2,-2]])
    ans_p = np.arange(1,100) / (np.sum(np.arange(1,100)**2))**0.5
    ans_w = np.array([np.arange(500), np.arange(1000,1500)])
    
    assert np.allclose(res_f, ans_f, 0.001) # freq
    assert np.allclose(np.abs(res_p)[:,:,0], np.abs(res_p)[:,:,1]) # check polarity of phi
    assert np.allclose(np.angle(res_p)[0,:,0], np.ones(99)*np.angle(res_p)[0,0,0]) # phase
    assert np.allclose(np.abs(res_p)[0,:,0], ans_p, 0.0001) # amplitude
    assert np.allclose(res_w, ans_w) # windows