import numpy as np
from nsdmd.nsdmd import opt_dmd_win
from nsdmd.nsdmd import group_by_similarity
from nsdmd.nsdmd import get_soln
from nsdmd.nsdmd import get_t_delay_from_soln

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
    
def test_group_by_similarity():
    freqs = np.array([[1,-1],[2,-2],[2.1,-2.1],[1,-1]])
    phis =  np.array([[[1,1],[2,2],[3,3]],[[2,2],[2,2],[2,2]],\
                      [[3.1,3.1],[3.1,3.1],[3.1,3.1]],[[1,1],[2,2],[3,3]]])
    
    res_1 = group_by_similarity(freqs,phis,thresh_freq=4,thresh_phi_amp=0.1)
    ans_1 = [[[0,1,2,3]],[]]
    assert res_1==ans_1, 'Case where everything is similar'
    
    res_2 = group_by_similarity(freqs,phis,thresh_freq=0.5,thresh_phi_amp=0.1)
    ans_2 = [[[0],[1,2],[3]],[]]
    assert res_2==ans_2, 'Case where freq is different'
    
    res_3 = group_by_similarity(freqs,phis,thresh_freq=4,thresh_phi_amp=0.95)
    ans_3 = [[[0],[1,2],[3]],[]]
    assert res_3==ans_3, 'Case where phi amp is different'
    
    freqs = np.array([[1,-1],[1,-3]])
    phis =  np.array([[[1,1],[1,1],[1,1]],[[1,1],[1,1],[1,1]]])
    res_4 = group_by_similarity(freqs,phis,thresh_freq=4,thresh_phi_amp=0.1)
    ans_4 = [[[0,1]],[[1]]]
    assert res_4==ans_4, 'Case where freq polarity is different'
    
    freqs = np.array([[1,-1],[1,-1]])
    phis =  np.array([[[1,2],[1,5],[1,2]],[[1,2],[1,5],[1,2]]])
    res_5 = group_by_similarity(freqs,phis,thresh_freq=4,thresh_phi_amp=0.1)
    ans_5 = [[[0,1]],[[0],[1]]]
    assert res_5==ans_5, 'Case where phi amp polarity is different'
    

def test_get_soln():
    freqs = np.array([[1,-1],[1,-1]])
    phi = np.array([[[1,1],[1,1],[1,1]],[[-1,-1],[-1,-1],[-1,-1]]])
    t = np.arange(1000)*0.001
    offsets = np.array([0,.5])
    
    res = get_soln(freqs,phi,t,offsets)
    ans = np.ones((2,2,3))[:,:,:,None] * np.cos(np.arange(1000)*0.001*2*np.pi)[None,None,None,:]
    assert np.allclose(res, ans)
    
def test_get_t_delay_from_soln():
    freqs = np.array([[1,-1],[1,-1]])
    phi = np.array([[[np.exp(1j*0),np.exp(1j*0)],[np.exp(1j*.1),np.exp(1j*.1)],\
                     [np.exp(1j*.2),np.exp(1j*.2)]],\
                    [[-np.exp(1j*0),-np.exp(1j*0)],[-np.exp(1j*.1),-np.exp(1j*.1)],\
                     [-np.exp(1j*.2),-np.exp(1j*.2)]]])
    t = np.arange(1000)*0.001
    t_step = 0.001
    offsets = np.array([0,.5])
    
    res = get_t_delay_from_soln(freqs,phi,t,t_step,offsets)
    a = int(np.round(1000*0.1/2/np.pi))
    ans = np.array([[[0,a,2*a],[0,-a,-2*a]],[[0,a,2*a],[0,-a,-2*a]]])
    assert np.allclose(res, ans)

