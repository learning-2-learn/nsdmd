import numpy as np
from nsdmd.nsdmd import opt_dmd_win
from nsdmd.nsdmd import group_by_similarity
from nsdmd.nsdmd import get_soln
from nsdmd.nsdmd import get_t_delay_from_soln
from nsdmd.nsdmd import exact_Bf
from nsdmd.nsdmd import exact_f_from_Bf
from nsdmd.nsdmd import get_reconstruction
from nsdmd.nsdmd import get_reconstruction_error
from nsdmd.nsdmd import exact_f_greedy
from nsdmd.nsdmd import grad_f_init
from nsdmd.nsdmd import grad_f_grad_loss
from nsdmd.nsdmd import grad_f
from nsdmd.nsdmd import grad_f_amp

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
    assert np.allclose(np.abs(res_p)[:,0,:], np.abs(res_p)[:,1,:]) # check polarity of phi
    assert np.allclose(np.angle(res_p)[0,0,:], np.ones(99)*np.angle(res_p)[0,0,0]) # phase
    assert np.allclose(np.abs(res_p)[0,0,:], ans_p, 0.0001) # amplitude
    assert np.allclose(res_w, ans_w) # windows
    
def test_group_by_similarity():
    freqs = np.array([[1,-1],[2,-2],[2.1,-2.1],[1,-1]])
    phis =  np.array([[[1,2,3],[1,2,3]],[[2,2,2],[2,2,2]],\
                      [[3.1,3.1,3.1],[3.1,3.1,3.1]], [[1,2,3],[1,2,3]]])
    
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
    phis =  np.array([[[1,1,1],[2,5,2]],[[1,1,1],[2,5,2]]])
    res_5 = group_by_similarity(freqs,phis,thresh_freq=4,thresh_phi_amp=0.1)
    ans_5 = [[[0,1]],[[0],[1]]]
    assert res_5==ans_5, 'Case where phi amp polarity is different'
    

def test_get_soln():
    freqs = np.array([1,-1])
    phi = np.array([[1,1,1], [-1,-1,-1]])
    t = np.arange(1000)*0.001
    offsets = np.array([0,.5])
    
    res = get_soln(freqs,phi,t,offsets)
    ans = np.ones((2,3))[:,:,None] * np.cos(np.arange(1000)*0.001*2*np.pi)[None,None,:]
    assert np.allclose(res, ans)
    
def test_get_t_delay_from_soln():
    freqs = np.array([1,-1,1,-1])
    phi = np.array([[np.exp(1j*0),np.exp(1j*.1),np.exp(1j*.2)], [np.exp(1j*0),np.exp(1j*.1),np.exp(1j*.2)], \
                    [-np.exp(1j*0),-np.exp(1j*.1),-np.exp(1j*.2)], [-np.exp(1j*0),-np.exp(1j*.1),-np.exp(1j*.2)]])
    t = np.arange(1000)*0.001
    t_step = 0.001
    offsets = np.array([0,0,0.5,0.5])
    
    res = get_t_delay_from_soln(freqs,phi,t,t_step,offsets)
    a = int(np.round(1000*0.1/2/np.pi))
    ans = np.array([[0,a,2*a],[0,-a,-2*a],[0,a,2*a],[0,-a,-2*a]])
    assert np.allclose(res, ans)
    
def test_exact_Bf():
    x1 = np.arange(1,2)[:,None]*np.cos(np.arange(4)*0.001*2*np.pi)[None,:]
    x2 = np.arange(2,1,-1)[:,None]*np.cos(np.arange(4)*0.001*2*np.pi)[None,:]
    x = x1 + x2
    s = np.vstack((x1[None,:,:], x2[None,:,:]))
    
    res_B, res_f = exact_Bf(x, s)
    ans_B = np.array([[[1,1,1,1],[2,2,2,2]], [[0.5,0.5,0.5,0.5],[1,1,1,1]]])
    ans_f = np.array([[3,3,3,3],[1.5,1.5,1.5,1.5]])
    
    assert np.allclose(res_B, ans_B), "B is wrong"
    assert np.allclose(res_f, ans_f), "f is wrong"

def test_exact_f_from_Bf():
    B = np.array([[[2],[1]],[[1],[2]]])
    f = np.array([[4],[5]])
    ans = np.array([[1],[2]])
    
    res = exact_f_from_Bf(B,f)
    assert np.allclose(ans, res)
    
def test_get_reconstruction():
    x1 = np.arange(1,3)[:,None]*np.cos(np.arange(400)*0.001*2*np.pi)[None,:]
    x2 = np.arange(3,1,-1)[:,None]*np.cos(np.arange(400)*0.001*2*np.pi)[None,:]
    x = x1 + x2
    s = np.vstack((x1[None,:,:], x2[None,:,:]))
    f = np.ones((2,400))
    
    ans = x
    res = get_reconstruction(s, f)
    assert np.allclose(ans, res)
    
def test_get_reconstruction_error():
    x = np.array([[1,1],[1,1]])
    y = np.array([[2,2],[2,2]])
    
    ans = 1
    res = get_reconstruction_error(x, y)
    assert ans==res
    
def test_exact_f_greedy():
    x1 = np.arange(1,2)[:,None]*np.cos(np.arange(4)*0.001*2*np.pi)[None,:]
    x2 = np.arange(2,1,-1)[:,None]*np.cos(np.arange(4)*0.001*2*np.pi)[None,:]
    
    x = x1 + x2
    s = np.vstack((x1[None,:,:], x2[None,:,:]))
    B = np.array([[[1,1,1,1],[2,2,2,2]], [[0.5,0.5,0.5,0.5],[1,1,1,1]]])
    f = np.array([[3,3,3,3],[1.5,1.5,1.5,1.5]])
    
    res_i, res_e = exact_f_greedy(B,f,s,x,1,verbose=False)
    
    assert np.allclose(res_i[0], np.array([0,1])), "indexing issue"
    assert len(res_i[1])==1, "indexing issue"
    assert res_i[1][0]==0 or res_i[1][0]==1, "indexing issue"
    assert np.allclose(res_e, np.ones((2)), 0.0001)
    
def test_grad_f_init():
    x1 = np.arange(1,2)[:,None]*np.ones(4)[None,:]
    x2 = np.arange(2,1,-1)[:,None]*np.ones(4)[None,:]
    x = x1 + x2
    s = np.vstack((x1[None,:,:], x2[None,:,:]))
    beta = 1
    N = 1
    
    res = grad_f_init(x, s, beta, N)
    ans = np.array([[1.5,1.5,1.5,1.5], [1.2,1.2,1.2,1.2]])
    
    assert np.allclose(res, ans)
    
def test_grad_f_grad_loss():
    x1 = np.arange(1,2)[:,None]*np.ones(4)[None,:]
    x2 = np.arange(2,1,-1)[:,None]*np.ones(4)[None,:]
    x = x1 + x2
    s = np.vstack((x1[None,:,:], x2[None,:,:]))
    f = np.ones((2,4))/2
    
    res_l2 = grad_f_grad_loss(f, x, s, 0, 0, 1)
    ans_l2 = np.array([[-1.5,-1.5,-1.5,-1.5], [-3,-3,-3,-3]])
    assert np.allclose(res_l2, ans_l2), "L2 problem"
    
    x1 = np.arange(1,2)[:,None]*np.ones(4)[None,:]
    x2 = np.arange(2,1,-1)[:,None]*np.ones(4)[None,:]
    x = x1 + x2
    s = np.vstack((x1[None,:,:], x2[None,:,:]))
    f = np.ones((2,4))
    
    res_alpha = grad_f_grad_loss(f, x, s, 1, 0, 1)
    ans_alpha = np.ones((2,4))
    assert np.allclose(res_alpha, ans_alpha), "alpha problem, may also be L2 problem"
    
    x1 = np.arange(1,2)[:,None]*np.ones(6)[None,:]
    x2 = np.arange(2,1,-1)[:,None]*np.ones(6)[None,:]
    x = (x1 + x2)*np.arange(6)[None,:]
    s = np.vstack((x1[None,:,:], x2[None,:,:]))
    f = np.ones((2,6))*np.arange(6)[None,:]
    
    res_beta1 = grad_f_grad_loss(f, x, s, 0, 1, 0)
    res_beta2 = grad_f_grad_loss(f, x, s, 0, 1, 2)
    ans_beta1 = np.zeros((2,6))
    ans_beta2 = np.array([[-3,-2,0,0,2,3], [-3,-2,0,0,2,3]])
    assert np.allclose(res_beta1, ans_beta1), "trivial beta problem, may also be L2 problem"
    assert np.allclose(res_beta2, ans_beta2), "beta problem, may also be L2 problem"

def test_grad_f():
    x1 = np.arange(1,3)[:,None]*np.cos(np.arange(400)*0.001*2*np.pi)[None,:]
    x2 = np.arange(3,1,-1)[:,None]*np.cos(np.arange(400)*0.001*2*np.pi*2)[None,:]
    x = x1 + x2
    s = np.vstack((x1[None,:,:], x2[None,:,:]))
    
    res = grad_f(x, s, 0,0.1,20, 0.1, 1000)
    ans = np.ones((2,400))
    assert np.allclose(res, ans, 0.001), "Problem with gradient descent"
    
def test_grad_f_amp():
    x1 = np.arange(1,3)[:,None]*np.cos(np.arange(400)*0.001*2*np.pi)[None,:]
    x2 = np.arange(3,1,-1)[:,None]*np.cos(np.arange(400)*0.001*2*np.pi*2)[None,:]
    x = x1 + x2
    s = np.vstack((x1[None,:,:], x2[None,:,:]))
    f = np.ones((2,400))/2
    
    res = grad_f_amp(f, s, x)
    ans = np.ones((2,400))
    assert np.allclose(res, ans), "Problem with amplitude fixes"