from numba import njit, prange
import numpy as np
import os
from scipy import io
from scipy.interpolate import interp1d
import h5py

def sample_run(pck, tneural, rot = 'pitch'):
    run = np.array([pck[k][rot] for k in range(len(pck))])
    run[0] = 0
    
    trun = np.array([pck[k]['nSamples'] for k in range(len(pck))])
    trun = trun - np.diff(trun).mean()/2
    
    crun = np.cumsum(run)
    crun_neuro = np.interp(tneural, trun, crun)
    run_neuro = np.diff(crun_neuro).astype('float32')
    
    run_neuro = np.concatenate((run_neuro, run_neuro[:1]), 0)
    return run_neuro, run, trun
    

def baselining(ops, tlag, F, Fneu):    
    F = preprocess(F, Fneu, ops['baseline'], ops['win_baseline'], ops['sig_baseline'], ops['fs'])
    #F = dcnv.preprocess(F, ops['baseline'], ops['win_baseline'], ops['sig_baseline'], 
     #                   ops['fs'], ops['prctile_baseline'])    
    
    x = np.arange(F.shape[1])
    if tlag<0:
        F[:, 1:] = (1 + tlag) * F[:, 1:] + (- tlag) * F[:, :-1]
    else:
        F[:, :-1] = (1 - tlag) * F[:, :-1] + tlag * F[:, 1:]        
    return F

@njit(['float32[:], float32[:], float32[:], int64[:], float32[:], float32[:], float32, float32'])
def oasis_trace(F, v, w, t, l, s, tau, fs):
    """ spike deconvolution on a single neuron """
    NT = F.shape[0]
    g = -1./(tau * fs)

    it = 0
    ip = 0

    while it<NT:
        v[ip], w[ip],t[ip],l[ip] = F[it],1,it,1
        while ip>0:
            if v[ip-1] * np.exp(g * l[ip-1]) > v[ip]:
                # violation of the constraint means merging pools
                f1 = np.exp(g * l[ip-1])
                f2 = np.exp(2 * g * l[ip-1])
                wnew = w[ip-1] + w[ip] * f2
                v[ip-1] = (v[ip-1] * w[ip-1] + v[ip] * w[ip]* f1) / wnew
                w[ip-1] = wnew
                l[ip-1] = l[ip-1] + l[ip]
                ip -= 1
            else:
                break
        it += 1
        ip += 1

    s[t[1:ip]] = v[1:ip] - v[:ip-1] * np.exp(g * l[:ip-1])

@njit(['float32[:,:], float32[:,:], float32[:,:], int64[:,:], float32[:,:], float32[:,:], float32, float32'], parallel=True)
def oasis_matrix(F, v, w, t, l, s, tau, fs):
    """ spike deconvolution on many neurons parallelized with prange  """
    for n in prange(F.shape[0]):
        oasis_trace(F[n], v[n], w[n], t[n], l[n], s[n], tau, fs)

def preprocess(F, Fneu, baseline, win_baseline, sig_baseline, fs):
    from scipy import ndimage
    win = int(win_baseline*fs)
    
    Flow = ndimage.gaussian_filter(F,    [0., sig_baseline])
    Flow = ndimage.minimum_filter1d(Flow,    win)
    Flow = ndimage.maximum_filter1d(Flow,    win)
    
    F = F - 0.7*Fneu      
    
    Flow2 = ndimage.gaussian_filter(F,    [0., sig_baseline])
    Flow2 = ndimage.minimum_filter1d(Flow2,    win)
    Flow2 = ndimage.maximum_filter1d(Flow2,    win)

    Fdiv = np.maximum(10, Flow.mean(1))
    F = (F - Flow2)/Fdiv[:, np.newaxis] 

    return F

def oasis(F: np.ndarray, batch_size: int, tau: float, fs: float) -> np.ndarray:
    """ computes non-negative deconvolution
    no sparsity constraints
    
    Parameters
    ----------------
    F : float, 2D array
        size [neurons x time], in pipeline uses neuropil-subtracted fluorescence
    batch_size : int
        number of frames processed per batch
    tau : float
        timescale of the sensor, used for the deconvolution kernel
    fs : float
        sampling rate per plane
    Returns
    ----------------
    S : float, 2D array
        size [neurons x time], deconvolved fluorescence
    """
    NN,NT = F.shape
    F = F.astype(np.float32)
    S = np.zeros((NN,NT), dtype=np.float32)
    for i in range(0, NN, batch_size):
        f = F[i:i+batch_size]
        v = np.zeros((f.shape[0],NT), dtype=np.float32)
        w = np.zeros((f.shape[0],NT), dtype=np.float32)
        t = np.zeros((f.shape[0],NT), dtype=np.int64)
        l = np.zeros((f.shape[0],NT), dtype=np.float32)
        s = np.zeros((f.shape[0],NT), dtype=np.float32)
        oasis_matrix(f, v, w, t, l, s, tau, fs)
        S[i:i+batch_size] = s
    return S


### Load the neural data
def load_neurons(db, data_path, dual_plane = 1, irev = [1]):    
    mname, datexp, blk = db['mname'], db['datexp'], db['blk']
    root = os.path.join(data_path, 'data/PROC', mname, datexp, blk)
    ops = np.load(os.path.join(root, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item() 
    
    if dual_plane:
        tlags = np.linspace(.2, -.8, ops['nplanes']//2+1)[:-1]
        tlags = np.hstack((tlags, tlags))
        tlags = tlags.flatten()
    else:
        tlags = np.linspace(.2, -.8, ops['nplanes']+1)[:-1]    
    print(tlags.shape)

    spks = np.zeros((0, ops['nframes']), np.float32) 
    stat = np.zeros((0,)) 
    iplane = np.zeros((0,))
    xpos, ypos = np.zeros((0,)) , np.zeros((0,))

    
    
    for n in range(ops['nplanes']):
        ops = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'ops.npy'), allow_pickle=True).item()

        stat0 = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'stat.npy'), allow_pickle=True)
        ypos0 = np.array([stat0[n]['med'][0] for n in range(len(stat0))]) 
        xpos0 = np.array([stat0[n]['med'][1] for n in range(len(stat0))]) 

        if irev[n%len(irev)]:
            ypos0 = -ypos0
        
        ypos0 += ops['dy'] 
        xpos0 += ops['dx'] 

        #### EDIT: TAU FIX 
        ops['tau'] = 0.75

        if 1: #redo the deconvoloution after shifting the traces by subsamples
            F = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'F.npy'))
            Fneu  = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'Fneu.npy'))            
            F     = baselining(ops, tlags[n], F, Fneu)            
            spks0 = oasis(F, ops['batch_size'], ops['tau'], ops['fs'])
        else:
            spks0 = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'spks.npy'), allow_pickle=True)

        spks0 = spks0.astype('float32')
        if spks.shape[1]>spks0.shape[0]:
            spks0 = np.concatenate((spks0, np.zeros((spks0.shape[0], spks.shape[1]-spks0.shape[1]))), axis=1)
        spks = np.concatenate((spks,spks0.astype('float32')), axis=0)            
        ypos = np.concatenate((ypos, ypos0), axis=0)
        xpos = np.concatenate((xpos, xpos0), axis=0)
        iplane = np.concatenate((iplane, n*np.ones(len(stat0),)))
        stat = np.concatenate((stat,stat0), axis=0)     
        
        print('plane %d, '%n, 'neurons: %d'%len(xpos0))

    print('total neurons %d'%len(spks))
    
    return spks, xpos, ypos, iplane, stat

# load the timeline
def load_exp_info(db, data_path):
    mname, datexp, blk = db['mname'], db['datexp'], db['blk']
    root = os.path.join(data_path, 'data/PROC', mname, datexp, blk)

    fname     = 'Timeline_%s_%s_%s'%(mname, datexp, blk) 
    fnamepath = os.path.join(root, fname) 

    Timeline  = io.loadmat(fnamepath, squeeze_me=True)['TL']     

    return Timeline

def loadh5py(filename): # read Timeline from .mat file
    Timeline = h5py.File(filename, 'r')['TL']
    return Timeline

def get2Pft(data):
    # data[:,0] analog signal     data[:,1] time for each batch
    if np.any(data) and np.sum(data.shape)>1000:
        ind = np.where(data[:,1]>0)[0]
        chunkTime = data[ind,1]
        frame_start_temp = (data[:-1,0] > 2.5) & (data[1:,0] <= 2.5)
        frame_start = np.where(frame_start_temp==1)[0]
        ft = np.interp(frame_start, ind, chunkTime)
    else:
        ft = []
    return ft

def event2frame(ft=[],et=[],ev=[]):
    """align events to 2p frame
        ft: frames time
        et: event time
        ev: event value
    """
    Model_ = interp1d(et, ev, fill_value='extrapolate')
    return Model_(ft)

def move2frame(ft=[],et=[],ev=[]):
    """align movement to 2p frame"""
    return np.insert(np.diff(event2frame(ft=ft,et=et,ev=ev)),0,0)