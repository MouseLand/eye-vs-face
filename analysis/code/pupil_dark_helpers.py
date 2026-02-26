import numpy as np
import os, sys
from scipy.stats import pearsonr
import loaders
from sklearn.decomposition import PCA
from scipy.linalg import eigh

def carbachol_in_dark():
    """ if mname, sess, blk and rewType are all empty, it will output all session
        if only want first session from all all animals, sess=[1]
    """
    # all the following recordings have direction: nasal (left side of video) to temporal (right side of video)
    print('Sessions from carbachol_in_dark >>>')
    db=[]
    db.append({'sess#':1,'mname':'TX120','datexp':'2024_06_19','blk':'1','is2p':1,'rewType':'None','exptype':'baseline','Note':'3Hz'})
    db.append({'sess#':2,'mname':'TX120','datexp':'2024_06_19','blk':'3','is2p':1,'rewType':'None','exptype':'carbochol','Note':'3Hz', 'blink_baselined_threshold':'.4', 'speed_threshold':'99.5'})
    db.append({'sess#':3,'mname':'TX120','datexp':'2024_06_19','blk':'4','is2p':1,'rewType':'None','exptype':'carbochol','Note':'3Hz', 'blink_baselined_threshold':'.5', 'speed_threshold':'99.7'})
    
    db.append({'sess#':1,'mname':'TX121','datexp':'2024_06_19','blk':'1','is2p':1,'rewType':'None','exptype':'baseline','Note':'3Hz'})
    db.append({'sess#':2,'mname':'TX121','datexp':'2024_06_19','blk':'3','is2p':1,'rewType':'None','exptype':'carbochol','Note':'3Hz', 'blink_baselined_threshold':'.1', 'speed_threshold':'99.7'})
    db.append({'sess#':3,'mname':'TX121','datexp':'2024_06_19','blk':'4','is2p':1,'rewType':'None','exptype':'carbochol','Note':'3Hz', 'blink_baselined_threshold':'.1', 'speed_threshold':'99.7'})  
    
    db.append({'sess#':1,'mname':'TX123','datexp':'2024_06_20','blk':'1','is2p':1,'rewType':'None','exptype':'baseline','Note':'3Hz'})
    db.append({'sess#':2,'mname':'TX123','datexp':'2024_06_20','blk':'2','is2p':1,'rewType':'None','exptype':'carbochol','Note':'3Hz', 'blink_baselined_threshold':'1.5', 'speed_threshold':'99.5'})  
      
    db.append({'sess#':1,'mname':'LZ14','datexp':'2024_06_20','blk':'1','is2p':1,'rewType':'None','exptype':'baseline','Note':'3Hz'})
    db.append({'sess#':2,'mname':'LZ14','datexp':'2024_06_20','blk':'2','is2p':1,'rewType':'None','exptype':'carbochol','Note':'3Hz', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.5'})  
    
    db = np.array(db)
    return db

def get_light_static_info():
    # all the following recordings have direction: temporal (left side of video) to nasal (right side of video)
    print("Sessions from get_light_static_info >>>")
    db=[]
    db.append({'sess#':1,'mname':'TX143','datexp':'2024_10_28','blk':'1', 'blink_baselined_threshold':'1.5', 'speed_threshold':'99.5'})
    db.append({'sess#':2,'mname':'TX143','datexp':'2024_10_28','blk':'3'})
    
    db.append({'sess#':1,'mname':'TX139','datexp':'2024_10_31','blk':'1', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.5'})
    db.append({'sess#':2,'mname':'TX139','datexp':'2024_10_31','blk':'2'})

    db.append({'sess#':1,'mname':'LZ16','datexp':'2024_11_04','blk':'1', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.9'})
    db.append({'sess#':2,'mname':'LZ16','datexp':'2024_11_04','blk':'2'})
    
    db.append({'sess#':1,'mname':'TX140','datexp':'2024_11_06','blk':'1', 'blink_baselined_threshold':'.1', 'speed_threshold':'99.5'})
    db.append({'sess#':2,'mname':'TX140','datexp':'2024_11_06','blk':'2'})

    db.append({'sess#':1,'mname':'FXa1','datexp':'2026_01_15','blk':'1', 'blink_baselined_threshold':'.1', 'speed_threshold':'99.5'})
    db.append({'sess#':2,'mname':'FXa1','datexp':'2026_01_15','blk':'2'})

    db.append({'sess#':1,'mname':'FXa8','datexp':'2026_01_16','blk':'1', 'blink_baselined_threshold':'.1', 'speed_threshold':'99.5'})
    db.append({'sess#':2,'mname':'FXa8','datexp':'2026_01_16','blk':'2'})

    db.append({'sess#':1,'mname':'FXa10','datexp':'2026_01_16','blk':'1', 'blink_baselined_threshold':'.1', 'speed_threshold':'99.5'})
    db.append({'sess#':2,'mname':'FXa10','datexp':'2026_01_16','blk':'2'})

    #db.append({'sess#':1,'mname':'TX150','datexp':'2025_02_20','blk':'1', 'blink_baselined_threshold':'.1', 'speed_threshold':'99.5'})
    #db.append({'sess#':2,'mname':'TX150','datexp':'2025_02_20','blk':'2'})

    db = np.array(db)
    return db

def get_light_random_info():
    print("Sessions from get_light_random_info >>>")
    db=[]
    # random stim data
    # all the following recordings have direction: temporal (left side of video) to nasal (right side of video)
    db.append({'mname':'LZ16','datexp':'2024_07_29','blk':'1', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.9'})
    db.append({'mname':'TX139','datexp':'2024_08_10','blk':'1', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.5'})
    db.append({'mname':'TX143','datexp':'2024_08_10','blk':'1', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.7'})
    db.append({'mname':'TX144','datexp':'2024_08_10','blk':'1', 'blink_baselined_threshold':'.4', 'speed_threshold':'99.5'})
    db.append({'mname':'FXa8','datexp':'2026_01_13','blk':'1', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.5'})
    db.append({'mname':'FXa9','datexp':'2026_01_15','blk':'1', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.5'})
    db.append({'mname':'FXa10','datexp':'2026_01_13','blk':'1', 'blink_baselined_threshold':'.2', 'speed_threshold':'99.5'})    
    db = np.array(db)
    return db

def segment_data(data, frames_per_folder, idx):
    idx = np.arange(sum(frames_per_folder[:idx]), sum(frames_per_folder[:idx+1])) 
    return data[idx]


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
        
        if 'dy' in ops:
            ypos0 += ops['dy'] 
            xpos0 += ops['dx'] 

        if 1: #redo the deconvoloution after shifting the traces by subsamples
            F = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'F.npy'))
            Fneu  = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'Fneu.npy'))            
            F     = loaders.baselining(ops, tlags[n], F, Fneu)            
            spks0 = loaders.oasis(F, ops['batch_size'], ops['tau'], ops['fs'])
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

def get2Pft(data, tcam=True):
    # data[:,0] analog signal     data[:,1] time for each batch
    if not tcam and np.any(data) and np.sum(data.shape)>1000:
        ind = np.where(data[:,1]>0)[0]
        chunkTime = data[ind,1]
        frame_start_temp = (data[:-1,0] > 2.5) & (data[1:,0] <= 2.5)
        frame_start = np.where(frame_start_temp==1)[0]
        tneural = np.interp(frame_start, ind, chunkTime)
        ft = tneural, []
    elif tcam:
        x = data[:,1]
        ix = (x<3).nonzero()[0][0]
        tneural = (np.logical_and(x[ix:-1]<3.2, x[ix+1:]>3.2)).nonzero()[0] + ix - 1
        #tneural = tneural[::ops['nplanes']//2]

        tneural = tneural[:-1]
        tneural = tneural + np.diff(tneural).mean()/2
        
        x = data[:, 0]
        threshold = (x.max() + x.min()) / 2
        low = x < threshold
        edges = np.where(np.diff(low.astype(int)) == 1)[0]

        tcam = np.zeros(len(x), dtype=int)

        for i in range(len(edges)):
            start = edges[i]
            end = edges[i+1] if i+1 < len(edges) else len(x)
            tcam[start:end] = i


        """
        x = data[:,0]
        ix = (x<0.5).nonzero()[0][0]
        tcam = (np.logical_and(x[ix:-1]<.5, x[ix+1:]>.5)).nonzero()[0] + ix - 1
        tcam = tcam[2:]"""
        
        ft = tneural, tcam
    else:
        ft = []
    return ft

def datetime2datenum(dt):
    """ dt: list or array, y:m:d:h:m:s:ms """
    import datetime
    dtime = datetime.datetime(dt[0], dt[1], dt[2], dt[3], dt[4], dt[5], dt[6])
    mdn = dtime + datetime.timedelta(days = 366)
    frac_seconds = (dtime-datetime.datetime(dtime.year,dtime.month,dtime.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
    frac_microseconds = dtime.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
    return mdn.toordinal() + frac_seconds + frac_microseconds

def Get_facedn(dt):
    dn = np.empty((dt.shape[0],1))
    for i in range(dt.shape[0]):
#         dn[i] = datetime2matlabdn(dt[i,:].astype(int))
        dn[i] = datetime2datenum(dt[i,:].astype(int))
    return dn

def Get_PupMsg(Msg):
    """Msg: frameInd,ypos,xpos,size,y,m,d,h,m,s,ms
       return: PupMsg--- frameInd,ypos,xpos,size,datenum 
    """
    PupMsg = np.empty((Msg.shape[0],5))
    PupMsg[:,:4] = Msg[:,:4].copy()
    dn = Get_facedn(Msg[:,4:])
    PupMsg[:,4] = dn[:,0]
    return PupMsg


def sample_run(pck, tneural, rot = 'roll'):
    run = np.array([pck[k][rot] for k in range(len(pck)-1)])
    run[0] = 0
    
    trun = np.array([pck[k]['nSamples'] for k in range(len(pck)-1)])
    trun = trun - np.diff(trun).mean()/2
    
    if len(tneural)!=len(trun):
        crun = np.cumsum(run)
        crun_neuro = np.interp(tneural, trun, crun)
        run_neuro = np.diff(crun_neuro).astype('float32')
        run_neuro = np.concatenate((run_neuro, run_neuro[:1]), 0)
    else:
        run_neuro = run
    return run_neuro, run, trun

def get_tneural(dat, nplanes=1):
    daq_period = dat['daq']['data'].shape[0]

    tdaq = dat['daq']['time']
    tdaq = np.array(tdaq).squeeze()
    idaq = daq_period * (1+np.arange(len(tdaq)))
    ddaq = np.array(dat['daq']['data']).T.flatten()
    ix = (ddaq[1:] < 2) * (ddaq[:-1]>2)
    imic = ix.nonzero()[0]
    imic = imic[:int(np.array(dat['daq']['nframes'])[0,0])*nplanes:nplanes]
    #imic = imic[:nframes*nplanes:nplanes]
    tmic = np.interp(imic, idaq, tdaq)
    
    return tmic

def get_neural_pc(neural_activity, n_comps=128):
    neural_activity -= neural_activity.mean(axis=1)[:,np.newaxis]
    std = ((neural_activity**2).mean(axis=1)**0.5)[:,np.newaxis]
    std = np.where(std==0, 1, std)  # don't scale when std==0
    neural_activity /= std
    #neural_activity /= ((neural_activity**2).mean(axis=1)**0.5)[:,np.newaxis]
    model = PCA(n_components=n_comps, copy=False).fit(neural_activity)
    # obtain neural PCs (project PCs onto raw data): time x components
    Y = model.components_.T * model.singular_values_
    return Y

def ridge_regression(X, Y, lam=0):
    """ predict Y from X using regularized reduced rank regression 
        
        *** subtract mean from X and Y before predicting
    
        Prediction:
        >>> Y_pred = X @ A
        Parameters
        ----------
        X : 2D array, input data (n_samples, n_features)
        
        Y : 2D array, data to predict (n_samples, n_predictors)
        
        Returns
        --------
        A : 2D array - prediction matrix 1 (n_predictors, rank)
    """
    CXX = (X.T @ X + lam * np.eye(X.shape[1])) / X.shape[0]
    CXY = (X.T @ Y) / X.shape[0]
    A = np.linalg.solve(CXX, CXY)
    return A

def get_rrr_coefficients(X, Y, rank=None, lam=0):
    """ predict Y from X using regularized reduced rank regression 
        *** subtract mean from X and Y before predicting
        if rank is None, returns A and B of full-rank (minus one) prediction
        Prediction:
        >>> Y_pred = X @ B @ A.T
        Parameters
        ---------
        X : 2D array, input data (n_samples, n_features)
        Y : 2D array, data to predict (n_samples, n_predictors)
        
        Returns
        ------
        A : 2D array - prediction matrix 1 (n_predictors, rank)
        B : 2D array - prediction matrix 2 (n_features, rank)
    """
    min_dim = min(Y.shape[1], min(X.shape[0], X.shape[1])) - 1
    if rank is None:
        rank = min_dim
    else:
        rank = min(min_dim, rank)
    # make covariance matrices
    CXX = (X.T @ X + lam * np.eye(X.shape[1])) / X.shape[0]
    CYX = (Y.T @ X) / X.shape[0]

    # compute inverse square root of matrix
    s, u = eigh(CXX)
    #u = model.components_.T
    #s = model.singular_values_**2
    CXXMH = (u * (s + lam)**-0.5) @ u.T

    # project into prediction space
    M = CYX @ CXXMH
    
    # do svd of prediction projection
    model = PCA(n_components=rank).fit(M)
    c = model.components_.T
    s = model.singular_values_
    A = M @ c
    B = CXXMH @ c
    
    return A, B
    
def reduced_rank_regression(X, Y, tbin=3, rank=None, lam=0, A=None, B=None):
    Y = bin1d(Y, tbin)
    X -= X.mean(axis=0)
    X = bin1d(X, tbin)

    n_t, n_feats = Y.shape
    itest, itrain =  split_testtrain(n_t)
    X_train, Y_train = X[itrain], Y[itrain]
    X_test, Y_test = X[itest], Y[itest]
    del X, Y
    # Get reduced rank regression coefficients
    if A is not None and B is not None:
        pass
    elif A is None and B is not None:
        A = ridge_regression(X_train @ B, Y_train, lam=lam).T
    else: 
        A, B = get_rrr_coefficients(X_train, Y_train, rank, lam) 
        
    del X_train, Y_train
   # rank = A.shape[1]
    corr = np.zeros((rank, n_feats))
    varexp = np.zeros((rank, n_feats))
    avg_varexp = np.zeros(rank)
    for i,rank in enumerate(range(rank)):
        Y_pred = X_test @ B[:,:rank+1] @ A[:,:rank+1].T
        varexp[i,:] = 1 - ((Y_pred - Y_test)**2).sum(axis=0)/(Y_test**2).sum(axis=0)
        avg_varexp[i] = 1 - (((Y_test - Y_pred)**2).mean(axis=0)).mean() / ((Y_test**2).mean(axis=0)).mean()
        for component in range(corr.shape[1]):
            corr[i, component], _ = pearsonr(Y_test[:,component], Y_pred[:,component]) # Corr for first pc

    return Y_pred, corr, varexp, Y_test, A, B, avg_varexp

def split_testtrain(n_t, frac=0.25):
    ''' this returns indices of testing data and training data '''
    n_segs = int(min(20, n_t/4)) #usu want 20 segs, but might not have enough frames for that
    n_len = int(n_t/n_segs)
    ninds = np.linspace(0, n_t - n_len, n_segs).astype(int)
    itest = (ninds[:,np.newaxis] + np.arange(0,n_len * frac,1,int)).flatten()
    itrain = np.ones(n_t, bool)
    itrain[itest] = 0
    
    return itest, itrain

def bin1d(X, tbin, axis=1):
    """ bin over first axis of data with bin tbin """
    size = list(X.shape)
    X = X[:size[0]//tbin*tbin].reshape((size[0]//tbin, tbin, -1)).mean(axis=axis)
    size[0] = X.shape[0]
    return X.reshape(size)

def remove_spines(ax):
    """Helper function to remove the spines from a subplot"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
# Code from Miguel for loading new suite2p files
import numbers
from pathlib import Path
def load_ops_safe(root):
    ops_path = os.path.join(root, 'ops.npy')
    if os.path.isfile(ops_path):
        ops = np.load(ops_path, allow_pickle=True).item()
        #check if the ops file is complete
        if isinstance(ops['dx'], numbers.Number) == False: ## checks if dx is a number , wrong ops keeps the dx and dy as empty lists []
            print('ops.npy file incomplete, replacing with corrected ops.npy file')
            os.remove(ops_path)
            create_ops(root)
            ops = np.load(ops_path, allow_pickle=True).item()
    else:
        print('ops.npy file not found, creating ops.npy file')
        create_ops(root)
        ops = np.load(ops_path, allow_pickle=True).item()
    return ops

def load_neurons_new(db, irev = np.zeros(30), verbose = True, root_dm11 = '/home/marius/dm11/data/PROC', iscell=False):    
    # this function loads and concatenates planes
    # it automatically applies a 160ms timelag to the deconvolved traces
    # it does not deconvolve the data again
    
    mname, datexp, blk = db['mname'], db['datexp'], db['blk']
    root = os.path.join(root_dm11, mname, datexp, blk)
    print("Loading data from: ", root)
    
    #ops = np.load(os.path.join(root, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item() 

    ops = load_ops_safe(os.path.join(root, 'suite2p', 'plane0'))
    
    spks = np.zeros((0, ops['nframes']), np.float32) 
    stat = np.zeros((0,)) 
    xpos, ypos, iplane, is_cell = np.zeros((0,)) , np.zeros((0,)) , np.zeros((0,)) , np.zeros((0,2))

    #tlags = .25 + np.linspace(.2, -.8, ops['nplanes']//2+1)[:-1]
    #tlags = np.hstack((tlags, tlags)) + .5
    nrecording_rois = len(list(Path(root).joinpath('suite2p').glob('plane*/'))) 
    print('number of recording planes: %d'%nrecording_rois)
    tlags = 10/60 * ops['fs'] * np.ones(nrecording_rois,)

    for n in range(nrecording_rois):

        #ops = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'ops.npy'), allow_pickle=True).item()
        ops = load_ops_safe(os.path.join(root, 'suite2p', 'plane%d'%n))
        stat0 = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'stat.npy'), allow_pickle=True)
        ypos0 = np.array([stat0[n]['med'][0] for n in range(len(stat0))]) 
        xpos0 = np.array([stat0[n]['med'][1] for n in range(len(stat0))]) 
        if verbose:
            print('plane %d, '%n, 'neurons: %d'%len(xpos0))
        if irev[n]>0:
            ypos0 = -ypos0

        #print('dy, dx offsets: ', ops['dy'], ops['dx'])
        ypos0 += ops['dy'] # add the per plane offsets (dy,dx)
        xpos0 += ops['dx'] # add the per plane offsets (dy,dx)
        
        ypos = np.concatenate((ypos, ypos0), axis=0)
        xpos = np.concatenate((xpos, xpos0), axis=0)

        spks0 = np.load(os.path.join(root, 'suite2p', 'plane%d'%n, 'spks.npy'), allow_pickle=True)
        if iscell:
            is_cell0 = np.load(
                os.path.join(root, "suite2p", "plane%d" % n, "iscell.npy"),
                allow_pickle=True,
            )
            is_cell = np.concatenate((is_cell, is_cell0), axis=0)
        if tlags[n]<0:
            spks0[:, 1:] = (1 + tlags[n]) * spks0[:, 1:] + (- tlags[n]) * spks0[:, :-1]
        else:
            spks0[:, :-1] = (1 - tlags[n]) * spks0[:, :-1] + tlags[n] * spks0[:, 1:]

        spks0 = spks0.astype('float32')
        iplane = np.concatenate((iplane, n*np.ones(len(stat0),)))
        stat = np.concatenate((stat,stat0), axis=0)     
        if spks.shape[1]>spks0.shape[0]:
            spks0 = np.concatenate((spks0, np.zeros((spks0.shape[0], spks.shape[1]-spks0.shape[1]), 'float32')), axis=1)
        spks = np.concatenate((spks,spks0), axis=0) 
                           

    print('total neurons %d'%len(spks))

    return spks, iplane, xpos, ypos, ops, is_cell