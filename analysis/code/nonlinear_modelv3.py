# Nonlinear model code refactored

import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import sys
sys.path.insert(0, '/home/asyeda/Facemap/facemap')
from facemap.neural_prediction import neural_model
import imp
imp.reload(neural_model)
from facemap.neural_prediction.prediction_utils import rrr_prediction, compute_varexp
import time, helper_functions
import random, copy
import pupil_dark_helpers as pdh
import matplotlib.pyplot as plt
from torch import nn

def loss_fn(y_pred, y_true, sm_penalty=0.1, cweight=None):
    # mse loss
    mse_loss = nn.MSELoss()
    loss = mse_loss(y_pred, y_true)

    # temporal conv smoothness penalty
    if sm_penalty > 0 and cweight is not None:
        smloss = sm_penalty * (torch.diff(cweight)**2).sum()
        loss += smloss
    return loss

def train_model(net, X_train, X_val, X_test, Y_train, Y_val, Y_test, 
                itrain_sample_b, ival_sample_b, itest_sample_b, hyperparameters, 
                smoothing_penalty=0.5, n_epochs=300, verbose=False, min_epochs=50):
    # Train behavior -> neural model using multiple animals
    learning_rate = hyperparameters['learning_rate']
    weight_decay = hyperparameters['weight_decay']
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    learning_rates = []

    n_blocks = len(X_train)
    ### determine total number of batches across all animals/blocks to sample from
    n_batches = [0]
    n_batches.extend([X_train[i].shape[0] for i in range(n_blocks)]) 
    n_batches = np.array(n_batches)
    c_batches = np.cumsum(n_batches)
    n_batches = n_batches.sum()   
    # validation batches
    n_batches_val = [0]
    n_batches_val.extend([X_val[i].shape[0] for i in range(n_blocks)])
    n_batches_val = np.array(n_batches_val)
    c_batches_val = np.cumsum(n_batches_val)
    n_batches_val = n_batches_val.sum()
    # test batches
    n_batches_test = [0]
    n_batches_test.extend([X_test[i].shape[0] for i in range(n_blocks)])
    n_batches_test = np.array(n_batches_test)
    c_batches_test = np.cumsum(n_batches_test)
    n_batches_test = n_batches_test.sum()

    # Initialize variables for tracking the best model
    best_ve = -float('inf')  # Initialize best VE
    best_y_pred = None  # Initialize best prediction
    best_state_dict = copy.deepcopy(net.state_dict())  # save initial state dict

    patience = hyperparameters.get('early_stopping_patience', 15)  # number of epochs to wait
    epochs_no_improve = 0

    varexp_return = None  # Initialize variance explained return value
    val_acc = []  # Track validation accuracy
    train_acc = []  # Track train accuracy
    test_acc = []  # Track test accuracy

    train_loss, val_loss, test_loss = np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs)
    #print("n_batches: ", n_batches, "c_batches: ", c_batches)
    # Loop over epochs
    for iepoch in range(n_epochs):
        net.train() # set model to training mode
        batch_perm = np.random.permutation(n_batches)  # random permutation of batches
        for istart in batch_perm:
            blk_idx = np.nonzero(istart >= c_batches)[0][-1]
            sample_start = istart - c_batches[blk_idx]
            X_batch = X_train[blk_idx][[sample_start]]
            Y_batch = torch.cat([Y_train[blk_idx][sample_start]], axis=0)

            # compute network output from inputs in train_data
            Y_pred = net(X_batch)[0]
            # sample at neural activity times
            sample_idx = [itrain_sample_b[blk_idx][sample_start]]
            Y_pred = Y_pred[sample_idx[0],:]

            # evaluate loss function
            loss = loss_fn(Y_pred, Y_batch)
            loss += smoothing_penalty * (torch.diff(net.core.features[1].weight)**2).sum()

            optimizer.zero_grad()   # clear previous gradients
            loss.backward()         # compute gradients
            optimizer.step()        # update weights
            train_loss[iepoch] += loss.item()

        train_loss[iepoch] /= n_batches

        # evaluate test loss and validation loss
        net.eval()
        val_loss[iepoch] = 0.0
        test_loss[iepoch] = 0.0
        Y_pred_val = []
        Y_val_cpu = []
        Y_pred_test = []
        Y_test_cpu = []
        with torch.no_grad():
            # validation loss
            for i in range(n_batches_val):
                blk_idx = np.nonzero(i >= c_batches_val)[0][-1]
                sample_start = i - c_batches_val[blk_idx]
                X_batch = X_val[blk_idx][[sample_start]]
                Y_batch = torch.cat([Y_val[blk_idx][sample_start]], axis=0)
                Y_pred = net(X_batch)[0]
                # sample at neural activity times
                sample_idx = ival_sample_b[blk_idx][sample_start]
                Y_pred = Y_pred[sample_idx,:]
                Y_pred_val.append(Y_pred.cpu().numpy())
                Y_val_cpu.append(Y_val[blk_idx][sample_start].cpu().numpy())
                # evaluate loss function
                loss = loss_fn(Y_pred, Y_batch)
                if torch.isnan(loss):
                    continue
                else: 
                    val_loss[iepoch] += loss
            val_loss[iepoch] /= n_batches_val
            Y_pred_val = np.vstack(Y_pred_val)
            Y_val_cpu = np.vstack(Y_val_cpu)
            varexp_val = 1 - ((Y_pred_val - Y_val_cpu)**2).mean() / ((Y_val_cpu - Y_val_cpu.mean())**2).mean()

            Y_pred_return = []
            for blk_idx in range(n_blocks):
                for i in range(c_batches_test[blk_idx], c_batches_test[blk_idx+1]):
                    sample_start = i - c_batches_test[blk_idx]
                    X_batch = X_test[blk_idx][[sample_start]]
                    Y_batch = torch.cat([Y_test[blk_idx][sample_start]], axis=0)
                    Y_pred = net(X_batch)[0]
                    # sample at neural activity times
                    sample_idx = itest_sample_b[blk_idx][sample_start]
                    Y_pred = Y_pred[sample_idx,:]
                    Y_pred_test.append(Y_pred.cpu().numpy())
                    Y_test_cpu.append(Y_test[blk_idx][sample_start].cpu().numpy())
                    Y_pred_return.append(Y_pred.cpu().numpy())
                    # evaluate loss function
                    loss = loss_fn(Y_pred, Y_batch)
                    if torch.isnan(loss):
                        continue
                    else:
                        test_loss[iepoch] += loss

            test_loss[iepoch] /= n_batches_test
            Y_test_cpu = np.vstack(Y_test_cpu)
            varexp_test = 1 - ((np.vstack(Y_pred_test) - Y_test_cpu)**2).mean() / ((Y_test_cpu - Y_test_cpu.mean())**2).mean()

            val_acc.append(varexp_val)               # Validation accuracy
            test_acc.append(varexp_test)              # Test accuracy

            scheduler.step(val_acc[-1])                # Step the scheduler with validation accuracy
            learning_rates.append(optimizer.param_groups[0]['lr'])

        if verbose:
            print(f'epoch {iepoch}, train_loss = {train_loss[iepoch]:.3f}, val_loss = {val_loss[iepoch]:.3f}, varexp_test = {varexp_test:.3f}, varexp_val = {varexp_val:.3f}')

        # Check if current val VE is the best and save model checkpoint
        if val_acc[-1] > best_ve:
            best_ve = val_acc[-1]
            varexp_return = test_acc[-1] # Save best model's test variance explained
            best_y_pred = Y_pred_return # used to compute final varexp on test data
            best_state_dict = copy.deepcopy(net.state_dict())
            epochs_no_improve = 0  # reset counter
            if verbose:
                print(f'New best VE: {best_ve:.6f}')
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience and not iepoch < min_epochs:
            print(f"Early stopping triggered at epoch {iepoch+1}.")
            break

    return best_y_pred, varexp_return, best_state_dict, best_ve

def split_traintest_fixed_segment(n_t, segment_length, hyperparameters):
    """This returns deterministic split of train, validation, and test in time chunks
    
    Parameters
    ----------
    n_t : int
        Total number of timepoints in behavior data
    segment_length : int
        Fixed length of segments in the train set
    hyperparameters : dict
        Dictionary containing 'frac', 'k_fold_idx', 'k_folds', and 'seed' keys
        frac : float
            Fraction of points to use for test and validation sets
        k_fold_idx : int
            Index of the current fold for k-fold cross-validation
        k_folds : int
            Total number of folds for k-fold cross-validation
        seed : int
            Random seed for reproducibility
    Returns
    --------
    itrain : 2D int array
        indices in train set, arranged in chunks, with padding in behavior timepoints
    ival : 2D int array
        indices in validation set, arranged in chunks, with padding in behavior timepoints
    itest : 2D int array
        indices in test set, arranged in chunks, without padding in behavior timepoints
    """
    set_seed(hyperparameters['seed'])
    k_fold_idx = hyperparameters['k_fold_idx']
    frac = hyperparameters['frac']
    
    # Calculate the number of segments based on the desired segment length
    print("Using segment length of: ", segment_length)
    n_segs = int(np.floor(n_t / segment_length))
    n_len = segment_length
    if n_segs < 3:
        raise ValueError("Not enough data points to create train, validation, and test sets with the desired segment length.")
    
    inds_all = np.arange(0, n_segs * n_len, n_len)
    # Total size of the dataset
    total_size = len(inds_all)

    # Calculate the number of segments for each split
    train_size = int(np.floor((1-(2*frac)) * n_segs))
    test_size = int(np.floor((frac) * n_segs))
    val_size = int(np.floor((frac) * n_segs))

    # use rperm to select random permutation of indices
    rperm = np.random.permutation(inds_all)
    test_start = k_fold_idx*test_size
    test_end = test_start + test_size if k_fold_idx < hyperparameters['k_folds'] - 1 else total_size
    inds_test = rperm[test_start:test_end]
    remaining_indices = np.setdiff1d(rperm, inds_test, assume_unique=True)
    inds_val = remaining_indices[-test_size:]
    inds_train = np.setdiff1d(remaining_indices, inds_val, assume_unique=True)

    # sort indices
    inds_train = np.sort(inds_train)
    inds_val = np.sort(inds_val)
    inds_test = np.sort(inds_test)

    # check if any overlap between train, val, and test
    if np.intersect1d(inds_train, inds_val).size > 0 or np.intersect1d(inds_train, inds_test).size > 0 or np.intersect1d(inds_val, inds_test).size > 0:
        print("1a Overlap between train, val, and test")
    # add padding 
    pad_before = 100 # 2 s padding
    pad_after = 100  # 2 s padding
    itrain = (inds_train[:, np.newaxis] + np.arange(pad_before, segment_length-pad_after))
    ival = (inds_val[:, np.newaxis] + np.arange(pad_before, segment_length-pad_after))
    # no padding for test set
    itest = (inds_test[:, np.newaxis] + np.arange(0, segment_length)) 
    return itrain, ival, itest

def have_overlapping_indices(listA, listB):
    # Convert each list of arrays into one flattened set of ints
    idxA = set(np.concatenate(listA)) if listA else set()
    idxB = set(np.concatenate(listB)) if listB else set()

    return len(idxA & idxB) > 0

def split_batches_deterministic(tcam, tneural, segment_length, hyperparameters, device=torch.device('cuda'),
                                itrain=None, ival=None, itest=None):
    """This returns deterministic split of train, validation, and test in time chunks for neural and cam times
    
    Parameters
    ----------
    tcam : 1D array
        Times of camera frames
    tneural : 1D array
        Times of neural frames
    segment_length : int
        Fixed length of segments in behavior timepoints
    hyperparameters : dict
        frac : float (optional, default 0.25)
            Fraction of points to put in test set
        val_frac : float (optional, default 0.1)
            Fraction of points from train set to put in validation set
        pad : int (optional, default 3)
            Number of timepoints to exclude from test set before and after training segment
    itrain: 2D int array
        Times in train set, arranged in chunks
    ival: 2D int array
        Times in validation set, arranged in chunks
    itest: 2D int array
        Times in test set, arranged in chunks
    
    Returns
    --------
    itrain: 1D int array
        Times in train set, arranged in chunks
    ival: 1D int array
        Times in validation set, arranged in chunks
    itest: 1D int array
        Times in test set, arranged in chunks
    itrain_cam: 2D int array
        Times in cam frames in train set, arranged in chunks
    ival_cam: 2D int array
        Times in cam frames in validation set, arranged in chunks
    itest_cam: 2D int array
        Times in cam frames in test set, arranged in chunks
    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Camera time splits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1) split in CAM TIME
    if itrain is None or ival is None or itest is None:
        itrain, ival, itest = split_traintest_fixed_segment(len(tcam), segment_length, hyperparameters)
    
    # check any overlap between train, val, and test
    if np.intersect1d(itrain, ival).size > 0 or np.intersect1d(itrain, itest).size > 0 or np.intersect1d(ival, itest).size > 0:
        print("!!!!!  1b Overlap between cam train, val, or test !!!!!")

    # INTERPOLATE to find corresponding neural indices for each cam index
    f = interp1d(tcam, np.arange(len(tcam)))
    icam_best = np.round(f(tneural)).astype('int')

    # find indices of elements of itrain / itest in icam_best
    in_train = np.isin(itrain, icam_best) # Boolean mask of where itrain matches icam_best
    #isample_train = [np.flatnonzero(row) for row in in_train] # Split the columns per segment (row) using boolean masking
    isample_train = np.nonzero(in_train)
    isample_train = [isample_train[1][isample_train[0] == i] for i in range(itrain.shape[0])]

    # obtain neural activity indices in each val batch
    in_val = np.isin(ival, icam_best)
    #isample_val  = [np.flatnonzero(row) for row in in_val]
    isample_val = np.nonzero(in_val)
    isample_val = [isample_val[1][isample_val[0] == i] for i in range(ival.shape[0])]

    # obtain neural activity indices in each test batch
    in_test = np.isin(itest, icam_best)
    #isample_test = [np.flatnonzero(row) for row in in_test]
    isample_test = np.nonzero(in_test)
    isample_test = [isample_test[1][isample_test[0] == i] for i in range(itest.shape[0])]

    # put on the GPU
    isample_train = [torch.from_numpy(isample_train[i]).to(device) for i in range(in_train.shape[0])]
    isample_val = [torch.from_numpy(isample_val[i]).to(device) for i in range(in_val.shape[0])]
    isample_test = [torch.from_numpy(isample_test[i]).to(device) for i in range(in_test.shape[0])]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Neural time splits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # obtain neural activity indices in each batch
    ineural_train = [np.nonzero(np.isin(icam_best, itrain[i]))[0] for i in range(itrain.shape[0])]
    ineural_val = [np.nonzero(np.isin(icam_best, ival[i]))[0] for i in range(ival.shape[0])]
    ineural_test = [np.nonzero(np.isin(icam_best, itest[i]))[0] for i in range(itest.shape[0])]
    # check any overlap between train, val, and test
    if have_overlapping_indices(ineural_train, ineural_test) or have_overlapping_indices(ineural_train, ineural_val) or have_overlapping_indices(ineural_val, ineural_test):
        print("!!!!!  2 Overlap between neural train, val, or test !!!!!")

    return itrain, ival, itest, ineural_train, ineural_val, ineural_test, isample_train, isample_val, isample_test

def split_data_deterministic(X, Y, tcam, tneural, segment_length, hyperparameters, device=torch.device('cuda')):

    splits = split_batches_deterministic(tcam, tneural, segment_length, hyperparameters, device=device)
    itrain, ival, itest, ineural_train, ineural_val, \
        ineural_test, itrain_ds, ival_ds, itest_ds = splits

    # X uses CAM indices
    X_train = torch.from_numpy(X[itrain]).float().to(device)
    X_val   = torch.from_numpy(X[ival]).float().to(device)
    X_test  = torch.from_numpy(X[itest]).float().to(device)

    # Y uses NEURAL indices
    Y_train = [torch.from_numpy(Y[ineural_train[i]]).to(device) for i in range(len(ineural_train))]
    Y_val   = [torch.from_numpy(Y[ineural_val[i]]).to(device) for i in range(len(ineural_val))]
    Y_test  = [torch.from_numpy(Y[ineural_test[i]]).to(device) for i in range(len(ineural_test))]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, itrain_ds, ival_ds, itest_ds, itrain, ival, itest, ineural_train, ineural_val, ineural_test

def zscore_neural_data(data, axis=1):
    data -= data.mean(axis=axis, keepdims=True)
    std = ((data ** 2).mean(axis=axis, keepdims=True) ** 0.5)
    std[std == 0] = 1
    data /= std
    return data

def zscore_behavior_data(data, axis=0):
    data -= np.nanmean(data, axis=axis, keepdims=True)
    data /= np.nanstd(data, axis=axis, keepdims=True)

    return data

def interpolate_data(x, tcam, tcam_rescale):
    f = interp1d(tcam, x, fill_value="extrapolate", axis=0)
    return f(tcam_rescale)

def load_data(base_folder, mname, datexp):
    npz_data = np.load(f"{base_folder}/{mname}_{datexp}_{1}.npz", allow_pickle=True)
    spks = npz_data['spks']
    xpos = npz_data['xpos']
    ypos = npz_data['ypos']
    tcams = npz_data['tcam'].item()
    tneurals = npz_data['tneural'].item()
    cumframes_neural = np.hstack(([0], np.cumsum(npz_data['neural_frames_list'])))
    cumframes_cam = np.hstack(([0], np.cumsum(npz_data['cam_frames_list'])))
    
    return spks, xpos, ypos, tcams, tneurals, cumframes_neural, cumframes_cam, npz_data

def process_blocks(mouse_dbs, Vsv, tcams, tneurals, cumframes_neural, cumframes_cam, input_data, hyperparameters, device):
    blk_splits = []
    for blk in range(len(mouse_dbs)): 
        blk_name = mouse_dbs[blk]["blk"]
        tneural = tneurals[list(tneurals.keys())[blk]]
        Vsv_blk = Vsv[cumframes_neural[blk]:cumframes_neural[blk+1]].copy()
        tcam = tcams[f"blk{blk_name}"].copy()
        tcam_rescale = np.linspace(tcam[0], tcam[-1], len(tcam))
        
        inds = (tneural < tcam_rescale[-1]) & (tneural > tcam_rescale[0])
        Vsv_blk = Vsv_blk[inds]
        tneural = tneural[inds]
        
        x_blk = input_data[cumframes_cam[blk]:cumframes_cam[blk+1], :]
        x_blk = interpolate_data(x_blk, tcam, tcam_rescale)
        
        X_dat, Y_dat, tcam_list, tneural_list = [x_blk], [Vsv_blk], [tcam_rescale], [tneural]
        arrs = [[] for _ in range(15)]
        
        for X, Y, tcam, tneural in zip(X_dat, Y_dat, tcam_list, tneural_list):
            dsplits = split_data_deterministic(X, Y, tcam, tneural, hyperparameters['segment_length'], hyperparameters, device=device)
            for d, a in zip(dsplits, arrs):
                a.append(d)
        blk_splits.append(arrs)
    return blk_splits

def train_and_evaluate(X_train, X_val, X_test, Y_train, Y_val, Y_test, itrain_sample_b, ival_sample_b, itest_sample_b, n_in, n_out, hyperparameters, device):
    net = neural_model.KeypointsNetwork(n_in=n_in, n_out=n_out, n_kp=hyperparameters['n_kp'], n_latents=hyperparameters['n_latents']).to(device)
    """
    # check any nan in X_train, Y_train
    for i, (xtr, ytr, xval, yval, xtst, ytst) in enumerate(zip(X_train, Y_train, X_val, Y_val, X_test, Y_test)):
        assert not torch.isnan(xtr).any(), f"X_train[{i}] contains NaN values"
        assert not torch.isnan(ytr).any(), f"Y_train[{i}] contains NaN values"
        assert not torch.isnan(xval).any(), f"X_val[{i}] contains NaN values"
        assert not torch.isnan(yval).any(), f"Y_val[{i}] contains NaN values"
        assert not torch.isnan(xtst).any(), f"X_test[{i}] contains NaN values"
        assert not torch.isnan(ytst).any(), f"Y_test[{i}] contains NaN values"
    """
    y_pred, vexp, best_state_dict, best_val_ve = train_model(
        net, X_train, X_val, X_test, Y_train, Y_val, Y_test, 
        itrain_sample_b, ival_sample_b, itest_sample_b, hyperparameters, 
        n_epochs=300
    )
    return y_pred, vexp, best_state_dict, best_val_ve

def peer_prediction(spks, xpos, ypos, dum=400):
    ineu1 = np.logical_xor((xpos % dum) < dum / 2, (ypos % dum) < dum / 2)
    # ineu1 = np.random.rand(len(spks)) > 0.5
    ineu2 = np.logical_not(ineu1)
    n_components = 128
    Vn = []
    for ineu in [ineu1, ineu2]:
        Vn.append(
            PCA(n_components=n_components, copy=False).fit_transform(spks[ineu].T)
        )
    varexp = 0
    varexp_neurons = np.zeros((spks.shape[0]))
    for k, ineu in enumerate([ineu1, ineu2]):
        V_pred_test, varexpk, itest = rrr_prediction(
            Vn[(k + 1) % 2], Vn[k % 2], rank=128, lam=1e-1
        )[:3]
        varexp += varexpk[-1]
        U = spks[ineu] @ Vn[k]
        U /= (U**2).sum(axis=0) ** 0.5
        spks_pred_test = V_pred_test @ U.T
        spks_test = spks[ineu][:, itest].T
        varexp_neurons[ineu] = compute_varexp(spks_test, spks_pred_test)

    # average variance explained for two halves
    varexp /= 2
    return varexp, varexp_neurons, itest

def interpolate_pupil_blink(pupil: np.ndarray, blink: np.ndarray, percentile: float) -> np.ndarray:
    """
    Interpolates pupil data where blink values are below a specified percentile threshold.

    Parameters:
    -----------
    pupil : np.ndarray
        Array of pupil values.
    blink : np.ndarray
        Array of blink values.
    percentile : float
        Percentile value to threshold blink data (e.g., 5 for 5th percentile).

    Returns:
    --------
    pupil_interp : np.ndarray
        Array of pupil values with interpolated values where blink < threshold.
    """
    # Step 1: Calculate the percentile threshold of blink values
    blink_threshold = np.percentile(blink, percentile)

    # Step 2: Create a mask for blink values below the threshold
    blink_mask = blink < blink_threshold

    # Step 3: Find valid and invalid indices
    valid_indices = np.where(~blink_mask)[0]  # Indices where blink is valid
    invalid_indices = np.where(blink_mask)[0]  # Indices where blink is invalid

    # Step 4: Perform linear interpolation on pupil data
    interp_func = interp1d(valid_indices, pupil[valid_indices], kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
    
    # Step 5: Replace pupil values where blink < threshold
    pupil_interp = pupil.copy()
    pupil_interp[blink_mask] = interp_func(invalid_indices)

    return pupil_interp

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_flattened_data_from_splits(data_splits):
    all_pc_data = []
    for fold in data_splits:        # loop over k folds
        for seg in fold[0]:       # loop over segments in each fold
            all_pc_data.append(seg)   # seg is shape (T,128)
    all_pc_data = np.concatenate(all_pc_data, axis=0)   # shape = (total_timepoints, 128)
    return all_pc_data

def get_flattened_predictions_from_splits(data_splits):
    all_pc_pred = []
    for fold in data_splits:        # loop over k folds
        for seg in fold:       # loop over segments in each fold
            all_pc_pred.append(seg)   # seg is shape (T,128)
    all_pc_pred = np.concatenate(all_pc_pred, axis=0)   # shape = (total_timepoints, 128)
    return all_pc_pred

def compute_variance_explained_kfolds(n_blocks, n_folds, condition, y_pred, y_pred_before_kfolds,\
                                       y_pred_after_kfolds, spks, U, data_splits_before_itest, \
                                        data_splits_after_itest, data_splits_itest, data_splits_before_ytest, \
                                        data_splits_after_ytest, data_splits_ytest, cumframes_neural, \
                                        varexp_expl_neurons, n_out, reset_indices):   
    
    # Get variance explained per principal component for entire session
    v_test = get_flattened_predictions_from_splits(data_splits_ytest)
    y_pred = get_flattened_predictions_from_splits(y_pred)
    vpcs = ((v_test - v_test.mean(axis=0))**2).mean(axis=0)
    vexp_pc = 1 - (y_pred - v_test).var() / (v_test).var()
    print("Variance explained (all PCs): ", vexp_pc * 100)

    if condition == "without_pupil":
        # ~~~~~~~~~~ Compute variance explained for each component ~~~~~~~~~~
        data_splits_before_ytest = get_flattened_predictions_from_splits(data_splits_before_ytest)
        y_pred_before_kfolds = get_flattened_predictions_from_splits(y_pred_before_kfolds)

        vexp_before_carbachol = 1 - ((y_pred_before_kfolds - data_splits_before_ytest)**2).mean() / ((data_splits_before_ytest - data_splits_before_ytest.mean(axis=0))**2).mean()
        vpcs_before_carbachol = ((data_splits_before_ytest - data_splits_before_ytest.mean(axis=0))**2).mean(axis=0)
        vexp_pc_before_carbachol = 1 - ((y_pred_before_kfolds - data_splits_before_ytest)**2).mean(axis=0) / vpcs_before_carbachol

        data_splits_after_ytest = get_flattened_predictions_from_splits(data_splits_after_ytest)
        y_pred_after_kfolds = get_flattened_predictions_from_splits(y_pred_after_kfolds)

        vexp_after_carbachol = 1 - ((y_pred_after_kfolds - data_splits_after_ytest)**2).mean() / ((data_splits_after_ytest - data_splits_after_ytest.mean(axis=0))**2).mean()
        vpcs_after_carbachol = ((data_splits_after_ytest - data_splits_after_ytest.mean(axis=0))**2).mean(axis=0)
        vexp_pc_after_carbachol = 1 - ((y_pred_after_kfolds - data_splits_after_ytest)**2).mean(axis=0) / vpcs_after_carbachol

        print("PC before VE (%): ", vexp_before_carbachol * 100)
        print("PC after VE (%): ", vexp_after_carbachol * 100)
        # ~~~~~~~~~~ Compute neurons variance explained ~~~~~~~~~~
        # Before
        spks_pred_test_before_carbachol = y_pred_before_kfolds @ U.T
        spks_test_before_carbachol = []
        for j in range(n_folds):
            for i in range(1): # only first block 
                flat_indices = [x for sub in data_splits_before_itest[j][i] for x in sub]
                spks_test_before_carbachol.append(spks[:, cumframes_neural[i]:cumframes_neural[i+1]][:, flat_indices].T)
        varexp_neurons_before_carbachol = compute_varexp(np.vstack(spks_test_before_carbachol), spks_pred_test_before_carbachol)
        print("NEURONS before (%)", np.nanmean(varexp_neurons_before_carbachol) * 100)

        # After
        spks_test_after_carbachol = []
        spks_pred_test_after_carbachol = y_pred_after_kfolds @ U.T
        for j in range(n_folds):
            for i in range(1, n_blocks): # all blocks except first
                if i != len(cumframes_neural)-2: # if not the last block
                    # flatten indices for all segments in the block
                    flat_indices = [x for sub in data_splits_after_itest[j][i-1] for x in sub]
                    spks_test_after_carbachol.append(spks[:, cumframes_neural[i]:cumframes_neural[i+1]][:, flat_indices].T)
                else: # if the last block
                    flat_indices = [x for sub in data_splits_after_itest[j][i-1] for x in sub]
                    spks_test_after_carbachol.append(spks[:, cumframes_neural[i]:][:, flat_indices].T)
        varexp_neurons_after_carbachol = compute_varexp(np.vstack(spks_test_after_carbachol), spks_pred_test_after_carbachol)
        print("NEURONS after (%)", np.nanmean(varexp_neurons_after_carbachol)* 100)

    elif condition == "with_pupil":

        vexp_before_carbachol = vexp_pc_before_carbachol = vpcs_before_carbachol = np.zeros((int(n_out)))
        varexp_neurons_before_carbachol = np.zeros((spks.shape[0]))

        data_splits_after_ytest = get_flattened_predictions_from_splits(data_splits_after_ytest)
        y_pred_after_kfolds = get_flattened_predictions_from_splits(y_pred_after_kfolds)

        vexp_after_carbachol = 1 - ((y_pred_after_kfolds - data_splits_after_ytest)**2).mean() / ((data_splits_after_ytest - data_splits_after_ytest.mean(axis=0))**2).mean()
        vpcs_after_carbachol = ((data_splits_after_ytest - data_splits_after_ytest.mean(axis=0))**2).mean(axis=0)
        vexp_pc_after_carbachol = 1 - ((y_pred_after_kfolds - data_splits_after_ytest)**2).mean(axis=0) / vpcs_after_carbachol

        print("PC after VE (%): ", vexp_after_carbachol * 100)

        # ~~~~~~~~~~ Compute neurons variance explained ~~~~~~~~~~
        # After
        spks_test_after_carbachol = []
        spks_pred_test_after_carbachol = y_pred_after_kfolds @ U.T
        for j in range(n_folds):
            for i in range(1, n_blocks): # all blocks except first 
                if i != len(cumframes_neural)-2: # if not the last block
                    # flatten indices for all segments in the block
                    flat_indices = [x for sub in data_splits_after_itest[j][i-1] for x in sub]
                    spks_test_after_carbachol.append(spks[:, cumframes_neural[i]:cumframes_neural[i+1]][:, flat_indices].T)
                else: # if the last block
                    flat_indices = [x for sub in data_splits_after_itest[j][i-1] for x in sub]
                    spks_test_after_carbachol.append(spks[:, cumframes_neural[i]:][:, flat_indices].T)
        varexp_neurons_after_carbachol = compute_varexp(np.vstack(spks_test_after_carbachol), spks_pred_test_after_carbachol)
        print("NEURONS after (%)", np.nanmean(varexp_neurons_after_carbachol)* 100)

    else: # Light recordings (no carbachol invovled; only analyze first session)
        vexp_after_carbachol = vexp_pc_after_carbachol = vpcs_after_carbachol = np.zeros((int(n_out)))
        varexp_neurons_after_carbachol = np.zeros((spks.shape[0]))

        data_splits_before_ytest = get_flattened_predictions_from_splits(data_splits_before_ytest)
        y_pred_before_kfolds = get_flattened_predictions_from_splits(y_pred_before_kfolds)

        vexp_before_carbachol = 1 - ((y_pred_before_kfolds - data_splits_before_ytest)**2).mean() / ((data_splits_before_ytest - data_splits_before_ytest.mean(axis=0))**2).mean()
        vpcs_before_carbachol = ((data_splits_before_ytest - data_splits_before_ytest.mean(axis=0))**2).mean(axis=0)
        vexp_pc_before_carbachol = 1 - ((y_pred_before_kfolds - data_splits_before_ytest)**2).mean(axis=0) / vpcs_before_carbachol
        print("PC before carbachol VE (%): ", vexp_before_carbachol * 100)
        # ~~~~~~~~~~ Compute neurons variance explained ~~~~~~~~~~
        # Before
        spks_pred_test_before_carbachol = y_pred_before_kfolds @ U.T
        spks_test_before_carbachol = []
        for j in range(n_folds):
            for i in range(1): # only one block in light recordings   
                flat_indices = [x for sub in data_splits_before_itest[j][i] for x in sub]
                spks_test_before_carbachol.append(spks[:, cumframes_neural[i]:cumframes_neural[i+1]][:, flat_indices].T)
        varexp_neurons_before_carbachol = compute_varexp(np.vstack(spks_test_before_carbachol), spks_pred_test_before_carbachol)
        print("NEURONS before (%)", np.nanmean(varexp_neurons_before_carbachol) * 100)

    return {
        "vexp_before_carbachol": vexp_before_carbachol,
        "vexp_pc_before_carbachol": vexp_pc_before_carbachol,
        "vpcs_before_carbachol": vpcs_before_carbachol,
        "vexp_after_carbachol": vexp_after_carbachol,
        "vexp_pc_after_carbachol": vexp_pc_after_carbachol,
        "vpcs_after_carbachol": vpcs_after_carbachol,
        "varexp_neurons_before_carbachol": varexp_neurons_before_carbachol,
        "varexp_neurons_after_carbachol": varexp_neurons_after_carbachol,
        "vexp_pc": vexp_pc,
        "vpcs": vpcs,
        "varexp_expl_neurons": varexp_expl_neurons
    }

# Main Processing Loop
def analyze_mouse_data(data_path, mouse_list, dbs, device, recording_type, hyperparameters):

    if "light" in recording_type and "static" in recording_type:
        recording_dbs = pdh.get_light_static_info()
    elif "light" in recording_type and "random" in recording_type:
        recording_dbs = pdh.get_light_random_info()

    nonlinear_results = {}
    rastermap_output = {}
    for mname in mouse_list:
        print(f"Processing {mname}")
        nonlinear_results[mname] = {}
        rastermap_output[mname] = {}
        
        # Load Data
        mouse_dbs = [db for db in dbs if db['mname'] == mname]
        datexp = mouse_dbs[0]['datexp']
        if mname in ['FXa10', 'FXa8', 'FXa9', 'FXa1']:
            spks, xpos, ypos, tcams, tneurals, cumframes_neural, cumframes_cam, npz_data = load_data(f'{data_path}/mouse_data/{mname}/{datexp}/{1}', mname, datexp) # different base folder for these mice
        else:
            _, xpos, ypos, tcams, tneurals, cumframes_neural, cumframes_cam, npz_data = load_data(f'{data_path}/mouse_data/{mname}/{datexp}/', mname, datexp)
            # Load spks using the new deconvolution method (udcnv)
            spks_udcnv_file = f"{data_path}/neural_data_udcnv/{mname}_{datexp}_1_spks.npz"
            spks = np.load(spks_udcnv_file)['spks']
            print(f"UDCNV neural data shape: {spks.shape}")
            
        tstart = 0
        tend = spks.shape[1]

        # Z-score neural activity
        spks_mean = spks.mean(axis=1, keepdims=True)
        spks_std = ((spks ** 2).mean(axis=1, keepdims=True) ** 0.5)
        spks_std[spks_std == 0] = 1
        spks -= spks_mean
        spks /= spks_std
        Vsv = PCA(n_components=128).fit_transform(spks.T)
        U = spks @ Vsv
        U /= (U ** 2).sum(axis=0) ** 0.5
        
        varexp_expl, varexp_expl_neurons, _ = peer_prediction(spks, xpos, ypos)
        print(f"Explainable variance: {varexp_expl:.3f}, {varexp_expl_neurons.mean():.3f}")
        
        # >>>>>>>>>>>>>>>>>>>>> Behavioral Data <<<<<<<<<<<<<<<<<<<<
        # replace pupil values during blinks with interpolated values
        if "light" in recording_type:
            percentile = float([entry['blink_baselined_threshold'] for entry in recording_dbs if entry['mname'] == mname and entry['blk']=="1"][0])
        else:
            percentile = 0.5
        pupil_x = npz_data['pupil_x_polar'] 
        pupil_y = npz_data['pupil_y_polar'] 
        blink = npz_data['blink_small']
        pupil_area = npz_data['pupil_area_small']
        pupil_x_interp = interpolate_pupil_blink(pupil_x, blink, percentile)
        pupil_y_interp = interpolate_pupil_blink(pupil_y, blink, percentile)
        pupil_yx = np.vstack((pupil_x_interp, pupil_y_interp)).T
        # check any nan in pupil_area, if so, interpolate 
        if np.isnan(pupil_area).any():
            pupil_area = helper_functions.interp_value(pupil_area[~np.isnan(pupil_area)], np.where(~np.isnan(pupil_area))[0], np.arange(len(pupil_area)))

        fullface_svd = zscore_behavior_data(npz_data['face_svd'].item()['movSVD'][0], axis=0)
        face_eye_svd = zscore_behavior_data(npz_data['face_eye_svd'].item()['movSVD'][0], axis=0)

        face_eye_pupilyx = zscore_behavior_data(np.hstack((face_eye_svd, pupil_yx)), axis=0)
        pupil_yx = zscore_behavior_data(pupil_yx, axis=0)
        pupil_area = zscore_behavior_data(pupil_area, axis=0)[:, np.newaxis]

        # Process each condition
        if "light" in recording_type:
            conditions = {
                "light": [pupil_area, pupil_yx, face_eye_pupilyx, face_eye_svd, fullface_svd],
            }
            input_names = {
                "light": ['pupil_area', 'pupil', 'face_eye_pupil', 'face_eye', 'fullface'],
            }
        elif recording_type == "dark":
            conditions = {
                "with_pupil": [pupil_area, pupil_yx, face_eye_pupilyx], # dark
                "without_pupil": [fullface_svd, face_eye_svd], # dark
            }
            input_names = {
                "with_pupil": ['pupil_area', 'pupil', 'face_eye_pupil'],
                "without_pupil": ['fullface', 'face_eye'],
            }
        else:
            print("Invalid recording type")
            return
        
        for condition, inputs in conditions.items():
            for input_data, name in zip(inputs, input_names[condition]):
                print(f"Processing: {name}")         
                reset_indices = []   
                data_splits_before_itest_kfold = []
                data_splits_after_itest_kfold = []
                data_splits_before_ytest = []
                data_splits_after_ytest = []
                data_splits_before_xtest = []
                data_splits_after_xtest = []
                data_splits_ytest = []
                y_pred_before_kfolds = []
                y_pred_after_kfolds = []
                y_pred_kfolds = []
                itest_sample_b_all = []
                data_splits_ineural_train, data_splits_ineural_val, data_splits_ineural_test = [], [], []
                for k_fold_idx in range(hyperparameters['k_folds']):
                    hyperparameters['k_fold_idx'] = k_fold_idx
                    hyperparameters['condition'] = condition
                    hyperparameters['input_data'] = name
                    hyperparameters['mname'] = mname
                    hyperparameters['n_kp'] = 5 if name in ['pupil', 'pupil_area'] else 250
                    hyperparameters['weight_decay'] = 1e-3 if name in ['pupil', 'pupil_area'] else 1e-4
                    hyperparameters['segment_length'] = 3350 if mname in ['FXa10', 'FXa8', 'FXa9', 'FXa1'] else 2350 # longer segments for 50 Hz recordings
                    blk_splits = process_blocks(mouse_dbs, Vsv, tcams, tneurals, cumframes_neural, cumframes_cam, input_data, hyperparameters, device)
                    if condition == "with_pupil":
                        blk_splits = blk_splits[1:] # carbachol only block data
                    if "light" in recording_type:
                        blk_splits = blk_splits[:1] # only first block for light recordings
                    print(f"{name}; fold{k_fold_idx+1}; n_blocks:{len(blk_splits)}")

                    # Initialize a dictionary to store each split category
                    data_splits = {
                        "X_train": [], "X_val": [], "X_test": [],
                        "Y_train": [], "Y_val": [], "Y_test": [],
                        "itrain_sample_b": [], "ival_sample_b": [], "itest_sample_b": [],
                        "itrain_cam": [], "ival_cam": [], "itest_cam": [],
                        "ineural_train": [], "ineural_val": [], "ineural_test": []
                    }

                    # Populate the dictionary with data from blk_splits
                    data_splits_before_itest = []
                    data_splits_after_itest = []
                    for i, blk in enumerate(blk_splits):
                        data_splits["X_train"].append(blk[0][0])
                        data_splits["X_val"].append(blk[1][0])
                        data_splits["X_test"].append(blk[2][0])
                        data_splits["Y_train"].append(blk[3][0])
                        data_splits["Y_val"].append(blk[4][0])
                        data_splits["Y_test"].append(blk[5][0])
                        data_splits["itrain_sample_b"].append(blk[6][0])
                        data_splits["ival_sample_b"].append(blk[7][0])
                        data_splits["itest_sample_b"].append(blk[8][0])
                        data_splits["itrain_cam"].append(blk[9][0])
                        data_splits["ival_cam"].append(blk[10][0])
                        data_splits["itest_cam"].append(blk[11][0])
                        data_splits["ineural_train"].append(blk[12])
                        data_splits["ineural_val"].append(blk[13])
                        data_splits["ineural_test"].append(blk[14])
                        # Set blk_end based on the length of the first test block (only once)
                        if condition == "without_pupil" and recording_type == "dark":
                            if i == 0:
                                blk_end = len(blk[5][0])
                                data_splits_before_itest.append(blk[14][0])
                            else:
                                data_splits_after_itest.append(blk[14][0])
                        elif condition == "with_pupil" and recording_type == "dark":
                            blk_end = 0
                            data_splits_before_itest.append([])
                            data_splits_after_itest.append(blk[14][0])
                        if "light" in recording_type:
                            blk_end = len(blk[5][0])
                            data_splits_after_itest.append([])
                            data_splits_before_itest.append(blk[14][0])

                    # Concatenate data along the appropriate axis                   
                    for key in ["itrain_cam", "ival_cam", "itest_cam"]:
                        data_splits[key] = [np.vstack(data_splits[key])]

                    # Train and Evaluate Model
                    n_in, n_out = data_splits["X_train"][0].shape[-1], Vsv.shape[-1]
                    y_pred, vexp, best_state_dict, best_val_ve = train_and_evaluate(data_splits["X_train"], data_splits["X_val"], data_splits["X_test"],                                                
                                                                                    data_splits["Y_train"], data_splits["Y_val"], data_splits["Y_test"], 
                                                                                    data_splits["itrain_sample_b"], data_splits["ival_sample_b"], 
                                                                                    data_splits["itest_sample_b"], n_in, n_out, hyperparameters, device)
                    # Results Storage
                    y_pred_kfolds.append(y_pred)
                    y_pred_before_kfolds.append(y_pred[:blk_end])
                    y_pred_after_kfolds.append(y_pred[blk_end:])

                    # Convert inner lists to numpy FIRST
                    np_list = []
                    for lst in data_splits['Y_test']:
                        np_list.append([to_numpy(x) for x in lst])
                    data_splits['Y_test'] = np_list
                    # flatten into a single list of length n_blocks * n_samples
                    data_splits['Y_test'] = [item for sublist in np_list for item in sublist]

                    # repeat for itest_sample_b
                    np_list = []
                    for lst in data_splits['itest_sample_b']:
                        np_list.append([to_numpy(x) for x in lst])
                    data_splits['itest_sample_b'] = np_list
                    # flatten into a single list of length n_blocks * n_samples
                    data_splits['itest_sample_b'] = [item for sublist in np_list for item in sublist]

                    data_splits_before_ytest.append(data_splits['Y_test'][:blk_end])
                    data_splits_after_ytest.append(data_splits['Y_test'][blk_end:])
                    data_splits_ytest.append(data_splits['Y_test'])
                    data_splits_before_xtest.append(data_splits['X_test'][0].to('cpu').numpy()[:blk_end])
                    data_splits_after_xtest.append(data_splits['X_test'][0].to('cpu').numpy()[blk_end:])
                    data_splits_before_itest_kfold.append(data_splits_before_itest)
                    data_splits_after_itest_kfold.append(data_splits_after_itest)
                    itest_sample_b_all.append(data_splits["itest_sample_b"])
                    data_splits_ineural_train.append(data_splits["ineural_train"])
                    data_splits_ineural_val.append(data_splits["ineural_val"])
                    data_splits_ineural_test.append(data_splits["ineural_test"][0])

                n_blocks = len(mouse_dbs)
                nonlinear_results[mname][name] = compute_variance_explained_kfolds(n_blocks, hyperparameters['k_folds'], \
                                                                                   condition, y_pred_kfolds, y_pred_before_kfolds, \
                                                                                    y_pred_after_kfolds, spks, U, \
                                                                                    data_splits_before_itest_kfold, \
                                                                                    data_splits_after_itest_kfold, \
                                                                                    data_splits_ineural_test, data_splits_before_ytest,\
                                                                                    data_splits_after_ytest, data_splits_ytest, \
                                                                                    cumframes_neural, varexp_expl_neurons, n_out, \
                                                                                    reset_indices)
                rastermap_output[mname][name] = {"before": {}, "after": {}, "U": U, 
                                                 "reset_indices": reset_indices, 
                                                 "cumframes_neural": cumframes_neural,
                                                 "tneural_start": tstart,
                                                 "tneural_end": tend,
                                                 "spks": spks, 
                                                 "spks_mean": spks_mean,
                                                 "spks_std": spks_std,
                                                 "data_splits_ineural_train": data_splits_ineural_train,
                                                 "data_splits_ineural_val": data_splits_ineural_val,
                                                 "data_splits_ineural_test": data_splits_ineural_test}
                rastermap_output[mname][name]["before"] = {"y_pred" : y_pred_before_kfolds,
                                            "y_test": data_splits_before_ytest,
                                            "itest": data_splits_before_itest_kfold,
                                            "x_test": data_splits_before_xtest,
                                            "itest_sample_b": itest_sample_b_all}
                rastermap_output[mname][name]["after"] = {"y_pred" : y_pred_after_kfolds,
                                            "y_test": data_splits_after_ytest,
                                            "itest": data_splits_after_itest_kfold,
                                            "x_test": data_splits_after_xtest,
                                            "itest_sample_b": itest_sample_b_all}
                
    return nonlinear_results, rastermap_output

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    if isinstance(x, (list, tuple)):
        return np.array([to_numpy(v) for v in x])

    return np.asarray(x)
