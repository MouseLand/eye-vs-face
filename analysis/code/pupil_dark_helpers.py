import numpy as np
import os, sys

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
