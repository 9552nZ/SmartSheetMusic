import numpy as np
import score_following
import pickle
import utils_audio_transcript as utils
import pandas as pd
from multiprocessing import Process, Manager
from operator import setitem
from os.path import exists

def run_matcher_evaluation(wd, filename_mid, config_matcher, idx_config_matcher, alignement_stats_all, idx_mid):
    '''
    This function runs the matcher evaluation procedure for 
    one track/one config_matcher/multiple corruptions. 
    '''
    # Run the matcher evaluation
    matcher_evaluator = score_following.MatcherEvaluator(wd, filename_mid, config_matcher=config_matcher)
    matcher_evaluator.main_evaluation()
    
    # Augment the alignement_stats with the track name and the config number
    map(lambda x: setitem(x, 'idx_config_matcher', idx_config_matcher), matcher_evaluator.alignement_stats)
    map(lambda x: setitem(x, 'filename_mid', filename_mid), matcher_evaluator.alignement_stats)   
    
    # Replace the Manager.list item with the alignement_stats
    alignement_stats_all[idx_mid] = matcher_evaluator.alignement_stats
    
    # Pickle the output of the matcher evaluation
    filename_pkl = utils.change_file_format(filename_mid, '.mid', '.pkl', append = '_matcher_evaluator') 
    output = open(wd+filename_pkl, 'wb')
    pickle.dump(matcher_evaluator, output)
    output.close()    
    
    return()

def save_alignment_stats(wd, filename_stats_pkl, alignement_stats_new):
    '''
    Open the existing stats (if any), read, and append the new stats at the end.
    '''
    
    # Check if there is a file stored to disk already
    alignement_stats = []
    if exists(wd+filename_stats_pkl):
        with open(wd+filename_stats_pkl,'rb') as rfp: 
            alignement_stats = pickle.load(rfp)
    
    # Append the new stats
    alignement_stats.extend(alignement_stats_new)
    
    # Pickle the updated list
    with open(wd+filename_stats_pkl,'wb') as wfp:
        pickle.dump(alignement_stats, wfp)                   
    
    return()

def format_alignment_stats(wd, filename_stats_pkl):
    '''
    This function reformats the pickled alignment stats as a dataframe.    
    '''
    
    # Load the pickled data
    with open(wd+filename_stats_pkl,'rb') as rfp: 
        alignement_stats = pickle.load(rfp)
    
    # Flatten/extract the metrics we need
    idxs_config_matcher = map(lambda x: x['idx_config_matcher'], alignement_stats)
    idxs_config_corruption = map(lambda x: x['idx_config_corruption'], alignement_stats)
    filenames_mid = map(lambda x: x['filename_mid'], alignement_stats)
    means_error = map(lambda x: x['mean_error'], alignement_stats)
    means_abs_error  = map(lambda x: x['mean_abs_error'], alignement_stats)
    prctile0_error = map(lambda x: x['prctile_error'][0], alignement_stats)
    prctile1_error = map(lambda x: x['prctile_error'][1], alignement_stats)
    prctile5_error = map(lambda x: x['prctile_error'][2], alignement_stats)
    prctile50_error = map(lambda x: x['prctile_error'][3], alignement_stats)
    prctile95_error = map(lambda x: x['prctile_error'][4], alignement_stats)
    prctile99_error = map(lambda x: x['prctile_error'][5], alignement_stats)
    prctile100_error = map(lambda x: x['prctile_error'][6], alignement_stats)

    
    df = {'idxs_config_matcher':idxs_config_matcher,
          'idxs_config_corruption':idxs_config_corruption,
          'filenames_mid':filenames_mid,
          'means_error':means_error, 
          'means_abs_error':means_abs_error,
          'prctiles0_error':prctile0_error,
          'prctiles1_error':prctile1_error,
          'prctiles5_error':prctile5_error,
          'prctiles50_error':prctile50_error,
          'prctiles95_error':prctile95_error,
          'prctiles99_error':prctile99_error,
          'prctiles100_error':prctile100_error,
          }
    
    # Output as dataframe
    df = df = pd.DataFrame(df)
    
    return(df)
    
 
if __name__ == '__main__':
    
    wd = "C:\Users\Alexis\Business\SmartSheetMusic\Samples\TestScoreFollowing/"
    
    # List of all the midi files in the evaluation universe
    filenames_mid = [ 
        "Bach_BWV871-02_002_20090916-SMD.mid",
        "Bach_BWV888-02_008_20110315-SMD.mid",
        "Bartok_SZ080-02_002_20110315-SMD.mid",
#         "Beethoven_Op027No1-02_003_20090916-SMD.mid",
        "Chopin_Op028-01_003_20100611-SMD.mid",
    ]
    
    # List the concurrent configs for the matcher
    configs_matcher = []
    for diag_cost in np.arange(0.8, 2.2, 0.2):
#     for diag_cost in np.arange(2.0, 2.2, 0.2):
        configs_matcher.append({'diag_cost':diag_cost})
    
    # Loop over the configs
    for k in range(len(configs_matcher)):
        
        # Parallelise the run (split the tracks across different workers)        
        processes = []
        manager = Manager()
        # alignement_stats_all is a Manager.list placeholder to gather all the stats (for one config)
        alignement_stats_all = manager.list(range(len(filenames_mid)))
                
        for m in range(len(filenames_mid)):
            p = Process(target=run_matcher_evaluation, args=(wd, filenames_mid[m], configs_matcher[k], k, alignement_stats_all, m))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
        
        # Convert the Manager.list to list and flatten    
        alignement_stats_all = [item for sublist in list(alignement_stats_all) for item in sublist] 
        
        # Pickle the stats to disk    
        save_alignment_stats(wd, 'alignments_stats.pkl', alignement_stats_all)            
        