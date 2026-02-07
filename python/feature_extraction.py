
#!/usr/bin/env python3
'''
Python script to extract features from emg data

Authors: Brian R. Mullen, Sero Toriano Parel, Revati Jadhav, Carrie Clark, Philip Nelson
Date: 2025-10-15

 
Examples:

python feature_extraction.py -i ../data/emg_data/ -sf

'''

import os
import glob
import sys

import pandas as pd
import numpy as np

from scipy.fft import fft, fftfreq

from datetime import datetime
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Import the data loader from the generic_neuromotor_interface package
from generic_neuromotor_interface.explore_data.load import load_data



def get_task_dataset_paths(task: str) -> list[str]:
    # Only return files that contain the task name in their filename

    folder = os.path.expanduser(task)
    datasets = glob.glob(os.path.join(folder, '*.hdf5'))
    
    return [d for d in datasets if task in d]


def get_gesture_prompt_times(prompts,
                             timestamps):
    '''
    gets times each different stage was used

    Arguments:
        prompts: prompt information
        timestamps: timestamps from experiment

    Returns:
       stage_labels: 1D array with gesture in time of when prompts were given
    '''
    gesture_labels = np.full_like(timestamps, '', dtype=object)
    n_gestures = 0

    if not prompts.empty:
        prompt_times = prompts['time'].values
        prompt_names = prompts['name'].values
        prompt_idx = 0
        # For each sample, check if a gesture occurs at that timestamp
        for i, t in enumerate(timestamps):
            while prompt_idx + 1 < len(prompt_times) and prompt_times[prompt_idx + 1] <= t:
                prompt_idx += 1
            if prompt_times[prompt_idx] == t:
                gesture_labels[i] = prompt_names[prompt_idx]
                n_gestures += 1
    return gesture_labels, n_gestures

def get_gesture_stage_times(stages,
                             timestamps):
    '''
    gets times each different stage was used

    Arguments:
        stages: stage information in dataframe with 'start', 'end', 'name' columns
        timestamps: timestamps from experiment

    Returns:
       stage_labels: 1D array with stage in place of when it was used
    '''

    if not stages.empty:
        #for each timestamp check if it belongs to each stage
        conditions = [np.logical_and((stages.start[i] <= timestamps),(stages.end[i] >= timestamps)) for i in range(len(stages))]
        choices = stages.name
        #for each timestamp select the corresponding stage name
        stage_labels = np.select(conditions,choices)

    return stage_labels


def get_zscore(emg, axis=0):
    '''
    make zscores based on emg data, normalizes the EMG signals

    Arguments:
        emg: emg data across 16 channels
        axis: which axis of the array are normalized against

    Returns:
       zscore: data normalized to stdev (with mean=0)
    '''
    return (emg - np.mean(emg, axis=axis))/np.std(emg, axis=axis)


def get_large_event_array(emg: np.array, 
               gesture_name:str, 
               gest_indices:int = None, 
               sample_rate:float = 2000, 
               lowend:float = -0.05,
               window_size: np.array = np.array([-1,1]), 
               trim_window:np.array= np.array([-0.01,.1]),
               show: bool = False,
               savedir: str = None,
               file: str = None):

    '''
    Takes emg data and gesture indices to output an array of aligned and trimmed around the 
    largest event dataset
    
    Arguments:
        emg: emg data across 16 channels
        gesutre_name: name of the gesture of the output array 
        gest_indices: Index of when this gesture was prompted
        sample_rate: sample rate of emg data
        lowend: lowest temporal allowance to shift the event time
        window_size: window to search for large event
        trim_window: output array will be only this time frame, with 0 being the largest event
        show: show the aligned plots of zscore and emg
        savedir: directory of where you will save the file
        file: file from which the data was extracted

    Returns:
       emg_trimmed: emg data aligned and trimmed 
       zscore_trimmed: zscore data aligned and trimmed 
       shifts: number of indices the prompt was shifted
    '''

    save = False
    if savedir is not None:
        save = True
    
    window_size = np.array(window_size)*sample_rate 
    trim_window = np.array(trim_window)*sample_rate
    shifts = np.zeros_like(gest_indices)

    # time oif view window, with 0 at the event
    time_rel = np.arange(window_size[0], window_size[1])/sample_rate 
    lowcut = np.where(time_rel > lowend)[0][0]
    emg_windowed = np.zeros((len(gest_indices), int(np.diff(window_size)[0]), emg.shape[1]))
    # loop through waveforms of gestures
    for i, index in enumerate(gest_indices): # loop through waveforms of gestures
        start = int(window_size[0] + index) #non-shifted search window, 0 is prompt time
        stop = int(window_size[1] + index)
        emg_windowed[i,:,:] = emg[start:stop,:]

    zscore_windowed = get_zscore(emg_windowed, axis=(0,1))
    
    zscore_trimmed = np.zeros((len(gest_indices), int(np.diff(trim_window)[0]), 
                                emg.shape[1])) * np.nan
    emg_trimmed = np.zeros_like(zscore_trimmed) * np.nan

    for i, index in enumerate(gest_indices): # loop through waveforms of gestures
        start = int(window_size[0] + index) #non-shifted search window, 0 is prompt time
        stop = int(window_size[1] + index)
    
        # determine big event, independent of channel
        try:  # first event that reaches a zscore of 3 (3 * stdev)
            large_event = np.where(np.max(zscore_windowed[i, lowcut:, :], axis=1) > 3)[0][0]
        except IndexError:
            try:
                # else first event that reaches a zscore of 1.65, alpha=0.05
                large_event = np.where(np.max(zscore_windowed[i, lowcut:, :], axis=1) > 1.65)[0][0]
                print('\t\t\tLower threshold used', gesture_name, i)
            except IndexError:
                print('\t\t\tCould not find a large event', gesture_name, i)
                continue
        except Exception as e:
            print('\t\t\tError finding large event', gesture_name, i, e)
            continue

        # new index relative to the start index from this iteration        
        new_index = start + lowcut + large_event 
        shift = new_index - index # how much it shifts
        shifts[i] = shift
        
        startl = int(lowcut+large_event+trim_window[0]) #shifted indices
        stopl = int(lowcut+large_event+trim_window[1])
        window_emg = emg_windowed[i, startl:stopl, :]
        window_z = zscore_windowed[i, startl:stopl, :]
        target_len = emg_trimmed.shape[1]

        if window_emg.shape[0] == target_len:
            # emg value, shifted, 0 is first large muscle movement around prompt time
            emg_trimmed[i, :, :] = window_emg
            # zscore, shifted, 0 is first large muscle movement around prompt time
            zscore_trimmed[i, :, :] = window_z
        elif window_emg.shape[0] > 0 and window_emg.shape[0] < target_len:
            # Pad short windows with zeros to match expected shape
            print('\t\t\tEvent at end of the windowed area', gesture_name, i)
            emg_trimmed[i, :window_emg.shape[0], :] = window_emg
            zscore_trimmed[i, :window_z.shape[0], :] = window_z
        else:
            # Empty or invalid slice; skip this gesture instance
            print('\t\t\tWindow too short; skipping', gesture_name, i)
            continue

    if save:
        fig, axs = plt.subplots(1,2 , figsize=(2.5,2.5))
        # time oif view window, with 0 at the event
        time_rel = np.arange(trim_window[0], trim_window[1])/sample_rate 
        for i in range(16):
            channelmean = np.nanmean(emg_trimmed[:,:,i], axis=0)
            channelstd = np.nanstd(emg_trimmed[:,:,i], axis=0)
            axs[0].plot(time_rel, channelmean+i*100, color='k')
            axs[0].fill_between(time_rel, channelmean-channelstd+i*100, channelmean+channelstd+i*100, color=(0,i/16,1))
            channelmean = np.nanmean(zscore_trimmed[:,:,i], axis=0)
            channelstd = np.nanstd(zscore_trimmed[:,:,i], axis=0)
            axs[1].plot(time_rel, channelmean+i*10, color='k')
            axs[1].fill_between(time_rel, channelmean-channelstd+i*10, channelmean+channelstd+i*10, color=(1,0,i/16))
        axs[0].set_title('zscore')    
        axs[0].set_yticks(np.arange(16)*100)   
        axs[0].set_yticklabels(np.arange(16))   
        axs[0].set_ylabel('channel') 
        axs[0].set_xticks([0,0.05,0.1])    
        axs[0].set_xlabel('time(s)')

        axs[1].set_title('EMG')    
        axs[1].set_yticks(np.arange(16)*10)   
        axs[1].set_yticklabels([])   
        axs[1].set_xlabel('time(s)')
        axs[1].set_xticks([0,0.05,0.1])    
        # axs[1].set_yticklabels(np.arange(16))   
        # plt.tight_layout()
        plt.savefig(os.path.join(savedir, '{0}-{1}.png'.format(gesture_name, file[-25:-5])))
        if show:
            plt.show()
        else:
            plt.close()

    return emg_trimmed, zscore_trimmed, shifts


def fft_windowed(timeseries, sample_rate, hanning=True, show=False):

    '''
    Takes timeseries data to output frequency characteristics
    
    Arguments:
        timeseries: emg data
        sample_rate: sample rate of emg data
        hanning: boolean to apply hanning window
        show: show the fft, 

    Returns:
        highfreq: highest power frequency 
        maxpower: power at highfreq 
        freq_range: list 2 values, half max full width of fourier frequencies
    '''

    if hanning:
        hanwin = np.hanning(timeseries.shape[0])
        signal = timeseries * hanwin
    else:
        signal = timeseries
    fourier_signal = np.convolve(np.abs(fft(signal)), np.ones(10)/10, mode='same')
    freq = fftfreq(len(signal), d=1/sample_rate)
    fourier_signal = fourier_signal[freq>0]
    freq = freq[freq>0]
    
        
    maxarg = np.argmax(fourier_signal)
    maxpower = fourier_signal[maxarg]
    highfreq = freq[maxarg]
    mphw = np.where(fourier_signal > maxpower/2)[0]
    
    ranges = [[]]
    for val in mphw:
        if not ranges[-1] or ranges[-1][-1] == val-1:
            ranges[-1].append(val)
        else:
            ranges.append([val])
            
    if len(ranges) > 1:
        biggest = 0
        for r, ran in enumerate(ranges):
            if (ran[0] < highfreq) & (ran[-1] > highfreq):
                ranges = [ran]
                break

    freq_range = freq[[ranges[0][0], ranges[0][-1]]]

    if show:
        plt.plot(freq, fourier_signal, color='k', label='FFT')
        plt.scatter(highfreq, maxpower, color='red', label='Max')
        plt.hlines(maxpower/2, freq_range[0], freq_range[1], color='blue', label='range')
        plt.legend()
        plt.xlabel('freq')
        plt.ylabel('power')
        plt.show()

    return highfreq, maxpower, freq_range
    

def get_threshold_events(timeseries, threshold):

    '''
    Takes timeseries data and outputs the number times it goes above this value
    continuous indices only get counted once
    
    Arguments:
        timeseries: emg data
        threshold: threshold to count events, those above get counted

    Returns:
        ranges = list of list of indices when its above the threshold
    '''


    indices = np.where(timeseries > threshold)[0]
    if len(indices) > 0:
        ranges = [[]]
        for val in indices:
            if not ranges[-1] or ranges[-1][-1] == val-1:
                ranges[-1].append(val)
            else:
                ranges.append([val])
    else:
        ranges = None
    return ranges

def column_names(name, n_channel):
    '''
    Takes name and produces column names across n channels
    
    Arguments:
        name: attribute name
        nchannels: number of channels

    Returns:
        column attribute with channel number
    '''
    columns = []
    for i in range(n_channel): columns.append("ch{0}_{1}".format(str(i).zfill(2), name))
    return columns


def process_user_file(file: str, savefig: bool, fps: int, window_size: np.array, trim_window: np.array, data_folder: str):
    basename = os.path.basename(file)  # Get just the filename (not the full path)
    print(f"\nProcessing new file: {basename}")  # Print when starting a new file

    parts = basename.split('_')
    user_number = parts[3]  # Extract user_number from filename

    # We'll load the file using the `load_data` utility function.
    # Load the EMG data and associated info from single .hdf5 file
    data = load_data(file)
    emg = data.emg         # EMG signal, shape: (n_samples, 16)
    time_array = data.time       # Timestamps for each sample
    prompts = data.prompts # Dataframe of gesture events (name, time)
    stages = data.stages   # Dataframe of stage events (start, end, name)

    n_samples, n_channels = emg.shape

    print('\t Getting gestures')
    gesture_labels, n_gestures = get_gesture_prompt_times(prompts, time_array)
    print('\t\t Found {} gestures'.format(n_gestures))
    print('\t Getting stages')
    stage_labels = get_gesture_stage_times(stages, time_array)

    all_gestures = np.unique(gesture_labels)

    #set up columns
    rms_list = column_names('rms', n_channels)
    maxabs_list = column_names('maxabs', n_channels)
    mav_list = column_names('mav', n_channels)

    peak_freq_list = column_names('fft-peakfreq', n_channels)
    max_power_list = column_names('fft-maxpower', n_channels)
    high_freq_list = column_names('fft-highfreq', n_channels)
    low_freq_list = column_names('fft-lowfreq', n_channels)
    halfwidth_list = column_names('fft-halfwidth', n_channels)

    thresh3_list = column_names('thresh3-events', n_channels)
    thresh2_list = column_names('thresh2-events', n_channels)

    full_list = []
    full_list.extend(rms_list)         
    full_list.extend(maxabs_list)         
    full_list.extend(mav_list)         
    full_list.extend(peak_freq_list)         
    full_list.extend(max_power_list)         
    full_list.extend(high_freq_list)         
    full_list.extend(low_freq_list)         
    full_list.extend(halfwidth_list)         
    full_list.extend(thresh3_list)  
    full_list.extend(thresh2_list)

    # set up current DataFrame for this participant
    current_DF = pd.DataFrame(np.zeros((n_gestures, len(full_list)))*np.nan, columns=full_list)

    total_gestures = 0
    for gesture in all_gestures:
        if gesture == '': # skip empty gestures
            continue
        print('\t\t', gesture)

        gest_indices = np.where(gesture_labels == gesture)[0] # find index of these gestures
        # print(stage_labels[gest_indices])

        gesture_n = len(gest_indices)
        total_gestures += gesture_n

        current_indices = np.arange(total_gestures-gesture_n, total_gestures).astype(int)

        current_DF.loc[current_indices, ['gesture']] = gesture
        current_DF.loc[current_indices, ['stage']] =  stage_labels[gest_indices]
        current_DF.loc[current_indices, ['user']] = user_number

        # process data splits separately
        if savefig:
            emg_trimmed, zscore_trimmed, shifts = get_large_event_array(emg, 
                                           gesture_name=gesture, 
                                           gest_indices=gest_indices, 
                                           sample_rate=fps, 
                                           lowend = -0.05,
                                           window_size=window_size, 
                                           trim_window=trim_window,
                                           show=False,
                                           savedir=data_folder,
                                           file=file)
        else:
            emg_trimmed, zscore_trimmed, shifts = get_large_event_array(emg, 
                                           gesture_name=gesture, 
                                           gest_indices=gest_indices, 
                                           sample_rate=2000, 
                                           lowend = -0.05,
                                           window_size=window_size, 
                                           trim_window=trim_window,
                                           show=False,
                                           savedir=None,
                                           file=file)
        
        # Extract feature here
        
        # --- Find the correct metadata row for this event ---
        # Each event should fall within a start/end interval in the metadata
        
        rms_emg = np.sqrt(np.mean(emg_trimmed ** 2, axis=1))  # Root Mean Square
        maxabs_emg = np.max(np.abs(emg_trimmed), axis=1)  # Root Mean Square
        mav_emg = np.mean(np.abs(emg_trimmed), axis=1)

        for t, curr in enumerate(current_indices):
            curr = int(curr)
            
            for i in range(16):
                ranges = get_threshold_events(zscore_trimmed[t,:,i], threshold=3)
                if ranges != None:
                    current_DF.loc[curr, [thresh3_list[i]]] = len(ranges)
                else:
                    current_DF.loc[curr, [thresh3_list[i]]] = 0

                ranges = get_threshold_events(zscore_trimmed[t,:,i], threshold=2)
                if ranges != None:
                    current_DF.loc[curr, [thresh2_list[i]]] = len(ranges)
                else:
                    current_DF.loc[curr, [thresh2_list[i]]] = 0
                try:
                    highfreq, maxpower, freq_range = fft_windowed(emg_trimmed[t,:,i], sample_rate=2000, hanning=True)
                except:
                    print('\t\t\t\tCould not find fft information for index {0}, channel {1}'.format(t, i))
                    highfreq = np.nan
                    maxpower = np.nan
                    freq_range = [np.nan, np.nan]

                current_DF.loc[curr, rms_list[i]] = rms_emg[t, i]
                current_DF.loc[curr, maxabs_list[i]] = maxabs_emg[t, i]
                current_DF.loc[curr, mav_list[i]] = mav_emg[t, i]

                current_DF.loc[curr, peak_freq_list[i]] = highfreq
                current_DF.loc[curr, max_power_list[i]] = maxpower
                current_DF.loc[curr, high_freq_list[i]] = freq_range[1]
                current_DF.loc[curr, low_freq_list[i]] = freq_range[0]
                current_DF.loc[curr, halfwidth_list[i]] = np.diff(freq_range)
    print("Processed {} gestures".format(total_gestures))
    return current_DF


if __name__ == '__main__':

    import argparse
    import time
    import datetime

    # Argument Parsing
    # -----------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input_directory', type = str,
        required = True, 
        help = 'path to the director of .hdf5 files')
    ap.add_argument('-sf', '--savefig', action='store_true',
        default=False,
        help='Boolean indicating that figures will be saved')
    ap.add_argument('-f', '--fps', type = int,
        default=2000,  
        help = 'frames per second for data collection')
    ap.add_argument('-s', '--split', type = str,
        default='personal',  
        help = '"personal" or "universal" split')
    args = vars(ap.parse_args())


    # set seed here to ensure all data is the same between computers
    np.random.seed(112617)

    DATA_FOLDER = args['input_directory']
    savefig = args['savefig']
    fps = args['fps']
    split = args['split']
    # assert (split=='universal') | (split=='personal') 'Split needs to be defined as either "universal" or "personal" '

    files = get_task_dataset_paths(DATA_FOLDER)

    # hardcoded features 
    sample_rate = 2000
    trim_window = np.array([-0.01,0.1])
    window_size = np.array([-1,1])

    any_data_processed = False
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(process_user_file)(
            file, savefig, fps, window_size, trim_window, DATA_FOLDER
        )
        for file in files
    )

    # Filter out any None results
    results = [df for df in results if df is not None]
    any_data_processed = len(results) > 0

    if any_data_processed:
        final_df = pd.concat(results, ignore_index=True)

    if any_data_processed:
        final_df.to_csv(os.path.join(DATA_FOLDER, 'features_emg_data.csv'))
    else:
        print("No data processed; skipping CSV export.")
