"""
define methods to load datasets
"""

import os
import warnings

def get_audio_files(main_path, dataset, dataset_type=None):
    """
    Return a list with full path of all matching audio files
    For the moment only drums.

    DATASETS:
    - DSD100
        - Dev
        - Test
    - IDMT
        - RealDrum
        - TechnoDrum
        - WaveDrum
    - ENST
        - drummer_1
        - drummer_2
        - drummer_3

    :param dataset:
    :return:
    """
    audio_files = []

    if dataset is 'DSD100':
        if dataset_type is 'dev':
            data_folder_path = os.path.join(main_path,'DSD100/Sources/Dev')
        elif dataset_type is 'test':
            data_folder_path = os.path.join(main_path,'DSD100/Sources/Test')
        else: # all
            data_folder_path = main_path
        for root, dir, files in os.walk(data_folder_path):
            for f in files:
                instrument, extension = os.path.splitext(f)
                if 'drums' in instrument and "wav" in extension:
                    audio_files.append(os.path.join(root, f))

    elif dataset is 'IDMT':
        data_folder_path = os.path.join(main_path, 'IDMT-SMT-DRUMS-V2/audio')
            # Only tracks which contain "MIX" in the name, audio folder
        for root, dir, files in os.walk(data_folder_path):
            for f in files:
                name, extension = os.path.splitext(f)
                if dataset_type in ['RealDrum', 'TechnoDrum', 'WaveDrum']:
                    if dataset_type in name and 'MIX' in name and "wav" in extension:
                        audio_files.append(os.path.join(root, f))
                else: # all
                    if 'MIX' in name and "wav" in extension:
                        audio_files.append(os.path.join(root, f))

    elif dataset is 'ENST':
        # only dry_mix folder
        if dataset_type in ['drummer_1', 'drummer_2','drummer_3']:
            data_folder_path = os.path.join(main_path, 'ENST-drums-public', dataset_type, 'audio/dry_mix')
            for root, dir, files in os.walk(data_folder_path):
                for f in files:
                    extension = os.path.splitext(f)[-1]
                    if "wav" in extension:
                        audio_files.append(os.path.join(root, f))
        else: # all
            for drummer in ['drummer_1', 'drummer_2', 'drummer_3']:
                data_folder_path = os.path.join(main_path, 'ENST-drums-public', drummer, 'audio/dry_mix')
                for root, dir, files in os.walk(data_folder_path):
                    for f in files:
                        extension = os.path.splitext(f)[-1]
                        if "wav" in extension:
                            audio_files.append(os.path.join(root, f))
    else:
        warnings.warn('dataset not known: '+dataset)

    return audio_files




def get_audio_files_DSD(main_path, mixtures=False, dataset_instrument='drums', dataset_type='Test'):
    """
    Return a list with full path of all matching audio files

    DATASETS:
    - DSD100
        - Dev
        - Test
    """
    audio_files = []
    instruments = ['bass', 'drums', 'other', 'vocals']
    dataset_types = ['Dev', 'Test']


    if dataset_type not in dataset_types:
        warnings.warn('dataset type not known: ' + dataset_type)
        return

    if mixtures:
        dataset_instrument = 'mixture'
        folder = 'Mixtures'
    else:
        if dataset_instrument not in instruments:
            warnings.warn('instrument not known: ' + dataset_instrument)
            return
        folder = 'Sources'

    data_folder_path = os.path.join(main_path, 'DSD100', folder, dataset_type)
    for root, dir, files in os.walk(data_folder_path):
        for f in files:
            instrument, extension = os.path.splitext(f)
            if instrument==dataset_instrument and extension=='.wav':
                audio_files.append(os.path.join(root, f))

    return audio_files


def get_audio_files_librispeech(main_path, dataset_type='Test'):
    """
    test set
    :param main_path:
    :return:
    """
    audio_files = []

    dataset_types = ['Dev', 'Test']


    if dataset_type not in dataset_types:
        warnings.warn('dataset type not known: ' + dataset_type)
        return

    dataset_type_folder = None
    if dataset_type == 'Test':
        dataset_type_folder = 'test-clean'
    elif dataset_type == 'Dev':
        dataset_type_folder = 'dev-clean'

    data_folder_path = os.path.join(main_path, 'LibriSpeech', dataset_type_folder)
    for root, dir, files in os.walk(data_folder_path):
        for f in files:
            filename, extension = os.path.splitext(f)
            if extension == ".flac":
                audio_files.append(os.path.join(root, f))
    return audio_files


