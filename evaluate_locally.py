import os
from tqdm.auto import tqdm
import numpy as np
import soundfile

from my_submission.aicrowd_wrapper import AIcrowdWrapper
from local_evaluator.sisec21_evaluation.metrics import GlobalSDR


def check_data(datafolder):
    """
    Checks if the data is downloaded and placed correctly
    """
    inputsfolder = datafolder
    groundtruthfolder = datafolder
    dl_text = ("Please download the public data from"
               "\n https://www.aicrowd.com/challenges/music-demixing-challenge-2023/problems/robust-music-separation/dataset_files"
               "\n And unzip it with ==> unzip <zip_name> -d public_dataset")
    if not os.path.exists(datafolder):
        raise NameError(f'No folder named {datafolder} \n {dl_text}')
    if not os.path.exists(groundtruthfolder):
        raise NameError(f'No folder named {groundtruthfolder} \n {dl_text}')

def calculate_metrics(ground_truth_path, prediction_path):
    """
    Calculate metrics for one prediction and ground truth pair for all instruments
    """
    metric = GlobalSDR()
    # read in all WAV files for the four instruments
    gt = []
    se = []
    instruments = ['bass', 'drums', 'other', 'vocals']
    for instr in instruments:
        _gt, _ = soundfile.read(os.path.join(ground_truth_path, instr + '.wav'))
        _se, _ = soundfile.read(os.path.join(prediction_path, instr + '.wav'))
        gt.append(_gt)
        se.append(_se)
    gt = np.stack(gt) # shape: n_sources x n_samples x n_channels
    se = np.stack(se) # shape: n_sources x n_samples x n_channels
    # compute scores
    music_scores = metric(gt, se)

    metrics = {f"sdr_{instr}" : float(score)  for instr, score in zip(instruments, music_scores)}
    metrics['mean_sdr'] = float(np.mean(music_scores))
    return metrics

def evaluate(LocalEvalConfig):
    """
    Runs local evaluation for the model
    Final evaluation code is the same as the evaluator
    """
    datafolder = LocalEvalConfig.DATA_FOLDER
    
    check_data(datafolder)
    inputsfolder = datafolder
    groundtruthfolder = datafolder

    preds_folder = LocalEvalConfig.OUTPUTS_FOLDER

    model = AIcrowdWrapper(predictions_dir=preds_folder, dataset_dir=datafolder)
    folder_names = os.listdir(datafolder)

    for fname in tqdm(folder_names, desc="Demixing"):
        model.separate_music_file(fname)

    # Evalaute metrics
    all_metrics = {}
    
    folder_names = os.listdir(datafolder)
    for fname in tqdm(folder_names, desc="Calculating scores"):
        ground_truth_path = os.path.join(groundtruthfolder, fname)
        prediction_path = os.path.join(preds_folder, fname)
        all_metrics[fname] = calculate_metrics(ground_truth_path, prediction_path)
        
    
    metric_keys = list(all_metrics.values())[0].keys()
    metrics_lists = {mk: [] for mk in metric_keys}
    for metrics in all_metrics.values():
        for mk in metrics:
            metrics_lists[mk].append(metrics[mk])
    
    print("Evaluation Results")
    results = {key: np.mean(metric_list) for key, metric_list in metrics_lists.items()}
    for k,v in results.items():
        print(k,v)


if __name__ == "__main__":
    # change the local config as needed
    class LocalEvalConfig:
        DATA_FOLDER = './public_dataset/MUSDB18-7-WAV/test/'
        OUTPUTS_FOLDER = './evaluator_outputs'

    outfolder=  LocalEvalConfig.OUTPUTS_FOLDER
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    
    evaluate(LocalEvalConfig)
