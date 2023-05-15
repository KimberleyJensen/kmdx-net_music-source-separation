"""
eval.py

Evaluate all metrics for a given separation.
Usage:
  $ python eval.py --references-folder REF_DIR        \
                   --separations-folder SEP_DIR       \
                   --output RESULT.mat

Stefan Uhlich, Sony RDC
"""
import argparse
from multiprocessing import Pool
import musdb
import numpy as np
import os
from scipy.io import savemat
import stempeg

from metrics import MeanSDR_BSSEval4, MedianSDR_BSSEval4
from metrics import SDR_BSSEval3, MeanFramewiseSDR_BSSEval3, MedianFramewiseSDR_BSSEval3
from metrics import GlobalMSE, MeanFramewiseMSE, MedianFramewiseMSE
from metrics import GlobalMAE, MeanFramewiseMAE, MedianFramewiseMAE
from metrics import GlobalSDR, MeanFramewiseSDR, MedianFramewiseSDR
from metrics import GlobalSISDR, MeanFramewiseSISDR, MedianFramewiseSISDR

parser = argparse.ArgumentParser(description='Eval parser')

parser.add_argument('--references-folder', type=str,
                    required=True, help='path to musdb (groundtruth)')
parser.add_argument('--separations-folder', type=str,
                    required=True, help='path to separations')
parser.add_argument('--output', type=str,
                    required=True, help='path to MAT file where metrices are stored')
parser.add_argument('--num-processes', type=int,
                    required=True, help='number of processes for multiprocessing')
args = parser.parse_args()

# load mus data
mus = musdb.DB(root=args.references_folder, subsets='test', is_wav=True)

# define instruments
instruments = ['drums', 'bass', 'other', 'vocals']

# define metrics
global_metrics = [MeanSDR_BSSEval4, MedianSDR_BSSEval4,
                  SDR_BSSEval3, MeanFramewiseSDR_BSSEval3, MedianFramewiseSDR_BSSEval3,
                  GlobalMSE, GlobalMAE, GlobalSDR, GlobalSISDR]
framewise_metrics = [MeanFramewiseMSE, MedianFramewiseMSE,
                     MeanFramewiseMAE, MedianFramewiseMAE,
                     MeanFramewiseSDR, MedianFramewiseSDR,
                     MeanFramewiseSISDR, MedianFramewiseSISDR]

metrics = []
metrics.extend([metric() for metric in global_metrics])
metrics.extend([metric(win=44100, hop=44100) for metric in framewise_metrics])

# define output dictionary
n_metrics = len(global_metrics) + len(framewise_metrics)
results = {instr: np.zeros((n_metrics, 50)) for instr in instruments}
results['metrics'] = [str(metric.__class__) for metric in metrics]


# define evaluation function
def _eval(track):
    print(f'Evaluating track {track.path}')
    # create return variable
    res = {instr: np.zeros((n_metrics, )) for instr in instruments}

    # get reference (order: 'drums', 'bass', 'other', 'vocals')
    references = track.stems[1:]

    # get separation
    sep_path = os.path.dirname(track.path).replace(args.references_folder,
                                                   args.separations_folder)
    separations = [stempeg.read_stems(os.path.join(sep_path, instr + '.wav'), ffmpeg_format="s16le")[0] \
        for instr in instruments]
    separations = np.array(separations)

    # apply metrics
    for idx_metric, metric in enumerate(metrics):
        m = metric(references, separations)
        for idx_instr, instr in enumerate(instruments):
            res[instr][idx_metric] = m[idx_instr]

    return res


# use multi-processing to evaluate tracks in parallel
with Pool(args.num_processes) as tp:
    _results = tp.map(_eval, mus)

# store results
for idx_track in range(len(mus)):
    for idx_instr, instr in enumerate(instruments):
        results[instr][:, idx_track] = _results[idx_track][instr]

# store matlab file
savemat(args.output, results)

