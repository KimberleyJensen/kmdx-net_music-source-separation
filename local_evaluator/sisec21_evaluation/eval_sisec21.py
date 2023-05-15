"""
Evaluate separation using signal-to-distiortion ratio (SDR)

Stefan Uhlich, Sony RDC
"""
import argparse
import numpy as np
import os
import soundfile as sf

from metrics import GlobalSDR

parser = argparse.ArgumentParser(description='Eval parser')
parser.add_argument('--reference-folder', type=str,
                    required=True, help='path to single song (groundtruth)')
parser.add_argument('--separation-folder', type=str,
                    required=True, help='path to single song (separations)')
args = parser.parse_args()

# Create metric object
metric = GlobalSDR()

# read in all WAV files for the four instruments
gt = []
se = []
for instr in ['bass', 'drums', 'other', 'vocals']:
    _gt, _fs = sf.read(os.path.join(args.reference_folder, instr + '.wav'))
    _se, _fs = sf.read(os.path.join(args.separation_folder, instr + '.wav'))
    gt.append(_gt)
    se.append(_se)

gt = np.stack(gt) # shape: n_sources x n_samples x n_channels
se = np.stack(se) # shape: n_sources x n_samples x n_channels

# compute scores
scores = metric(gt, se)

print(f'Evaluated separation in {args.separation_folder}:')
print(f'\tBass: SDR_instr = {scores[0]}')
print(f'\tDrums: SDR_instr = {scores[1]}')
print(f'\tOther: SDR_instr = {scores[2]}')
print(f'\tVocals: SDR_instr = {scores[3]}')

print(f'SDR_song = {np.mean(scores)}')
