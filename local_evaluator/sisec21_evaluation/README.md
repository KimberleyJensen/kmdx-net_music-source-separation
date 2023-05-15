# SiSEC 2021 Evaluation

This repository contains code to see which measure we should use for the
contest.

```bash
$ python eval.py --references-folder REF_DIR        \
                 --separations-folder SEP_DIR       \
                 --output RESULT.mat
```

From the different metrics, we choose `GlobalSDR` as the metric for SiSEC2021.
Here is an example how to compute it:

```
$ python eval_sisec21.py --reference-folder sample/groundtruth/AM\ Contra\ -\ Heart\ Peripheral \
                         --separation-folder sample/separation/AM\ Contra\ -\ Heart\ Peripheral
Evaluated separation in sample/separation/AM Contra - Heart Peripheral/:
	Bass: SDR_instr = 2.0729923248291016
	Drums: SDR_instr = 5.259542465209961
	Other: SDR_instr = 0.4605589509010315
	Vocals: SDR_instr = 10.389951705932617
SDR_song = 4.5457611083984375
```

