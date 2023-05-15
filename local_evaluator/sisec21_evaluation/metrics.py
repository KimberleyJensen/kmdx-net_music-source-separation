"""
metrics.py

Contains metrics to evaluate source separation results.
Stefan Uhlich, Sony RDC
"""
import numpy as np
from museval import evaluate
from mir_eval.separation import bss_eval_images, bss_eval_images_framewise


class BaseMetric(object):
    """ Base class that implements a metric for audio source separation.

    ```
    metric = BaseMetric(references, separations)  # shape: [n_sources]
    ```

    where `references` and `separations` are numpy arrays of shape
    [n_sources x n_samples x n_channels].
    """
    def __call__(self, references, separations):
        # make sure that we have `np.float32` data type
        references = references.astype(np.float32)
        separations = separations.astype(np.float32)

        # make sure that `separations` and `references` have the same shape
        assert references.shape == separations.shape,\
               (f"Shape mismatch between references ({references.shape}) and "
                f"estimates ({separations.shape}).")

        # return metric
        return self._metric(references, separations)

    def _metric(self, references, separations):
        """ implements our metric that we use to compare `references` with
        `separations` """
        raise NotImplementedError


class MedianSDR_BSSEval4(BaseMetric):
    """ SDR from BSS Eval v4: https://github.com/sigsep/sigsep-mus-eval
    This is the main metric for SiSEC 2018 """
    def _metric(self, references, separations):
        sdr, _isr, _sir, _sar = evaluate(references, separations)
        return np.nanmedian(sdr, axis=1)


class MeanSDR_BSSEval4(BaseMetric):
    """ SDR from BSS Eval v4: https://github.com/sigsep/sigsep-mus-eval
    This is the main metric for SiSEC 2018 """
    def _metric(self, references, separations):
        sdr, _isr, _sir, _sar = evaluate(references, separations)
        return np.nanmean(sdr, axis=1)


class MedianFramewiseSDR_BSSEval3(BaseMetric):
    """ SDR from BSS Eval v3 implemented by mir-eval:
            https://github.com/craffel/mir_eval/blob/master/mir_eval/separation.py
        This is the main metric for SiSEC 2016 """
    def _metric(self, references, separations):
        sdr, _isr, _sir, _sar, _perm = bss_eval_images_framewise(references, separations,
                                                                 compute_permutation=False)
        # remove all -inf/inf values
        sdr = [_ for _ in sdr if not np.any(np.isinf(sdr))]
        return np.nanmedian(sdr, axis=1)


class MeanFramewiseSDR_BSSEval3(BaseMetric):
    """ SDR from BSS Eval v3 implemented by mir-eval:
            https://github.com/craffel/mir_eval/blob/master/mir_eval/separation.py
        This is the main metric for SiSEC 2016 """
    def _metric(self, references, separations):
        sdr, _isr, _sir, _sar, _perm = bss_eval_images_framewise(references, separations,
                                                                 compute_permutation=False)
        # remove all -inf/inf values
        sdr = [_ for _ in sdr if not np.any(np.isinf(sdr))]
        return np.nanmean(sdr, axis=1)


class SDR_BSSEval3(BaseMetric):
    """ SDR from BSS Eval v3 implemented by mir-eval:
            https://github.com/craffel/mir_eval/blob/master/mir_eval/separation.py
        This is the original BSSEvalv3 (Matlab) """
    def _metric(self, references, separations):
        sdr, _isr, _sir, _sar, _popt = bss_eval_images(references, separations,
                                                       compute_permutation=False)
        return sdr


class GlobalSISDR(BaseMetric):
    """ SI-SDR - see, e.g., https://arxiv.org/pdf/1811.02508.pdf """
    def _metric(self, references, separations):
        delta = 1e-7  # avoid numerical errors

        alpha = np.sum(separations * references, axis=(1, 2)) / \
            (delta + np.sum(references * references, axis=(1, 2)))
        alpha = alpha[:, np.newaxis, np.newaxis]

        num = np.sum(np.square(alpha * references), axis=(1, 2))
        den = np.sum(np.square(alpha * references - separations), axis=(1, 2))
        num += delta
        den += delta
        return 10 * np.log10(num / den)


class MedianFramewiseSISDR(GlobalSISDR):
    """ Framewise SI-SDR + Median averaging """
    def __init__(self, win, hop):
        super().__init__()
        self.win = win
        self.hop = hop

    def _metric(self, references, separations):
        vals = np.zeros((references.shape[0],
                         1 + (references.shape[1] - self.win) // self.hop))
        for i, idx in enumerate(range(0, references.shape[1] - self.win, self.hop)):
            vals[:, i] = super()._metric(references[:, idx:idx+self.win, :],
                                         separations[:, idx:idx+self.win, :])
        return np.median(vals, axis=1)


class MeanFramewiseSISDR(GlobalSISDR):
    """ Framewise SI-SDR + Mean averaging """
    def __init__(self, win, hop):
        super().__init__()
        self.win = win
        self.hop = hop

    def _metric(self, references, separations):
        vals = np.zeros((references.shape[0],
                         1 + (references.shape[1] - self.win) // self.hop))
        for i, idx in enumerate(range(0, references.shape[1] - self.win, self.hop)):
            vals[:, i] = super()._metric(references[:, idx:idx+self.win, :],
                                         separations[:, idx:idx+self.win, :])
        return np.mean(vals, axis=1)


class GlobalMSE(BaseMetric):
    """ Global MSE """
    def _metric(self, references, separations):
        return np.mean(np.square(references - separations), axis=(1, 2))


class MedianFramewiseMSE(GlobalMSE):
    """ Framewise MSE + Median averaging """
    def __init__(self, win, hop):
        super().__init__()
        self.win = win
        self.hop = hop

    def _metric(self, references, separations):
        vals = np.zeros((references.shape[0],
                         1 + (references.shape[1] - self.win) // self.hop))
        for i, idx in enumerate(range(0, references.shape[1] - self.win, self.hop)):
            vals[:, i] = super()._metric(references[:, idx:idx+self.win, :],
                                         separations[:, idx:idx+self.win, :])
        return np.median(vals, axis=1)


class MeanFramewiseMSE(GlobalMSE):
    """ Framewise MSE + Mean averaging """
    def __init__(self, win, hop):
        super().__init__()
        self.win = win
        self.hop = hop

    def _metric(self, references, separations):
        vals = np.zeros((references.shape[0],
                         1 + (references.shape[1] - self.win) // self.hop))
        for i, idx in enumerate(range(0, references.shape[1] - self.win, self.hop)):
            vals[:, i] = super()._metric(references[:, idx:idx+self.win, :],
                                         separations[:, idx:idx+self.win, :])
        return np.mean(vals, axis=1)


class GlobalMAE(BaseMetric):
    """ Global MSE """
    def _metric(self, references, separations):
        return np.mean(np.abs(references - separations), axis=(1, 2))


class MedianFramewiseMAE(GlobalMAE):
    """ Framewise MSE + Median averaging """
    def __init__(self, win, hop):
        super().__init__()
        self.win = win
        self.hop = hop

    def _metric(self, references, separations):
        vals = np.zeros((references.shape[0],
                         1 + (references.shape[1] - self.win) // self.hop))
        for i, idx in enumerate(range(0, references.shape[1] - self.win, self.hop)):
            vals[:, i] = super()._metric(references[:, idx:idx+self.win, :],
                                         separations[:, idx:idx+self.win, :])
        return np.median(vals, axis=1)


class MeanFramewiseMAE(GlobalMAE):
    """ Framewise MSE + Mean averaging """
    def __init__(self, win, hop):
        super().__init__()
        self.win = win
        self.hop = hop

    def _metric(self, references, separations):
        vals = np.zeros((references.shape[0],
                         1 + (references.shape[1] - self.win) // self.hop))
        for i, idx in enumerate(range(0, references.shape[1] - self.win, self.hop)):
            vals[:, i] = super()._metric(references[:, idx:idx+self.win, :],
                                         separations[:, idx:idx+self.win, :])
        return np.mean(vals, axis=1)


class GlobalSDR(BaseMetric):
    """ Global SDR """
    def _metric(self, references, separations):
        delta = 1e-7  # avoid numerical errors
        num = np.sum(np.square(references), axis=(1, 2))
        den = np.sum(np.square(references - separations), axis=(1, 2))
        num += delta
        den += delta
        return 10 * np.log10(num / den)


class MedianFramewiseSDR(GlobalSDR):
    """ Framewise SDR + Median averaging """
    def __init__(self, win, hop):
        super().__init__()
        self.win = win
        self.hop = hop

    def _metric(self, references, separations):
        vals = np.zeros((references.shape[0],
                         1 + (references.shape[1] - self.win) // self.hop))
        for i, idx in enumerate(range(0, references.shape[1] - self.win, self.hop)):
            vals[:, i] = super()._metric(references[:, idx:idx+self.win, :],
                                         separations[:, idx:idx+self.win, :])
        return np.median(vals, axis=1)


class MeanFramewiseSDR(GlobalSDR):
    """ Framewise SDR + Mean averaging """
    def __init__(self, win, hop):
        super().__init__()
        self.win = win
        self.hop = hop

    def _metric(self, references, separations):
        vals = np.zeros((references.shape[0],
                         1 + (references.shape[1] - self.win) // self.hop))
        for i, idx in enumerate(range(0, references.shape[1] - self.win, self.hop)):
            vals[:, i] = super()._metric(references[:, idx:idx+self.win, :],
                                         separations[:, idx:idx+self.win, :])
        return np.mean(vals, axis=1)
