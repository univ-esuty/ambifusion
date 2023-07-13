# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import copy
import scipy.linalg
from . import metric_utils

#----------------------------------------------------------------------------

def compute_fid(opts, max_items):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.
    opts_1 = copy.deepcopy(opts)
    opts_2 = copy.deepcopy(opts)
    
    opts_1.dataset_kwargs = copy.deepcopy(opts.data_1_kwargs)
    mu_1, sigma_1 = metric_utils.compute_feature_stats_for_dataset(
        opts=opts_1, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_items, batch_size=opts.batch_size).get_mean_cov()
    
    opts_2.dataset_kwargs = copy.deepcopy(opts.data_2_kwargs)
    mu_2, sigma_2 = metric_utils.compute_feature_stats_for_dataset(
        opts=opts_2, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_items, batch_size=opts.batch_size).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_2 - mu_1).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_2, sigma_1), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_2 + sigma_1 - s * 2))
    return float(fid)

#----------------------------------------------------------------------------
