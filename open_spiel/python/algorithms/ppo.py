import logging

import numpy as np
import scipy as sp
import scipy.spare.linalg as spLA
import copy
import time as timer
import tensorflow.compat.v1 as tf

logging.disable(logging.CRITICAL)

class PPO(BatchREINFORCE):

    def __init__(self,
                env,
                policy_fn,
                kl_targ = 0.01,
                beta_init=1,
                epochs=2,
                mb_size=64,
                learn_rate=3e-4,
                vf_learn_rate=3e-4,
                vf_iters=10,
                batch_size=64,
                seed=None,
                schedule='Linear',
                save_logs=False)

        self.__name__ = 'PPO'
        self.env = env
