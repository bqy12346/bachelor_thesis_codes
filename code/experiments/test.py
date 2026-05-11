import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.scp_experiment import SCP_Experiment
from configs.ncp_configs import conf_cfc_ncp

datafolder   = '../../data/ptbxl/'
outputfolder = '../../output/'

e = SCP_Experiment(
    'exp_ncp_superdiagnostic',
    'superdiagnostic',
    datafolder,
    outputfolder,
    [conf_cfc_ncp],
)
e.prepare()
e.perform()     # 只跑 CfC，生成 y_train/val/test_pred.npy