import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.scp_experiment import SCP_Experiment
from configs.fastai_configs import conf_fastai_xresnet1d101
from configs.ncp_configs import conf_ltc_ncp, conf_cfc_ncp

datafolder   = '../../data/ptbxl/'
outputfolder = '../../output/'

models = [
    conf_fastai_xresnet1d101,
    conf_ltc_ncp,
    conf_cfc_ncp,
]

e = SCP_Experiment(
    'exp_ncp_superdiagnostic',
    'superdiagnostic',
    datafolder,
    outputfolder,
    models,
)
e.prepare()
e.perform()
e.evaluate()