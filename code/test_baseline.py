####################################
#   First Test:  Superdiagnostic tesk with original models from repository.
####################################



from experiments.scp_experiment import SCP_Experiment
from configs.fastai_configs import conf_fastai_xresnet1d101

datafolder = '../data/ptbxl/'
outputfolder = '../output/'

models = [conf_fastai_xresnet1d101]
e = SCP_Experiment('exp_superdiagnostic_test', 'superdiagnostic',
                    datafolder, outputfolder, models)
e.prepare()
e.perform()
e.evaluate()