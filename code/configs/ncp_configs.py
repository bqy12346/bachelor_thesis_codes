conf_ltc_ncp = {
    'modelname':  'ltc_ncp',
    'modeltype':  'LTC_NCP',
    'parameters': {
        'motor_neurons': 32,
        'mixed_memory':  False,
        'epochs':        50,
        'batch_size':    128,
        'lr':            0.002,
        'ode_unfolds': 1,
    }
}

conf_cfc_ncp = {
    'modelname':  'cfc_ncp',
    'modeltype':  'CFC_NCP',
    'parameters': {
        'motor_neurons': 32,
        'mixed_memory':  False,
        'epochs':        50,
        'batch_size':    128,
        'lr':            0.002,
    }
}
