hparams = [
    {
        "model_name": "Modelo #1",
        "arch": {"hidden_units": [], "activation": "relu"},
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_name": "Modelo #2",
        "arch": {"hidden_units": [], "activation": "tanh"},
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_name": "Modelo #3",
        "arch": {"hidden_units": [], "activation": "sigmoid"},
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_name": "Modelo #4",
        "arch": {"hidden_units": [8], "activation": "relu"},
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_name": "Modelo #5",
        "arch": {"hidden_units": [16], "activation": "relu"},
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_name": "Modelo #6",
        "arch": {"hidden_units": [32], "activation": "relu"},
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_name": "Modelo #7",
        "arch": {"hidden_units": [128], "activation": "relu"},
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_name": "Modelo #8",
        "arch": {"hidden_units": [8, 8], "activation": "relu"},
        "opt": {"lr": 0.0001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_name": "Modelo #9",
        "arch": {"hidden_units": [16, 16], "activation": "relu"},
        "opt": {"lr": 0.0001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_name": "Modelo #10",
        "arch": {"hidden_units": [32, 32], "activation": "relu"},
        "opt": {"lr": 0.0001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_name": "Modelo #11",
        "arch": {"hidden_units": [128, 128], "activation": "relu"},
        "opt": {"lr": 0.0005, "epochs": 500, "batch_size": 32},
    },
    {
        "model_name": "Modelo #12",
        "arch": {"hidden_units": [32, 32, 32], "activation": "relu"},
        "opt": {"lr": 0.0005, "epochs": 500, "batch_size": 32},
    },
    {
        "model_name": "Modelo #13",
        "arch": {"hidden_units": [128, 128, 128], "activation": "relu"},
        "opt": {"lr": 0.0001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_name": "ANN-STAB",
        "arch": {"hidden_units": [64, 128, 128, 128, 64], "activation": "relu"},
        "opt": {"lr": 0.0001, "epochs": 80, "batch_size": 64},
    },
    {
        "model_name": "PTFlash Classifier",
        "arch": {"hidden_units": [32, 32, 32], "activation": "silu"},
        "opt": {"lr": 0.0005, "epochs": 500, "batch_size": 32},
    },
]

hparams = [{"model_id": i + 1, **hp} for i, hp in enumerate(hparams)]
