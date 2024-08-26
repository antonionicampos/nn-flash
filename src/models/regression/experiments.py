hparams = [
    {
        "model_type": "neural_network",
        "params": {
            "hidden_layers": [0, 1, 2, 3],
            "hidden_units": [8, 16, 32, 64, 128],
            "activation": ["relu"],
        },
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_name": "Phase-Split [Ref 22]",
        "model_type": "neural_network",
        "params": {"hidden_units": [64, 128, 128, 128, 64], "activation": "relu"},
        "opt": {"lr": 0.0001, "epochs": 80, "batch_size": 64},
    },
    {
        "model_name": "PTFlash Initializer [Ref 19]",
        "model_type": "neural_network",
        "params": {"hidden_units": [64, 64, 64], "type": "residual"},
        "opt": {"lr": 0.0005, "epochs": 500, "batch_size": 32},
    },
]
