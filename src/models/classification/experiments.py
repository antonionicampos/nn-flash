hparams = [
    {
        "model_type": "svm",
        "params": {
            "kernel": ["linear", "sigmoid", "rbf"],
            "degree": [2, 3, 4, 5, 6, 7],
            "C": [1.0, 10.0],
        },
    },
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
        "model_name": "Stability test [Ref 22]",
        "model_type": "neural_network",
        "params": {"hidden_units": [64, 128, 128, 128, 64], "activation": "relu"},
        "opt": {"lr": 0.0001, "epochs": 80, "batch_size": 64},
    },
    {
        "model_name": "PTFlash Classifier [Ref 19]",
        "model_type": "neural_network",
        "params": {"hidden_units": [32, 32, 32], "activation": "silu"},
        "opt": {"lr": 0.0005, "epochs": 500, "batch_size": 32},
    },
]
