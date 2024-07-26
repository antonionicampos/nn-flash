hparams = [
    {
        "model_id": 1,
        "model_name": "Modelo #1",
        "arch": {
            "hidden_units": [],
            "activation": "relu",
        },
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_id": 2,
        "model_name": "Modelo #2",
        "arch": {
            "hidden_units": [32],
            "activation": "relu",
        },
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_id": 3,
        "model_name": "Modelo #3",
        "arch": {
            "hidden_units": [128],
            "activation": "relu",
        },
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_id": 4,
        "model_name": "Modelo #4",
        "arch": {
            "hidden_units": [32, 32],
            "activation": "relu",
        },
        "opt": {"lr": 0.0001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_id": 5,
        "model_name": "Modelo #5",
        "arch": {
            "hidden_units": [128, 128],
            "activation": "relu",
        },
        "opt": {"lr": 0.0005, "epochs": 500, "batch_size": 32},
    },
    {
        "model_id": 6,
        "model_name": "Modelo #6",
        "arch": {
            "hidden_units": [32, 32, 32],
            "activation": "relu",
        },
        "opt": {"lr": 0.0005, "epochs": 500, "batch_size": 32},
    },
    {
        "model_id": 7,
        "model_name": "Modelo #7",
        "arch": {
            "hidden_units": [128, 128, 128],
            "activation": "relu",
        },
        "opt": {"lr": 0.0001, "epochs": 500, "batch_size": 32},
    },
    {
        "model_id": 8,
        "model_name": "ANN-STAB",
        "arch": {
            "hidden_units": [64, 128, 128, 128, 64],
            "activation": "relu",
        },
        "opt": {"lr": 0.0001, "epochs": 80, "batch_size": 64},
    },
    {
        "model_id": 9,
        "model_name": "PTFlash Classifier",
        "arch": {
            "hidden_units": [32, 32, 32],
            "activation": "silu",
        },
        "opt": {"lr": 0.0005, "epochs": 500, "batch_size": 32},
    },
]
