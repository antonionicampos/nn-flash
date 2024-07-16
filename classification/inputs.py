import tensorflow as tf

hparams = [
    {
        "id": 1,
        "model_name": "Modelo #1",
        "arch": {
            "hidden_layers": [],
            "activation": tf.keras.activations.relu,
        },
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "id": 2,
        "model_name": "Modelo #2",
        "arch": {
            "hidden_layers": [32],
            "activation": tf.keras.activations.relu,
        },
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "id": 3,
        "model_name": "Modelo #3",
        "arch": {
            "hidden_layers": [128],
            "activation": tf.keras.activations.relu,
        },
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "id": 4,
        "model_name": "Modelo #4",
        "arch": {
            "hidden_layers": [32, 32],
            "activation": tf.keras.activations.relu,
        },
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "id": 5,
        "model_name": "Modelo #5",
        "arch": {
            "hidden_layers": [128, 128],
            "activation": tf.keras.activations.relu,
        },
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "id": 6,
        "model_name": "Modelo #6",
        "arch": {
            "hidden_layers": [32, 32, 32],
            "activation": tf.keras.activations.swish,
        },
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "id": 7,
        "model_name": "Modelo #7",
        "arch": {
            "hidden_layers": [32, 32, 32],
            "activation": tf.keras.activations.relu,
        },
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
    {
        "id": 8,
        "model_name": "Modelo #8",
        "arch": {
            "hidden_layers": [128, 128, 128],
            "activation": tf.keras.activations.relu,
        },
        "opt": {"lr": 0.001, "epochs": 500, "batch_size": 32},
    },
]
