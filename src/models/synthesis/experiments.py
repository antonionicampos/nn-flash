hparams = [
    {
        "model_type": "dirichlet",
        "model_name": "Dirichlet Estimator",
    },
    {
        "model_type": "wgan",
        "params": {
            "hidden_layers": [2, 3, 4],
            "hidden_units": [64, 128, 256],
            "activation": ["relu"],
            "latent_dim": [8, 16],
        },
        "opt": {
            "lr": 0.0001,
            "beta_1": 0.5,
            "beta_2": 0.9,
            "lambda": 10.0,
            "n_critic": 5,
            "epochs": int(1e4),
            "batch_size": 32,
        },
    },
]
