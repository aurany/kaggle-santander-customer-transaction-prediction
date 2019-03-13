import didactic_meme as model_suite
from didactic_meme import adict, make_config_class

Config = make_config_class(
    dataset='data/train.csv',
    n_epochs=5,
    train_batch_size=2000,
    val_batch_size=2000,
    learning_rate=0.01,
    weight_decay=0,
    log_interval=10,
)
