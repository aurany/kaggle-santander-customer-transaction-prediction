import didactic_meme as model_suite
from didactic_meme import adict, make_config_class


Config = make_config_class(
    n_epochs=500,
    train_batch_size=2000,
    val_batch_size=2000,
    learning_rate=1e-3,
    weight_decay=1e-3,
    momentum=1e-3, # SGD
    model='Network'
)
