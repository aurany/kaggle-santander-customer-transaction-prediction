
import didactic_meme as model_suite
from didactic_meme import adict, make_config_class


Config = make_config_class(
    boosting_type='gbdt',
    objective='binary',
    metric='auc',
    tree_learner='serial',
    is_training_metric=True,
    boost_from_average=False,
    verbose=0,
    device_type='gpu',

    max_depth=-1,
    max_bin=255,            # (255) 200 255
    num_leaves=31,          # (31) 30 50
    min_data_in_leaf=60,    # (20) 10 30 60 120 240
    learning_rate=0.1,      # (0.1) 0.01 0.05 0.1 0.2 0.4
    feature_fraction=0.9,   # (1.0) 0.8 0.9 1.0
    bagging_fraction=0.8,   # (1.0) 0.6 0.8 1.0
    bagging_freq=10,        # (0.0)
    lambda_l2=0,            # (0.0)
)
