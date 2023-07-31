View these logs by running the following command while in the parent directory. (one level up)

# To see logs for building IMDB and MNIST models use:
tensorboard --logdir archive/logs/

# To see MNIST alone use: 
tensorboard --logdir archive/MNIST-extra-metrics1690486849/

# To see experiments hyper parameter comparisons use:
tensorboard --logdir archive/hparam_tuning

# To see logs for individual models built during the hparam search use:
tensorboard --logdir archive/experiments

Then open a brower to : http://localhost:6006/ to view the log data.

Enjoy!
