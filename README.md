## theano models

### load and prepare the dataset

```
from loader import Loader
from trainer import Trainer

<!-- load dataset -->
load = Loader('/path/to/dataset/mnist')
trX, trY, teX, teY = load.mnist()
dataset = (trX, trY, teX, teY)
trainer = Trainer(dataset)
```

### training logistic regression


```
trainer.train_logistic_regression(distribution='randn', fan_in=28*28, fan_out=10)
```

### train multilayer perceptron

```
trainer.train_multilayer_perceptron(distribution='randn', fan_in=28*28, n_hidden=500, fan_out=10)
```

### train autoencoder

```
trainer.train_autoencoder(distribution='randn', fan_in=28*28, fan_out=28*28)
```

### train denoisy autoencoder

```
trainer.train_denoisy_autoencoder(distribution='randn', n_visible=28*28, n_hidden=500)
```

### train stacked denoisy autoencoder

``` trainer.train_stacked_denoisy_autoencoder(distribution='randn',
fan_in=784, n_hidden_sizes=[500, 1024], fan_out=10, noise_levels=[0.1, 0.2])
```
