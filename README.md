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
trainer.train_logistic_regression('randn', 28*28, 10)
```

### train multilayer perceptron

```
trainer.train_multilayer_perceptron('randn', 28*28, 500, 10)
```

### train autoencoder

```
trainer.train_autoencoder('randn', 28*28, 28*28)
```

### train denoisy autoencoder

```
trainer.train_denoisy_autoencoder('randn', n_visible=28*28, n_hidden=500)
```
