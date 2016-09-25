## theano models

### training logistic regression

```
from loader import Loader
from trainer import Trainer

<!-- load dataset -->
load = Loader('/path/to/dataset/mnist')
trX, trY, teX, teY = load.mnist()
dataset = (trX, trY, teX, teY)
trainer = Trainer(dataset)

trainer.train_logistic_regression('randn', 28*28, 10)
```

### train multilayer perceptron

```
trainer.train_multilayer_perceptron('randn', 28*28, 500, 10)
```
