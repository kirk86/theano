from loader import Loader
from trainer import Trainer
import theano

theano.config.exception_verbosity = 'high'


load = Loader('../media/datasets/')
trX, trY, teX, teY = load.mnist()

# load = Loader('../media/datasets/')
# trX, trY, teX, teY, _ = load.stl10()

# load = Loader('../media/datasets/cifar10/')
# trX, trY, teX, teY = load.cifar10()

dataset = (trX, trY, teX, teY)
trainer = Trainer(dataset)
# # trainer.train_logistic_regression('randn', trX.shape[1], 784,)
# # trainer.train_multilayer_perceptron('randn', trX.shape[1], 500, 10)
# # trainer.train_autoencoder('randn', trX.shape[1], 784)
# trainer.train_denoisy_autoencoder('randn', trX.shape[1], 500)
trainer.train_stacked_denoisy_autoencoder(distribution='randn')



# n_classes = 10
# X = tt.fmatrix(name='X')
# y = tt.fmatrix(name='y')

# classifier = LogisticRegression(X, trX.shape[1], n_classes)
# # y_pred = classifier.y_pred
# cost = classifier.neg_log_like(y)
# errors = classifier.errors(y)

# grads = [tt.grad(cost=cost, wrt=param) for param in classifier.params]

# updates = [(param, param - 0.01 * gparam)
#            for param, gparam in zip(classifier.params, grads)]

# train = theano.function(inputs=[X, y], outputs=cost, updates=updates,
#                         allow_input_downcast=True)

# validate = theano.function(inputs=[X, y], outputs=errors,
#                            allow_input_downcast=True)

# predict = theano.function(inputs=[X], outputs=classifier.y_pred,
#                           allow_input_downcast=True)

# for i in range(100):
#     for start, end in zip(range(0, len(trX), 64), range(64, len(trY), 64)):
#         loss = train(trX[start:end], trY[start:end])
#     valid_errors = validate(validX, validY)
#     accuracy = np.mean(np.argmax(teY, axis=1) == predict(teX))
#     print("iter. = {}, cost = {}, errors = {}, accuracy = {}"
#           .format(i, loss, valid_errors, accuracy))
