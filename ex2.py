"""
Name:        Tomer Gill
I.D.:        318459450
U2 Username: gilltom
Group:       89511-06
Date:        06/04/18
"""

import numpy as np
from time import time
from matplotlib import pyplot


def generate_examples(a, num_examples=1):
    return np.random.normal(2 * a, 1, num_examples).reshape((1, num_examples))


def generate_dataset(classes, num_examples=100):
    return np.vstack([np.concatenate([generate_examples(a, num_examples), np.array([[a] * num_examples])]).T
                     for a in classes])


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


def f(x, a):
    mu = 2*a
    sigma_square=1
    return np.exp(-np.square(x-mu)/(2*sigma_square))/(sigma_square*np.sqrt(2*np.pi))


def train(W, b, dataset, epochs=1, lr=0.1):
    print "Startin training..."
    for i in range(epochs):
        start = time()
        accuracy = 0.0
        np.random.shuffle(dataset)
        for x, y in dataset:
            probs = softmax(W*x+b)  # prediction output
            accuracy += ((np.argmax(probs) + 1) == y)  # accuracy for hyper parameter optimizing

            y = int(y-1)
            # gradients
            gW = probs * x
            gW[y] -= x
            gb = probs
            gb[y] -= 1

            # update as SGD
            W -= lr * gW
            b -= lr * gb
        print "epoch #{} ended in {} seconds with {}% accuracy".format(i+1, time()-start, accuracy / dataset.shape[0]
                                                                       * 100)
    return W, b


def main(classes):
    size_of_input = 1

    # get data and weights
    examples = generate_dataset(classes)
    weights = np.zeros((len(classes), size_of_input))
    bias = np.zeros((len(classes), 1))

    weights, bias = train(weights, bias, examples, lr=0.1, epochs=30)

    # predict
    test = np.linspace(0, 10, num=100)
    probs = []
    real = []
    for x in test:
        a = np.array(classes)
        probs.append(softmax(weights * x + bias)[0])  # prediction output
        real_probs = f(x, a)
        # print x, probs[0], real_probs[0] / np.sum(real_probs)
        real.append(real_probs[0] / np.sum(real_probs))

    test, probs, real = np.array(test), np.array(probs), np.array(real)
    learned = pyplot.plot(test, probs, "r", label="Learned Probability")
    density = pyplot.plot(test, real, "b--", label="Density Probability")
    pyplot.legend(("Learned Probability", "Density Probability"))
    pyplot.axis([0, 10, 0, 1])
    pyplot.xlabel("Point")
    pyplot.ylabel("Probability")
    pyplot.title("Learned P(Y=1|X) VS. Probability by Density")
    pyplot.xticks(np.arange(0, 11))
    pyplot.yticks(np.arange(0, 1.1, 0.1))
    pyplot.show()

if __name__ == '__main__':
    classes = [1, 2, 3]
    main(classes)
