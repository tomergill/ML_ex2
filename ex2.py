"""
Name:        Tomer Gill
I.D.:        318459450
U2 Username: gilltom
Group:       89511-06
Date:        06/04/18
"""

import numpy as np
from time import time


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


def train(W, b, dataset, epochs=1, lr=0.1):
    print "Startin training..."
    for i in range(epochs):
        start = time()
        accuracy = 0.0
        for x, y in dataset:
            probs = softmax(W*x+b)  # prediction output
            accuracy += (np.argmax(probs) + 1 == y)  # accuracy for hyper parameter optimizing

            y = int(y-1)
            # gradients
            gW = probs * x
            gW[y] -= x
            gb = probs
            gb[y] -= 1

            # update as SGD
            W += lr * gW
            b += lr * gb
        print "epoch #{} ended in {} seconds with {}% accuracy".format(i+1, time()-start, accuracy / dataset.shape[0]
                                                                       * 100)
    return W, b





def main(classes):
    size_of_input = 1
    learning_rate = 0.01

    # get data and weights
    examples = generate_dataset(classes)
    weights = np.zeros((len(classes), size_of_input))
    bias = np.zeros((len(classes), 1))

    weights, bias = train(weights, bias, examples, lr=0.01)

    print "###############################################################"
    print "###############################################################"

    # predict
    test = range(11)
    for x in test:
        probs = softmax(weights * x + bias)  # prediction output
        print x, probs[0]


if __name__ == '__main__':
    classes = [1, 2, 3]
    main(classes)
