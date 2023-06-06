import random

from sklearn.metrics import classification_report


def ResultCompare(output, target):
    output = output[:len(target)]
    n = len(target)
    for i in range(n):
        if i % 3 == 0:
            output[i] = int(random.random() * 3)
    score = classification_report(target, output)
    print(score)
