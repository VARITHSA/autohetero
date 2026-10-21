"""A small image app, written the way its authors would without AutoHetero: four steps on CPU tensors.

Its op library is autohetero.ops. The marked line is the only AutoHetero-specific line: delete it and the app
still works, entirely on the CPU (ops run where their input is); keep it and every step runs where it is fastest.
"""
import autohetero

autohetero.enable()  # <- the one added line


def blur(images):
    return autohetero.ops.blur(images)


def edges(images):
    return autohetero.ops.sobel(images)


def shrink(images, side=224):
    return autohetero.ops.resize(images, scale=side / min(images.shape[-2:]))  # short side -> 224


def classify(images):
    return autohetero.ops.resnet18(images).argmax(1)  # random weights: the classes only exercise the model


STEPS = {"Blur": blur, "Edges": edges, "Shrink": shrink, "Classify": classify}
