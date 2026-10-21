"""Inference workloads: ResNet-18 and MobileNetV2.

Each takes a float32 batch (N, 3, H, W), typically 224x224, and returns ImageNet logits (N, 1000) on the
batch's device. Standard torchvision definitions with random weights from a fixed seed (nothing downloaded).
Models are in eval mode, so every sample is independent and both ops split along the batch dim.
"""
import threading

import torch

from .. import devices


def resnet18(images):
    return _classify("resnet18", images)


def mobilenetv2(images):
    return _classify("mobilenet_v2", images)


def _classify(name, images):
    with torch.no_grad():  # not inference_mode: its outputs can't be edited in place by the caller afterwards
        return devices.model_on(_model(name), images.device)(images)


_models = {}
_lock = threading.Lock()  # Hybrid's two threads can ask for a model at once: build it once, from one seed


def _model(name):
    with _lock:
        if name not in _models:
            from torchvision import models  # imported here so `import autohetero` stays fast

            with torch.random.fork_rng(devices=[]):  # seed without touching the caller's random state
                torch.manual_seed(0)
                _models[name] = getattr(models, name)(weights=None).eval()
        return _models[name]
