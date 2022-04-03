import math
import os
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from torch import nn, Tensor


def entropy(p):
    return torch.sum(-p * torch.log(p), dim=1).mean()


# two moons example.

class _FeatureCollector:

    def __init__(self, max_limit=5) -> None:
        self.__n = 0
        self.feature: OrderedDict = OrderedDict()
        self._enable = False
        self.__max = max_limit

    def __call__(self, _, input_, result):
        if self._enable:
            self.feature[self.__n] = result
            self.__n += 1
            if self.__n >= self.__max:
                raise RuntimeError(f"You may forget to call clear as this hook "
                                   f"has registered data from {self.__n} forward passes.")
        return

    def clear(self):
        self.__n = 0
        del self.feature
        self.feature: OrderedDict = OrderedDict()

    def set_enable(self, enable=True):
        self._enable = enable

    @property
    def enable(self):
        return self._enable


class SingleFeatureExtractor:

    def __init__(self, model: nn.Module, feature_name: str) -> None:
        self._model = model
        self._feature_name = feature_name
        self._feature_extractor: _FeatureCollector = None  # type: ignore # noqa
        self._hook_handler = None
        self.__bind_done__ = False

    def bind(self):
        model = self._model
        extractor = _FeatureCollector()
        handler = getattr(model, self._feature_name).register_forward_hook(extractor)
        self._feature_extractor = extractor
        self._hook_handler = handler
        self.__bind_done__ = True

    def remove(self):
        self._hook_handler.remove()
        self.__bind_done__ = False

    def __enter__(self, ):
        self.bind()
        return self

    def __exit__(self, *args, **kwargs):
        self.remove()

    def clear(self):
        self._feature_extractor.clear()

    def feature(self):
        collected_feature_dict = self._feature_extractor.feature
        if len(collected_feature_dict) > 0:
            return torch.cat(list(collected_feature_dict.values()), dim=0)
        raise RuntimeError("no feature has been recorded.")

    def set_enable(self, enable=True):
        self._feature_extractor.set_enable(enable=enable)

    @contextmanager
    def enable_register(self, enable=True):
        prev_state = self._feature_extractor.enable
        self.set_enable(enable)
        yield
        self.set_enable(prev_state)


def random_iterator(X, y, batch_size, infinite=True):
    n_samples = X.shape[0]
    accumulator = 0
    while True:
        idx = np.random.choice(n_samples, batch_size, replace=False)
        yield X[idx], y[idx]
        accumulator += batch_size
        if accumulator >= n_samples:
            if not infinite:
                break


def create_two_moons_example(n_samples, noise=0.1):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=1)
    return X, y


def rotate(origin, point, degree):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point[:, 0], point[:, 1]
    angle = degree / 360 * 2 * math.pi
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.stack([qx, qy], axis=1)


def create_rotated_two_moons_example(n_samples, noise=0.1, degree=0.0, random_state=1):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X = rotate(origin=(0.5, 0.25), point=X, degree=degree)
    return X, y


def create_dataset(n_train=300, n_test=1000, noise=0.1, degree=0):
    X_source, y_source = create_rotated_two_moons_example(n_samples=n_train, noise=noise, degree=0.0, random_state=1)
    X_target, y_target = create_rotated_two_moons_example(n_samples=n_train, noise=noise, degree=degree, random_state=2)
    X_test, y_test = create_rotated_two_moons_example(n_samples=n_test, noise=noise, degree=degree, random_state=3)
    return (X_source, y_source), (X_target, y_target), (X_test, y_test),


class SimpleNet(nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(SimpleNet, self).__init__()
        self.hidden1 = nn.Linear(n_features, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x


@torch.no_grad()
def plot_decision_boundary(model, X, y, target_X=None):
    X = X.cpu()
    y = y.cpu()
    if target_X is not None:
        target_X = target_X.cpu()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(float(x_min), float(x_max), 0.1),
                         np.arange(float(y_min), float(y_max), 0.1))
    Z = model(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().to(device)).softmax(1)[:,
        1].detach().cpu().numpy()
    Z = Z.reshape(xx.shape)
    plt.clf()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)  # noqa
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)  # noqa
    if target_X is not None:
        plt.scatter(target_X[:, 0], target_X[:, 1], c='gray')
    return plt.gcf()


def initialize_training():
    n_features = 2
    n_hidden = 20
    n_output = 2
    n_epochs = 2500
    learning_rate = 0.0005
    batch_size = 300
    criterion = nn.CrossEntropyLoss()
    model = SimpleNet(n_features, n_hidden, n_output).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer, criterion, n_epochs, batch_size


@torch.no_grad()
def accuracy(model, X, y):
    y_pred = model(X)
    _, y_pred = torch.max(y_pred, 1)
    return (y_pred == y).float().mean()


def regularization(source_feature: Tensor, target_feature: Tensor, sigma: float = 0.8) -> Tensor:
    def _regularize(source_feature, target_feature):
        source_size = source_feature.size()[0]
        target_size = target_feature.size()[0]
        all_features = torch.cat([source_feature, target_feature], dim=0)
        distance_map = torch.exp(-torch.cdist(all_features, all_features, p=2) / (2 * sigma ** 2))
        distance_map = distance_map * (1 - torch.eye(all_features.size()[0]))
        intra_domain_distance_map = distance_map[:source_size, :source_size]
        intra_nominator = torch.max(intra_domain_distance_map, dim=1)[0]
        # intra_denominator = torch.sum(intra_domain_distance_map, dim=1)
        # intra_domain_distance_map = intra_nominator / intra_denominator

        inter_domain_distance_map = distance_map[:source_size, source_size:]
        inter_nominator = torch.max(inter_domain_distance_map, dim=1)[0]
        # inter_denominator = torch.sum(inter_domain_distance_map, dim=1)
        # inter_domain_distance_map = inter_nominator / inter_denominator
        return torch.stack([intra_nominator, inter_nominator], dim=1).softmax(1)

    p1 = _regularize(source_feature, target_feature)
    p2 = _regularize(target_feature, source_feature)
    return -entropy(p1) - entropy(p2)


def train(*, degree, reg_weight: float = 0.0, save_dir: str, args):
    model, optimizer, criterion, n_epochs, batch_size = initialize_training()
    feature_extractor = SingleFeatureExtractor(model, feature_name="hidden2")

    (X_source, y_source), (X_target, _), (X_test, y_test) = \
        create_dataset(n_train=300, n_test=500, noise=0.05, degree=degree)
    X_source = torch.from_numpy(X_source).float().to(device)
    y_source = torch.from_numpy(y_source).long().to(device)

    X_target = torch.from_numpy(X_target).float().to(device)

    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    X_target_iter = random_iterator(X_target, torch.ones_like(X_target), batch_size=batch_size, infinite=True)
    with feature_extractor, feature_extractor.enable_register(True):
        for epoch in range(n_epochs):
            for source_data in random_iterator(X_source, y_source, batch_size, infinite=False):
                feature_extractor.clear()
                X_source_, y_source_ = source_data
                X_target_, _ = next(X_target_iter)

                source_logit, target_logit = torch.split(model(torch.cat([X_source_, X_target_], dim=0)),
                                                         [X_source_.size()[0], X_target_.size()[0]], dim=0)

                loss = criterion(source_logit, y_source_)
                features_ = feature_extractor.feature()
                source_features_, target_features_ = torch.split(features_, [X_source_.size()[0], X_target_.size()[0]],
                                                                 dim=0)
                regularization_loss = regularization(source_features_, target_features_, sigma=args.sigma)
                optimizer.zero_grad()
                (loss + reg_weight * regularization_loss).backward()
                optimizer.step()

            if epoch % 100 == 0:
                test_accuray = accuracy(model, X_test, y_test)
                print(
                    'Epoch: {}/{}, Loss: {:.3f}, Reg: {:.3f}, Accuracy: {:.3f}'
                        .format(epoch, n_epochs, loss.item(), regularization_loss.item(), test_accuray)
                )
            if epoch % 100 == 0:
                plt.figure(figsize=(5, 5), num=0)
                figure = plot_decision_boundary(model, X_source, y_source, target_X=X_target)
                plt.text(1.5, 1.5, f'degree: {degree} \ntest_acc: {test_accuray * 100:.2f}', fontsize=10,
                         bbox=dict(facecolor='red', alpha=0.5))
                save_name = os.path.join(save_dir,
                                         f'degree_{degree:03d}/reg_weight_{reg_weight}/sigma_{args.sigma:.3f}/epoch_{epoch:05d}.png')
                Path(save_name).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_name, dpi=75)

    return os.path.join(save_dir, f'degree_{degree:03d}/reg_weight_{reg_weight}/sigma_{args.sigma:.3f}')


def gif(folder_path: str):
    import subprocess
    subprocess.call(f"convert -delay 50 -loop 0 {folder_path}/*.png {folder_path}/animated.gif", shell=True)


if __name__ == '__main__':
    import argparse

    import torch

    plt.switch_backend("agg")
    device = "cpu"

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--degree', type=int, default=0, help='degree of rotation')
    parser.add_argument('--reg-weight', type=float, default=0.0, help='regularization weight')
    parser.add_argument('--save-dir', type=str, default="./runs", help='save directory')
    parser.add_argument('--sigma', type=float, default=0.8, help='sigma')

    args = parser.parse_args()
    saved_folder = train(degree=args.degree, reg_weight=args.reg_weight, save_dir=args.save_dir, args=args)
    gif(str(saved_folder))
