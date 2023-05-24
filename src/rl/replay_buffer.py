from typing import Any, Generator

import numpy as np
from jax import tree_util

Nested = Any


class ReplayBuffer:

    def __init__(self,
                 rng: np.random.Generator | int,
                 capacity: int,
                 signature: Nested
                 ) -> None:
        self._rng = np.random.default_rng(rng)
        self.capacity = capacity
        self.signature = signature

        def from_specs(sp):
            shape = (capacity,) + sp.shape
            return np.zeros(shape, sp.dtype)

        leaves, self._treedef = tree_util.tree_flatten(signature)
        self._memory = tree_util.tree_map(from_specs, leaves)
        self._idx = 0
        self._len = 0

    def add(self, transition: Nested) -> None:
        leaves, struct = tree_util.tree_flatten(transition)
        assert struct == self._treedef, 'Structures dont match.'
        for i in range(len(leaves)):
            self._memory[i][self._idx] = leaves[i]
        self._idx += 1
        self._len = max(self._len, self._idx)
        self._idx %= self.capacity

    def as_generator(self, batch_size: int) -> Generator[Nested, None, None]:
        while True:
            idx = self._rng.integers(0, self._len, batch_size)
            batch = tree_slice(self._memory, idx)
            yield self._treedef.unflatten(batch)

    def as_tfdataset(self, batch_size: int) -> 'tf.data.Dataset':
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')

        def to_tf_spec(sp):
            return tf.TensorSpec((batch_size,) + sp.shape, sp.dtype)

        ds = tf.data.Dataset.from_generator(
            lambda: self.as_generator(batch_size),
            output_signature=tree_util.tree_map(to_tf_spec, self.signature)
        )
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds.as_numpy_iterator()

    def __len__(self) -> int:
        return self._len

    def save(self, path: str) -> None:
        public = (self.capacity, self.signature)
        private = (self._rng, self._idx, self._len)
        np.savez_compressed(path, *self._memory, public=public, private=private)

    @classmethod
    def load(cls, path: str) -> 'ReplayBuffer':
        data = np.load(path, allow_pickle=True)
        rng, idx, len_ = data['private']
        replay = cls(rng, *data['public'])
        replay._memory = [v for k, v in data.items() if k.startswith('arr')]
        replay._idx = idx
        replay._len = len_
        return replay


def tree_slice(tree_: 'T', sl: slice, is_leaf=None) -> 'T':
    return tree_util.tree_map(lambda t: t[sl], tree_, is_leaf=is_leaf)
