import dataclasses

from rltools.config import Config as _Config

Layers = tuple[int, ...]


@dataclasses.dataclass
class Config(_Config):
    gamma: float = .98
    lambda_: float = 0.
    entropy_coef: float = 1e-3
    utd: int = 1
    num_actions: int = 20

    # Architecture
    activation: str = 'relu'
    normalization: str = 'none'

    mlp_layers: Layers = ()
    cnn_depths: Layers = (32, 32, 32, 32)
    cnn_kernels: Layers = (3, 3, 3, 3)
    cnn_strides: Layers = (2, 1, 1, 1)

    actor_keys: str = r'.*'
    actor_layers: Layers = (256, 256)

    critic_keys: str = r'.*'
    critic_layers: Layers = (256, 256)
    ensemble_size: int = 2

    # Train
    jit: bool = True
    buffer_capacity: int = 10 ** 5
    learning_rate: float = 1e-3
    polyak_tau: float = 5e-3
    batch_size: int = 16
    sequence_len: int = 16
    max_grad: float = 50.
    weight_decay: float = 1e-6

    logdir: str = 'logdir/'
    task: str = 'ur_pick'
    action_space: str = 'discrete'
    seed: int = 0


_DEBUG_CONFIG = Config(
    gamma=.9,
    num_actions=13,
    cnn_depths=(31, 33),
    cnn_kernels=(2, 3),
    cnn_strides=(3, 3),
    actor_layers=(5, 7),
    critic_layers=(9, 11),
    buffer_capacity=100,
    batch_size=19,
    jit=False
)
