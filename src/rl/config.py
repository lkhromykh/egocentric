import dataclasses

from rltools.config import Config as _Config

Layers = tuple[int, ...]


@dataclasses.dataclass
class Config(_Config):
    gamma: float = .98
    lambda_: float = 1.
    entropy_coef: float = 3e-4
    num_actions: int = 20

    # Architecture
    activation: str = 'relu'
    normalization: str = 'none'
    mlp_layers: Layers = ()
    cnn_depths: Layers = (32, 32, 32, 32)
    cnn_kernels: Layers = (3, 3, 3, 3)
    cnn_strides: Layers = (2, 2, 2, 2)
    actor_keys: str = r'realsense'
    actor_layers: Layers = (256, 256, 256)
    critic_keys: str = r'robotiq|box'
    critic_layers: Layers = (512, 256, 256)
    ensemble_size: int = 1

    # Train
    jit: bool = True
    buffer_capacity: int = 10 ** 4
    batch_size: int = 16
    sequence_len: int = 16
    spi: int = 1
    learning_rate: float = 3e-4
    polyak_tau: float = 5e-3
    weight_decay: float = 1e-6
    max_grad: float = 50.
    eval_every: int = 10_000
    train_after: int = 5000

    logdir: str = 'logdir/src_assym'
    task: str = 'src'
    action_space: str = 'discrete'
    num_envs: int = 12
    seed: int = 0


_DEBUG_CONFIG = Config(
    gamma=.1,
    num_actions=13,
    cnn_depths=(31, 33),
    cnn_kernels=(2, 3),
    cnn_strides=(3, 3),
    actor_layers=(5, 7),
    critic_layers=(9, 11),
    buffer_capacity=100,
    batch_size=19,
    sequence_len=15,
    eval_every=1,
    train_after=1,
    spi=1,
    num_envs=1,
    jit=False
)
