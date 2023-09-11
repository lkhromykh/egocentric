import dataclasses

from rltools.config import Config as _Config

Layers = tuple[int, ...]


@dataclasses.dataclass
class Config(_Config):

    gamma: float = .98
    lambda_: float = 1.
    entropy_coef: float = 1e-3
    entropy_per_dim: float = .1
    num_actions: int = 20

    # Architecture
    activation: str = 'elu'
    normalization: str = 'layer'
    asymmetric: bool = True
    mlp_layers: Layers = (512,)
    cnn_depths: Layers = (64, 64, 64, 64)
    cnn_kernels: Layers = (3, 3, 3, 3)
    cnn_strides: Layers = (2, 2, 1, 1)
    actor_keys: str = r'image|tcp_pose|object_detected|length'
    actor_layers: Layers = (512, 512)
    critic_keys: str = r'robotiq_2f85|model'
    critic_layers: Layers = (512, 512)
    ensemble_size: int = 2

    # Train
    jit: bool = True
    buffer_capacity: int = 10 ** 5
    batch_size: int = 256
    sequence_len: int = 16
    utd: float = .05
    learning_rate: float = 3e-4
    init_temperature: float = 1e-3
    temp_learning_rate: float = 1e-2
    polyak_tau: float = 5e-3
    weight_decay: float = 1e-6
    max_grad: float = 20.
    eval_every: int = 40_000
    save_replay_every: int = 10 ** 5
    train_after: int = 10_000

    logdir: str = 'logdir/dr_image64_allitems1'
    task: str = 'src'
    action_space: str = 'discrete'
    num_envs: int = 16
    seed: int = 1


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
    utd=1,
    num_envs=1,
    jit=False
)
