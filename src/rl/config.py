import dataclasses

from rltools.config import Config as _Config

Layers = tuple[int, ...]


@dataclasses.dataclass
class Config(_Config):

    gamma: float = .97
    lambda_: float = 1.
    entropy_coef: float = 1e-3
    num_actions: int = 20

    # Architecture
    mlp_layers: Layers = 512,
    cnn_filters: Layers = (32, 64, 128, 256)
    cnn_kernels: Layers = (3, 3, 3, 3)
    cnn_strides: Layers = (2, 2, 2, 2)
    actor_keys: str = r'image|tcp_height|object_detected'
    actor_layers: Layers = 64, 512, 512
    critic_keys: str = r'robotiq_2f85|model'
    critic_layers: Layers = 512, 512, 512
    ensemble_size: int = 2

    # Train
    jit: bool = True
    buffer_capacity: int = 10 ** 5
    batch_size: int = 256
    sequence_len: int = 4
    utd: float = .1
    learning_rate: float = 3e-4
    polyak_tau: float = 5e-3
    weight_decay: float = 1e-6
    max_grad: float = 20.
    eval_every: int = 40_000
    save_replay_every: int = 10 ** 5
    train_after: int = 10_000

    img_size: tuple[int, int] = (96, 96)
    task: str = 'src'
    action_space: str = 'discrete'
    num_envs: int = 16
    seed: int = 2
    logdir: str = 'logdir/dr_image96_cnn_coloreditems'
