import dataclasses

from rltools.config import Config as _Config

Layers = tuple[int, ...]


@dataclasses.dataclass
class Config(_Config):

    gamma: float = .98
    lambda_: float = 1.
    entropy_coef: float = 1e-3
    num_actions: int = 20

    # Architecture
    asymmetric: bool = True
    densenet_layers: Layers = (2, 2, 2, 2)
    densenet_growth_rate: int = 16
    actor_keys: str = r'image|length|object_detected'
    actor_layers: Layers = (256,)
    critic_keys: str = r'robotiq_2f85|model'
    critic_layers: Layers = (64, 256, 256)
    ensemble_size: int = 2

    # Train
    jit: bool = True
    buffer_capacity: int = 10 ** 5
    batch_size: int = 64
    sequence_len: int = 8
    utd: float = .1
    learning_rate: float = 3e-4
    polyak_tau: float = 5e-3
    weight_decay: float = 1e-4
    max_grad: float = 10.
    eval_every: int = 40_000
    save_replay_every: int = 10 ** 5
    train_after: int = 10_000

    img_size: tuple[int, int] = (64, 64)
    task: str = 'src'
    action_space: str = 'discrete'
    num_envs: int = 16
    seed: int = 1
    logdir: str = 'logdir/nodr_image64_densenet_oneitem1'
