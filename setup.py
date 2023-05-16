from setuptools import setup

rl_requirements = (
    'jax',
    'jaxlib',
    'tensorflow_probability[jax]',
    'dm-haiku',
    'chex',
    'optax',
    'rltools[all] @ git+https://github.com/lkhromykh/rltools.git@dev'
)
suite_requirements = ('dm_control',)
setup(
)