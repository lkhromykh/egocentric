import pickle
# import chex
#chex.disable_asserts()

import jax
import haiku as hk

from rltools import loggers

from src.rl.config import Config, _DEBUG_CONFIG
from src.rl.builder import Builder
from src.rl.ops import environment_loop


def main(cfg: Config):
    rng = jax.random.PRNGKey(cfg.seed)
    rngseq = hk.PRNGSequence(rng)
    builder = Builder(config)

    np_rng1, np_rng2 = next(rngseq).tolist()
    env = builder.make_env(np_rng1)
    buffer = builder.make_replay_buffer(np_rng2, env)
    ds = buffer.as_generator(cfg.batch_size, cfg.sequence_len)
    nets = builder.make_networks(env)
    params = nets.init(next(rngseq))
    state = builder.make_training_state(next(rngseq), params)
    step = builder.make_step_fn(nets)

    logger = loggers.TerminalOutput()

    def policy(obs):
        return jax.jit(nets.act)(state.params, next(rngseq), obs, True)

    while True:
        trajectory = environment_loop(env, policy)
        breakpoint()
        buffer.add(trajectory)
        # if len(buffer) < cfg.batch_size:
        #     continue
        batch = next(ds)
        state, metrics = step(state, batch)
        logger.write(metrics)

    with open(builder.exp_path(Builder.PARAMS), 'wb') as f:
        pickle.dump(state.params, f)


if __name__ == '__main__':
    config = Config.from_entrypoint()
    config = _DEBUG_CONFIG
    main(config)
