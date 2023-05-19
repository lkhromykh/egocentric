import pickle
# import chex
#chex.disable_asserts()

import jax
import haiku as hk
# jax.config.update('jax_platform_name', 'cpu')

from rltools import loggers

from src.rl.config import Config, _DEBUG_CONFIG
from src.rl.builder import Builder
from src.rl.ops import train_loop, eval_loop


def main(cfg: Config):
    rng = jax.random.PRNGKey(cfg.seed)
    rngseq = hk.PRNGSequence(rng)
    builder = Builder(config)

    np_rng1, np_rng2 = next(rngseq).tolist()
    env = builder.make_env(np_rng1)
    buffer = builder.make_replay_buffer(np_rng2, env)
    ds = buffer.as_generator(cfg.batch_size)
    nets = builder.make_networks(env)
    params = nets.init(next(rngseq))
    state = builder.make_training_state(next(rngseq), params)
    step = builder.make_step_fn(nets)

    logger = loggers.TFSummaryLogger(cfg.logdir, label='', step_key='step')
    # logger = loggers.TerminalOutput()

    def policy(obs, train=True):
        return jax.jit(nets.act)(state.params, next(rngseq), obs, train)

    ts = env.reset()
    interactions = 0
    grad_steps = cfg.utd * cfg.sequence_len // cfg.batch_size
    while True:
        trajectory, ts = train_loop(env, policy, cfg.sequence_len, ts)
        buffer.add(trajectory)
        interactions += cfg.sequence_len
        if interactions < cfg.train_after:
            continue
        for _ in range(grad_steps):
            batch = next(ds)
            state, metrics = step(state, batch)
        metrics.update(step=interactions)

        if (interactions % cfg.eval_every) < cfg.sequence_len:
            eval_reward = eval_loop(env, lambda obs: policy(obs, False))
            metrics.update(eval_reward=eval_reward)
            ts = env.reset()
        logger.write(metrics)

    with open(builder.exp_path(Builder.PARAMS), 'wb') as f:
        pickle.dump(state.params, f)


if __name__ == '__main__':
    config = Config.from_entrypoint()
    # config = _DEBUG_CONFIG
    main(config)
