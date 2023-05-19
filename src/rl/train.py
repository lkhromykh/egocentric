import time
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

    def policy(obs, train=True):
        return jax.jit(nets.act)(state.params, next(rngseq), obs, train)

    start = time.time()
    ts = env.reset()
    interactions = 0
    grad_steps = cfg.utd * cfg.sequence_len * cfg.num_envs // cfg.batch_size
    while True:
        trajectories, ts = train_loop(env, policy, cfg.sequence_len, ts)
        for i in range(cfg.num_envs):
            trajectory = jax.tree_util.tree_map(lambda t: t[:, i], trajectories)
            buffer.add(trajectory)
        interactions += cfg.sequence_len * cfg.num_envs
        if interactions < cfg.train_after:
            continue
        for _ in range(grad_steps):
            batch = next(ds)
            state, metrics = step(state, batch)
        fps = interactions / (time.time() - start)
        metrics.update(step=interactions, fps=fps)

        if (interactions % cfg.eval_every) < cfg.sequence_len:
            eval_reward = eval_loop(env, lambda obs: policy(obs, False))
            metrics.update(eval_mean=eval_reward.mean(), eval_std=eval_reward.std())
            ts = env.reset()
        logger.write(metrics)

    with open(builder.exp_path(Builder.PARAMS), 'wb') as f:
        pickle.dump(state.params, f)


if __name__ == '__main__':
    config = Config.from_entrypoint()
    # config = _DEBUG_CONFIG
    main(config)
