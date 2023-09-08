import time

import cloudpickle
import numpy as np
import jax
import haiku as hk
import chex
chex.disable_asserts()
# jax.config.update('jax_platform_name', 'cpu')

from rltools import loggers

from src.rl.config import Config, _DEBUG_CONFIG
from src.rl.builder import Builder
from src.rl.ops import train_loop, eval_loop


def main(cfg: Config):
    rng = jax.random.PRNGKey(cfg.seed)
    rngseq = hk.PRNGSequence(rng)
    builder = Builder(cfg)

    np_rng1, np_rng2 = next(rngseq).tolist()
    env = builder.make_envs(np_rng1)
    replay = builder.make_replay_buffer(np_rng2, env)
    grad_steps = int(cfg.utd * cfg.sequence_len * cfg.num_envs)
    grad_steps = max(grad_steps, 1)
    ds = replay.as_tfdataset(cfg.batch_size * grad_steps)
    nets = builder.make_networks(env)
    params = nets.init(next(rngseq))
    state = builder.make_training_state(next(rngseq), params)
    step = builder.make_step_fn(nets)
    logger = loggers.TFSummaryLogger(cfg.logdir, label='', step_key='step')
    print("Number of params: %d" % hk.data_structures.tree_size(params))

    def policy(obs, train=True):
        obs = jax.device_put(obs)
        act_logprob = jax.jit(nets.act)(state.params, next(rngseq), obs, train)
        return jax.tree_util.tree_map(np.asarray, act_logprob)

    start = time.time()
    ts = env.reset()
    interactions = 0
    envs_steps = cfg.num_envs * cfg.sequence_len
    while True:
        rngseq.reserve(cfg.sequence_len)
        trajectories, ts = train_loop(env, policy, cfg.sequence_len, ts)
        for i in range(cfg.num_envs):
            trajectory = jax.tree_util.tree_map(lambda t: t[:, i], trajectories)
            replay.add(trajectory)
        interactions += envs_steps
        if interactions < cfg.train_after:
            continue
        batch = next(ds)
        state, metrics = step(state, batch)
        fps = interactions / (time.time() - start)
        metrics.update(step=interactions, fps=fps)

        if (interactions % cfg.eval_every) < envs_steps:
            eval_reward = eval_loop(env, lambda obs: policy(obs, False))
            metrics.update(eval_mean=eval_reward.mean(),
                           eval_std=eval_reward.std())
            ts = env.reset()
            with open(builder.exp_path(Builder.STATE), 'wb') as f:
                cloudpickle.dump(jax.device_get(state), f)
        if (interactions % cfg.save_replay_every) < envs_steps:
            replay.save(builder.exp_path(Builder.REPLAY))
        logger.write(metrics)


if __name__ == '__main__':
    config = Config.from_entrypoint()
    # config = _DEBUG_CONFIG
    main(config)
