import jax
import jax.numpy as jnp
import chex
import haiku as hk
import numpy as np


def random_crop(rng: chex.PRNGKey,
                img: chex.Array,
                crop_size: int
                ) -> chex.Array:
    """Crop HW dims preserving original shape via padding."""
    chex.assert_scalar_positive(crop_size)
    chex.assert_type(img, jnp.uint8)
    chex.assert_rank(img, 3)

    pad = (crop_size, crop_size)
    nopad = (0, 0)
    pad_with = 2 * (pad,) + (nopad,)
    crop = jax.random.randint(rng, (2,), 0, 2 * crop_size + 1)
    crop = jnp.concatenate([crop, jnp.zeros(1)], dtype=jnp.int32)
    padded = jnp.pad(img, pad_with, mode='edge')
    return jax.lax.dynamic_slice(padded, crop, img.shape)


def batched_random_crop(rng: chex.PRNGKey,
                        img: chex.Array,
                        crop_size: int
                        ) -> chex.Array:
    prefix = img.shape[:-3]
    if not prefix:
        return random_crop(rng, img, crop_size)
    batch = np.prod(prefix, dtype=int)
    rngs = jax.random.split(rng, batch)
    rngs = rngs.reshape(prefix + (2,))
    op = jax.vmap(random_crop, (0, 0, None))
    op = hk.BatchApply(op, len(prefix))
    return op(rngs, img, crop_size)


augmentation_fn = batched_random_crop
