import math

import einops
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
from ml_dtypes import bfloat16 as np_bfloat16_dtype
import jax.numpy as jnp
import ninjax as nj
import numpy as np

import torch
import perception_models.core.vision_encoder.pe as pe
from custom_models.dino import CheckpointableFlaxDinov2Model
from transformers import AutoImageProcessor, FlaxDinov2Model

from torch.nn import functional as F
from PIL import Image

f32 = jnp.float32
sg = jax.lax.stop_gradient


class RSSM(nj.Module):

    deter: int = 4096
    hidden: int = 2048
    stoch: int = 32
    classes: int = 32
    norm: str = 'rms'
    act: str = 'gelu'
    unroll: bool = False
    unimix: float = 0.01
    outscale: float = 1.0
    imglayers: int = 2
    obslayers: int = 1
    dynlayers: int = 1
    absolute: bool = False
    blocks: int = 8
    free_nats: float = 1.0

    def __init__(self, act_space, **kw):
        assert self.deter % self.blocks == 0
        self.act_space = act_space
        self.kw = kw

    @property
    def entry_space(self):
        return dict(
            deter=elements.Space(np.float32, self.deter),
            stoch=elements.Space(np.float32, (self.stoch, self.classes)))

    def initial(self, bsize):
        # Deter is the recurrent state, stoch is the latent variable
        carry = nn.cast(dict(
            deter=jnp.zeros([bsize, self.deter], f32),
            stoch=jnp.zeros([bsize, self.stoch, self.classes], f32)))
        return carry

    def truncate(self, entries, carry=None):
        assert entries['deter'].ndim == 3, entries['deter'].shape
        carry = jax.tree.map(lambda x: x[:, -1], entries)
        return carry

    def starts(self, entries, carry, nlast):
        B = len(jax.tree.leaves(carry)[0])
        return jax.tree.map(
            lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

    def observe(self, carry, tokens, action, reset, training, single=False):
        # tokens are the encoded observations
        carry, tokens, action = nn.cast((carry, tokens, action))
        if single:
            carry, (entry, feat) = self._observe(
                carry, tokens, action, reset, training)
            return carry, entry, feat
        else:
            unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
            carry, (entries, feat) = nj.scan(
                lambda carry, inputs: self._observe(
                    carry, *inputs, training),
                carry, (tokens, action, reset), unroll=unroll, axis=1)
            return carry, entries, feat

    def _observe(self, carry, tokens, action, reset, training):
        """
        Update recurrent state using previous state and action.
        Update latent state using encoded observations (tokens).
        """
        # zero out states and action if reset flag active
        deter, stoch, action = nn.mask(
            (carry['deter'], carry['stoch'], action), ~reset)

        action = nn.DictConcat(self.act_space, 1)(action)
        action = nn.mask(action, ~reset)

        # update the recurrent state using the GRU
        deter = self._core(deter, stoch, action)

        # update the latent state using a feedforward network and encoded observation (tokens)
        tokens = tokens.reshape((*deter.shape[:-1], -1))
        x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
        for i in range(self.obslayers):
            x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(
                self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))

        # sample distribution over latent state to produce stochastic feature
        logit = self._logit('obslogit', x)
        stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

        carry = dict(deter=deter, stoch=stoch)
        feat = dict(deter=deter, stoch=stoch, logit=logit)
        entry = dict(deter=deter, stoch=stoch)
        assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
        return carry, (entry, feat)

    def imagine(self, carry, policy, length, training, single=False):
        """
        Update recurrent state from sampled policy action.
        Update latent state using just the new recurrent state.
        """
        if single:
            # select action based on model state
            action = policy(sg(carry)) if callable(policy) else policy
            actemb = nn.DictConcat(self.act_space, 1)(action)

            # update recurrent state base on action
            deter = self._core(carry['deter'], carry['stoch'], actemb)

            # use recurrent state only to predict the next latent state
            logit = self._prior(deter)
            stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

            carry = nn.cast(dict(deter=deter, stoch=stoch))
            feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))
            assert all(x.dtype == nn.COMPUTE_DTYPE for x in (
                deter, stoch, logit))
            return carry, (feat, action)

        else:
            unroll = length if self.unroll else 1
            if callable(policy):
                carry, (feat, action) = nj.scan(
                    lambda c, _: self.imagine(
                        c, policy, 1, training, single=True),
                    nn.cast(carry), (), length, unroll=unroll, axis=1)
            else:
                # we run the imagination for length timesteps and produce length outputs
                carry, (feat, action) = nj.scan(
                    lambda c, a: self.imagine(c, a, 1, training, single=True),
                    nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
            # We can also return all carry entries but it might be expensive.
            # entries = dict(deter=feat['deter'], stoch=feat['stoch'])
            # return carry, entries, feat, action
            return carry, feat, action

    def loss(self, carry, tokens, acts, reset, training):
        # calculate dynamics and representation loss (no reconstruction here)
        metrics = {}
        carry, entries, feat = self.observe(
            carry, tokens, acts, reset, training)

        # minimise distance between distributions over latent states
        # between prior and posterior (from observation)
        prior = self._prior(feat['deter'])
        post = feat['logit']
        dyn = self._dist(sg(post)).kl(self._dist(prior))
        rep = self._dist(post).kl(self._dist(sg(prior)))

        # if kl is below some value set to free_nats we do not need to minimise more
        if self.free_nats:
            dyn = jnp.maximum(dyn, self.free_nats)
            rep = jnp.maximum(rep, self.free_nats)

        losses = {'dyn': dyn, 'rep': rep}
        metrics['dyn_ent'] = self._dist(prior).entropy().mean()
        metrics['rep_ent'] = self._dist(post).entropy().mean()
        return carry, entries, losses, feat, metrics

    def _core(self, deter, stoch, action):
        # Implements the GRU to predict the forward dynamics
        stoch = stoch.reshape((stoch.shape[0], -1))
        # Normalise
        action /= sg(jnp.maximum(1, jnp.abs(action)))
        g = self.blocks

        def flat2group(x): return einops.rearrange(
            x, '... (g h) -> ... g h', g=g)
        def group2flat(x): return einops.rearrange(
            x, '... g h -> ... (g h)', g=g)

        # transform input to correct size
        # recurrent state
        x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
        x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
        # latent state
        x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
        x1 = nn.act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
        # action
        x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
        x2 = nn.act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))

        # concatenate output
        x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
        x = group2flat(jnp.concatenate([flat2group(deter), x], -1))

        # apply GRU operations
        for i in range(self.dynlayers):
            x = self.sub(f'dynhid{i}', nn.BlockLinear,
                         self.deter, g, **self.kw)(x)
            x = nn.act(self.act)(
                self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))
        x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)

        gates = jnp.split(flat2group(x), 3, -1)
        reset, cand, update = [group2flat(x) for x in gates]
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deter

        return deter

    def _prior(self, feat):
        # predict latent state distribution given recurrent state
        x = feat
        for i in range(self.imglayers):
            x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(
                self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
        return self._logit('priorlogit', x)

    def _logit(self, name, x):
        # produce a categorical distribution over each element of the stochastic staet
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
        return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

    def _dist(self, logits):
        # produce a onehot distribution
        out = embodied.jax.outs.OneHot(logits, self.unimix)
        out = embodied.jax.outs.Agg(out, 1, jnp.sum)
        return out


class Encoder(nj.Module):

    units: int = 1024
    norm: str = 'rms'
    act: str = 'gelu'
    depth: int = 64
    mults: tuple = (2, 3, 4, 4)
    layers: int = 3
    kernel: int = 5
    symlog: bool = True
    outer: bool = False
    strided: bool = False

    def __init__(self, obs_space, freeze=False, **kw):
        assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
        self.obs_space = obs_space

        # Separate out vector and image observations
        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]

        # The output channels of the CNN layers
        self.depths = tuple(self.depth * mult for mult in self.mults)
        self.kw = kw

    @property
    def entry_space(self):
        return {}

    def initial(self, batch_size):
        return {}

    def truncate(self, entries, carry=None):
        return {}

    def encode_vector_obs(self, obs, bdims):
        vspace = {k: self.obs_space[k] for k in self.veckeys}
        vecs = {k: obs[k] for k in self.veckeys}

        # Symlog or identity function if specified
        squish = nn.symlog if self.symlog else lambda x: x
        x = nn.DictConcat(vspace, 1, squish=squish)(vecs)

        # Flattens the bdims dimensions into one (time and batch dimensions I think)
        x = x.reshape((-1, *x.shape[bdims:]))

        # MLP forward pass
        for i in range(self.layers):
            # Create/lookup a linear layer with the specified units
            x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
            x = nn.act(self.act)(
                self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))

        return x

    def encode_image_obs(self, obs, bdims, training=False):
        K = self.kernel
        imgs = [obs[k] for k in sorted(self.imgkeys)]
        assert all(x.dtype == jnp.uint8 for x in imgs)

        # Concatenate images along the last dimension (channel dimension)
        # and flatten the batch and time dimensions
        x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
        x = x.reshape((-1, *x.shape[bdims:]))

        # Forward pass through the image encoder
        for i, depth in enumerate(self.depths):
            if self.outer and i == 0:
                x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)

            elif self.strided:
                x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)

            else:
                x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
                B, H, W, C = x.shape
                # Effectively 2x2 max pooling halving dimensions: (B, H/2, W/2, C)
                x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))

            # Apply activation and normalization
            x = nn.act(self.act)(
                self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))

        assert 3 <= x.shape[-3] <= 16, x.shape
        assert 3 <= x.shape[-2] <= 16, x.shape
        # Flatten all dimensions except the first (batch dimension)
        x = x.reshape((x.shape[0], -1))

        return x

    def __call__(self, carry, obs, reset, training, single=False):
        bdims = 1 if single else 2
        outs = []
        bshape = reset.shape

        if self.veckeys:
            x = self.encode_vector_obs(obs, bdims)
            # Store output
            outs.append(x)

        if self.imgkeys:
            x = self.encode_image_obs(obs, bdims, training)
            # Store output
            outs.append(x)

        # Form a single output tensor by concatenating all outputs
        x = jnp.concatenate(outs, -1)

        # Recover time and batch dimensions
        tokens = x.reshape((*bshape, *x.shape[1:]))
        entries = {}
        return carry, entries, tokens


class PEEncoder(Encoder):
    """
    Use facebook perception encoder to encode image observation producing a single 1024 vector.
    Compared to default vector of length 4x4x256=4096.
    """
    size: int = 224
    units: int = 1024
    norm: str = 'rms'
    act: str = 'gelu'
    depth: int = 64
    mults: tuple = (2, 3, 4, 4)
    layers: int = 3
    kernel: int = 5
    symlog: bool = True
    outer: bool = False
    strided: bool = False

    def __init__(self, obs_space, freeze=False, **kw):
        super().__init__(obs_space, **kw)

        model = pe.VisionTransformer.from_config(
            "PE-Core-B16-224", pretrained=True)  # Downloads from HF
        self.model = model.cuda()

    def resize_batch_with_pillow(self, images):
        """
        Resize a batch of images using Pillow's bilinear interpolation.

        Parameters:
            images (np.ndarray): Array of shape (B, H, W, C).
            target_size (tuple): New size as (width, height).

        Returns:
            np.ndarray: Resized images with shape (B, target_height, target_width, C).
        """
        resized = []
        for img in images:
            # Convert image to uint8 if needed
            pil_img = Image.fromarray(np.uint8(img))
            pil_img = pil_img.resize((self.size, self.size), Image.BILINEAR)
            resized.append(np.array(pil_img))
        return np.stack(resized)

    def _pe_call(self, x):
        with torch.no_grad(), torch.autocast("cuda"):
            x_np = np.asarray(x)
            x_np_resized = self.resize_batch_with_pillow(x_np)
            x_pt = torch.from_numpy(x_np_resized)
            x_pt = x_pt.half().cuda()
            x_pt = x_pt.permute(0, 3, 1, 2)

            output_pt = self.model(x_pt)
            # Ensure output is on CPU before converting back to NumPy for JAX
            output_np = output_pt.cpu().numpy()
            return output_np.astype(np_bfloat16_dtype)

    def encode_image_obs(self, obs, bdims, training=False):
        imgs = [obs[k] for k in sorted(self.imgkeys)]
        assert all(x.dtype == jnp.uint8 for x in imgs)

        # Concatenate images along the last dimension (channel dimension)
        # and flatten the batch and time dimensions
        x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
        x_reshaped = x.reshape((-1, *x.shape[bdims:]))

        result_shape_abstract = jax.ShapeDtypeStruct(
            (x_reshaped.shape[0], 1024), x_reshaped.dtype)
        x_from_pytorch = jax.pure_callback(
            self._pe_call,  # The Python function to call
            result_shape_abstract,    # An abstract JAX array representing the output shape and dtype
            x_reshaped,               # Arguments to the callback function
            vectorized=True          # Set to True if your callback is vectorized for vmap
        )

        x = nn.act(self.act)(
            self.sub(f'pe_norm', nn.Norm, self.norm)(x_from_pytorch))

        x = x.reshape((x.shape[0], -1))
        return x


print("Loading Dinov2...")
_DINOV2_MODULE = CheckpointableFlaxDinov2Model.from_pretrained(
    "facebook/dinov2-small", dtype=jax.numpy.bfloat16)
DINO_PARAMS_HOST = jax.tree_util.tree_map(
    lambda x: np.asarray(x, dtype=x.dtype), _DINOV2_MODULE.params)


class DinoEncoder(nj.Module):
    def __init__(self, freeze=False):
        super().__init__()
        self.freeze = freeze

    """Thin wrapper that registers Dinov2 parameters in the Ninjax tree."""

    def __call__(self, x, *, train: bool):
        # 1) Retrieve or create the param tree inside Ninjax
        params = nj.Variable(
            lambda: DINO_PARAMS_HOST,
            name='pretrained_vision_params',
        ).read()

        # (Batch, Sequence, Hidden Size)
        out = _DINOV2_MODULE(
            x, train=train, params=params).last_hidden_state[:, 1:]
        out = out.reshape((out.shape[0], -1))
        return sg(out) if self.freeze else out


RESIZE = 256
CROP = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_images(imgs):
    imgs = imgs.astype(jnp.float32)
    batch_size, h, w, c = imgs.shape

    # Resize if necessary
    if h != RESIZE:
        imgs = jax.image.resize(
            imgs, (batch_size, RESIZE, RESIZE, c), method="cubic")

    # Central Crop
    left = top = (RESIZE - CROP) // 2
    imgs = jax.lax.dynamic_slice(
        imgs, (0, top, left, 0), (batch_size, CROP, CROP, c))

    # Rescale
    imgs = imgs / 255.0          # rescale
    imgs = (imgs - MEAN) / STD  # normalize

    return imgs


class DINOv2Encoder(Encoder):
    """
    Use facebook perception encoder to encode image observation producing a single 1024 vector.
    Compared to default vector of length 4x4x256=4096.
    """
    units: int = 1024
    norm: str = 'rms'
    act: str = 'gelu'
    depth: int = 64
    mults: tuple = (2, 3, 4, 4)
    layers: int = 3
    kernel: int = 5
    symlog: bool = True
    outer: bool = False
    strided: bool = False

    def __init__(self, obs_space, freeze=False, **kw):
        super().__init__(obs_space, **kw)
        self.freeze = freeze

    def encode_image_obs(self, obs, bdims, training=False):
        imgs = [obs[k] for k in sorted(self.imgkeys)]
        assert all(x.dtype == jnp.uint8 for x in imgs)

        x = nn.cast(jnp.concatenate(imgs, -1), force=True)
        x = x.reshape((-1, *x.shape[bdims:]))
        x = preprocess_images(x)
        x = jax.numpy.permute_dims(x, (0, 3, 1, 2))

        # Forward pass through the image encoder
        x = self.sub('dino_enc', DinoEncoder, self.freeze)(x, train=training)

        # Apply activation and normalization
        x = nn.act(self.act)(self.sub(f'dino_norm', nn.Norm, self.norm)(x))

        # Flatten all dimensions except the first (batch dimension)
        x = x.reshape((x.shape[0], -1))

        return x


class Decoder(nj.Module):

    units: int = 1024
    norm: str = 'rms'
    act: str = 'gelu'
    outscale: float = 1.0
    depth: int = 64
    mults: tuple = (2, 3, 4, 4)
    layers: int = 3
    kernel: int = 5
    symlog: bool = True
    bspace: int = 8
    outer: bool = False
    strided: bool = False

    def __init__(self, obs_space, **kw):
        assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
        self.obs_space = obs_space

        # Separate out vector and image observations
        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]

        # The output channels of the CNN layers
        self.depths = tuple(self.depth * mult for mult in self.mults)

        # Get the channels and resolutions in the image observations
        self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
        self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
        self.kw = kw

    @property
    def entry_space(self):
        return {}

    def initial(self, batch_size):
        return {}

    def truncate(self, entries, carry=None):
        return {}

    def __call__(self, carry, feat, reset, training, single=False):
        assert feat['deter'].shape[-1] % self.bspace == 0
        K = self.kernel
        recons = {}
        bshape = reset.shape

        # Input is the array of the latent state and recurrent state
        inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]

        # Flatten the batch and time dimensions and concatenate
        inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
        inp = jnp.concatenate(inp, -1)

        if self.veckeys:
            # Create a mapping for vector observations to output types
            spaces = {k: self.obs_space[k] for k in self.veckeys}
            o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')
            outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}

            # Forward pass of the MLP for vector observations and recover batch and time dimensions
            kw = dict(**self.kw, act=self.act, norm=self.norm)
            x = self.sub('mlp', nn.MLP, self.layers, self.units, **kw)(inp)
            x = x.reshape((*bshape, *x.shape[1:]))

            # Output distributions over the vector observations
            kw = dict(**self.kw, outscale=self.outscale)
            outs = self.sub('vec', embodied.jax.DictHead,
                            spaces, outputs, **kw)(x)
            recons.update(outs)

        if self.imgkeys:
            # Determine the scaling from original image to encoded resolution
            # other than the outer layer we halve the resolution at each layer
            factor = 2 ** (len(self.depths) - int(bool(self.outer)))
            minres = [int(x // factor) for x in self.imgres]

            assert 3 <= minres[0] <= 16, minres
            assert 3 <= minres[1] <= 16, minres
            shape = (*minres, self.depths[-1])

            # If we have a bspace, we separate the deterministic and stochastic features
            if self.bspace:
                u, g = math.prod(shape), self.bspace
                x0, x1 = nn.cast((feat['deter'], feat['stoch']))
                x1 = x1.reshape((*x1.shape[:-2], -1))
                x0 = x0.reshape((-1, x0.shape[-1]))
                x1 = x1.reshape((-1, x1.shape[-1]))

                # forward pass on deterministic features
                x0 = self.sub('sp0', nn.BlockLinear, u, g, **self.kw)(x0)
                x0 = einops.rearrange(
                    x0, '... (g h w c) -> ... h w (g c)',
                    h=minres[0], w=minres[1], g=g)

            # forward pass on stochastic features
                x1 = self.sub('sp1', nn.Linear, 2 * self.units, **self.kw)(x1)
                x1 = nn.act(self.act)(
                    self.sub('sp1norm', nn.Norm, self.norm)(x1))
                x1 = self.sub('sp2', nn.Linear, shape, **self.kw)(x1)

                # Final activation and normalisation on the sum of the two parts
                x = nn.act(self.act)(
                    self.sub('spnorm', nn.Norm, self.norm)(x0 + x1))

            else:
                # if not bspace, we just pass the input through a linear layer
                x = self.sub('space', nn.Linear, shape, **kw)(inp)
                x = nn.act(self.act)(
                    self.sub('spacenorm', nn.Norm, self.norm)(x))

            for i, depth in reversed(list(enumerate(self.depths[:-1]))):
                # use a strided transposed convolution to upsample the image
                if self.strided:
                    kw = dict(**self.kw, transp=True)
                    x = self.sub(f'conv{i}', nn.Conv2D, depth, K, 2, **kw)(x)

                # use standard convolution along with upsampling/cloning values along the width and height
                else:
                    x = x.repeat(2, -2).repeat(2, -3)
                    x = self.sub(f'conv{i}', nn.Conv2D, depth, K, **self.kw)(x)

                x = nn.act(self.act)(
                    self.sub(f'conv{i}norm', nn.Norm, self.norm)(x))

            # Last layer produces the correct number of output channels
            if self.outer:
                kw = dict(**self.kw, outscale=self.outscale)
                x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)

            elif self.strided:
                kw = dict(**self.kw, outscale=self.outscale, transp=True)
                x = self.sub('imgout', nn.Conv2D, self.imgdep, K, 2, **kw)(x)

            else:
                x = x.repeat(2, -2).repeat(2, -3)
                kw = dict(**self.kw, outscale=self.outscale)
                x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)

            # Shrink values between 0 and 1
            x = jax.nn.sigmoid(x)
            x = x.reshape((*bshape, *x.shape[1:]))

            # Split the output tensor into separate images for each key
            split = np.cumsum(
                [self.obs_space[k].shape[-1] for k in self.imgkeys][:-1])
            for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
                out = embodied.jax.outs.MSE(out)
                out = embodied.jax.outs.Agg(out, 3, jnp.sum)
                recons[k] = out

        entries = {}
        return carry, entries, recons
