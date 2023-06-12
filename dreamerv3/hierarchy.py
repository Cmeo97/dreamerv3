import functools
import jax
import jax.numpy as jnp
from jax import lax, nn, random
from jax.tree_util import tree_map
import embodied
import numpy as np
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from . import ninjax as nj
from . import jaxutils
from . import agent
from . import expl
from . import nets

sg = lambda x: tree_map(lax.stop_gradient, x) 

class Hierarchy(nj.Module):

  def __init__(self, wm, act_space, config):
    jax.config.update("jax_transfer_guard", "allow")
    self.wm = wm
    self.config = config
    self.extr_reward = lambda traj: self.wm.heads['reward'](traj).mean()[1:]
    self.skill_space = embodied.Space(
        jnp.int32 if config.goal_encoder.dist == 'onehot' else jnp.float32,
        config.skill_shape)

    wconfig = config.update({
        'actor.inputs': self.config.worker_inputs,
        'critic.inputs': self.config.worker_inputs,
    })

    self.worker = agent.ImagActorCritic({
        #'extr': agent.VFunction(lambda s: s['reward_extr'], wconfig, name='wcritic_extr'),
        #'expl': agent.VFunction(lambda s: s['reward_expl'], wconfig, name='wcritic_expl'),
        'goal': agent.VFunction(lambda s: s['reward_goal'], wconfig, name='wcritic_goal'),
    }, config.worker_rews, act_space, wconfig, name='worker') 

    mconfig = config.update({
        'actor_grad_cont': 'reinforce',
        'actent.target': config.manager_actent,
    })
    self.manager = agent.ImagActorCritic({
        'extr': agent.VFunction(lambda s: s['reward_extr'], mconfig, name='mcritic_extr'),
        'expl': agent.VFunction(lambda s: s['reward_expl'], mconfig, name='mcritic_expl'),
        #'goal': agent.VFunction(lambda s: s['reward_goal'], mconfig, name='mcritic_goal'),
    }, config.manager_rews, self.skill_space, mconfig, name='manager')

    # VAE manager prior definition
    shape = self.skill_space.shape
    if self.skill_space.discrete:
      self.prior = jaxutils.OneHotDist(jnp.zeros(shape))
      self.prior = tfd.Independent(self.prior, len(shape) - 1)
    else:
      self.prior = tfd.Normal(jnp.zeros(shape), jnp.ones(shape))
      self.prior = tfd.Independent(self.prior, len(shape))

    self.feat = nets.Input(['deter'])
    self.goal_shape = (self.config.rssm.deter,)
    self.enc = nets.MLP(
        config.skill_shape, dims='context', **config.goal_encoder, name='goal_encoder')
    self.dec = nets.MLP(
        self.goal_shape, dims='context', **self.config.goal_decoder, name='goal_decoder')
    
    self.opt = jaxutils.Optimizer(name='goal_opt', **config.encdec_opt)
    self.opt_actors = jaxutils.Optimizer(name='actors_opt', **config.actor_opt)
    self.update_kl = jaxutils.AutoAdapt((), **self.config.encdec_kl, name='kl_autoadapt') 

    if self.config.imagine == 'Director':
      self.imagine = lambda start: self.wm.imagine_carry_director(
        self.policy, True, start, self.config.imag_horizon,
          self.initial(len(start['is_first'])))
    else:
      self.imagine = lambda start: self.wm.imagine_carry(
        self.policy, True, start, self.config.imag_horizon,
          self.initial(len(start['is_first'])))
    
  

  def initial(self, batch_size):
    return {
        'step': jnp.zeros((batch_size,), jnp.int64),
        'skill': jnp.zeros((batch_size,) + self.config.skill_shape, jnp.float32),
        'goal': jnp.zeros((batch_size,) + self.goal_shape, jnp.float32),
    }


  def policy(self, latent, carry, imag=False):
    duration = self.config.train_skill_duration if imag else (
        self.config.env_skill_duration)
    update = (carry['step'] % duration) == 0
    switch = lambda x, y: (
        jnp.einsum('i,i...->i...', 1 - update.astype(x.dtype), x) +
        jnp.einsum('i,i...->i...', update.astype(x.dtype), y))
    skill = sg(switch(carry['skill'], self.manager.actor(sg(latent)).sample(seed=nj.rng())))
    new_goal = self.dec({'skill': skill, 'context': self.feat(latent)}).mode()
    new_goal = (
        self.feat(latent).astype(jnp.float32) + new_goal
        if self.config.manager_delta else new_goal)
    goal = sg(switch(carry['goal'], new_goal))
    delta = goal - self.feat(latent).astype(jnp.float32)
    dist = self.worker.actor(sg({**latent, 'goal': goal, 'delta': delta}))
    outs = {'action': dist}
    if 'image' in self.wm.heads['decoder'].shapes:
      outs['log_goal'] = self.wm.heads['decoder']({
          'deter': goal, 'stoch': self.wm.rssm.get_stoch(goal),
      })['image'].mode()
    carry = {'step': carry['step'] + 1, 'skill': skill, 'goal': goal}
    return outs, carry

  def train(self, imagine, start, data):
    success = lambda rew: (rew[-1] > 0.7).astype(jnp.float32).mean()
    metrics = {}
    metrics.update(self.train_vae_replay(data)) 
    if self.config.train_jointly == 'efficient':
      traj, mets = self.train_jointly_efficiently(start) 
    else:
      traj, mets = self.train_jointly(start)
    metrics.update(mets)
    metrics['success_manager'] = success(traj['reward_goal'])
    return None, metrics

  def train_jointly(self, start):  # Theoretically Wrong (Probably)
    start = start.copy()
    metrics = {}

    def wloss(start):
      traj = self.imagine(start)
      traj['reward_extr'] = self.extr_reward(traj)
      traj['reward_expl'] = self.expl_reward(traj)
      traj['reward_goal'] = self.goal_reward(traj)
      wtraj = self.split_traj(traj)
      loss, metrics = self.worker.loss(wtraj)
      return loss, (wtraj, metrics)
    def mloss(start):
      traj = self.imagine(start)
      traj['reward_extr'] = self.extr_reward(traj)
      traj['reward_expl'] = self.expl_reward(traj)
      traj['reward_goal'] = self.goal_reward(traj)
      mtraj = self.abstract_traj(traj)
      loss, metrics = self.manager.loss(mtraj)
      return loss, (mtraj, metrics, traj)
    

    mets, (wtraj, worker_metrics) = self.worker.opt(self.worker.actor, wloss, start, has_aux=True)
    worker_metrics.update({f'{k}': v for k, v in mets.items()})
    for key, critic in self.worker.critics.items():
      mets = critic.train(wtraj, self.worker.actor)
      worker_metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    
    mets, (mtraj, manager_metrics, traj) = self.manager.opt(self.manager.actor, mloss, start, has_aux=True)
    manager_metrics.update({f'{k}': v for k, v in mets.items()})
    for key, critic in self.manager.critics.items():
      mets = critic.train(mtraj, self.manager.actor)
      manager_metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    metrics.update({f'worker_{k}': v for k, v in worker_metrics.items()}) 
    metrics.update({f'manager_{k}': v for k, v in manager_metrics.items()}) 
    return traj, metrics

  def train_jointly_efficiently(self, start):  # Theoretically Correct (Probably)
    start = start.copy()
    metrics = {}

    def loss(start):
      traj = self.imagine(start)
      traj['reward_extr'] = self.extr_reward(traj)
      traj['reward_expl'] = self.expl_reward(traj)
      traj['reward_goal'] = self.goal_reward(traj)
      wtraj = self.split_traj(traj)
      mtraj = self.abstract_traj(traj)
      mloss, mmetrics = self.manager.loss(mtraj)
      wloss, wmetrics = self.worker.loss(wtraj)
      loss = mloss + wloss
      return loss, (mloss, wloss, traj, mtraj, wtraj, mmetrics, wmetrics)
    

    mets, (mloss, wloss, traj, mtraj, wtraj, mmetrics, wmetrics) = self.opt_actors([self.worker.actor, self.manager.actor], loss, start, has_aux=True)
    metrics.update({f'agent_{k}': v for k, v in mets.items()})
    metrics['worker_loss'] = wloss
    metrics['manager_loss'] = mloss

    for key, critic in self.worker.critics.items():
      mets = critic.train(wtraj, self.worker.actor)
      metrics.update({f'worker_{key}_critic_{k}': v for k, v in mets.items()})
    
    for key, critic in self.manager.critics.items():
      mets = critic.train(mtraj, self.manager.actor)
      metrics.update({f'manager_{key}_critic_{k}': v for k, v in mets.items()})

    metrics.update({f'worker_{k}': v for k, v in wmetrics.items()}) 
    metrics.update({f'manager_{k}': v for k, v in mmetrics.items()}) 

    return traj, metrics


  def train_vae_replay(self, data):
    metrics = {}
    feat = self.feat(data).astype(jnp.float32)
    if 'context' in self.config.goal_decoder.inputs:
      if self.config.vae_span:
        context = feat[:, 0]
        goal = feat[:, -1]
      else:
        assert feat.shape[1] > self.config.train_skill_duration
        context = feat[:, :-self.config.train_skill_duration]
        goal = feat[:, self.config.train_skill_duration:]
    else:
      goal = context = feat
    
    def loss(goal, context): 
      enc = self.enc({'goal': goal, 'context': context})
      dec = self.dec({'skill': enc.sample(seed=nj.rng()), 'context': context})
      rec = -dec.log_prob(sg(goal))
      kl = tfd.kl_divergence(enc, self.prior)
      kl, mets = self.update_kl(kl)
      assert rec.shape == kl.shape, (rec.shape, kl.shape)
      mets['rec_mean'] = rec.mean()
      mets['rec_std'] = rec.std()
      loss = (rec + kl).mean()
      mets['loss'] = loss
      return loss, mets

    goal_loss, mets = self.opt([self.enc, self.dec], loss, goal, context, has_aux=True)
    metrics.update({f'goal_vae_{k}': v for k, v in mets.items()})
    metrics.update(goal_loss)

    return metrics


#def train_vae_imag(self, traj):
#  metrics = {}
#  feat = self.feat(traj).astype(tf.float32)
#  if 'context' in self.config.goal_decoder.inputs:
#    if self.config.vae_span:
#      context = feat[0]
#      goal = feat[-1]
#    else:
#      assert feat.shape[0] > self.config.train_skill_duration
#      context = feat[:-self.config.train_skill_duration]
#      goal = feat[self.config.train_skill_duration:]
#  else:
#    goal = context = feat
#  with tf.GradientTape() as tape:
#    enc = self.enc({'goal': goal, 'context': context})
#    dec = self.dec({'skill': enc.sample(), 'context': context})
#    rec = -dec.log_prob(sg(goal.astype(tf.float32)))
#    if self.config.goal_kl:
#      kl = tfd.kl_divergence(enc, self.prior)
#      kl, mets = self.kl(kl)
#      metrics.update({f'goalkl_{k}': v for k, v in mets.items()})
#    else:
#      kl = 0.0
#    loss = (rec + kl).mean()
#  metrics.update(self.opt(tape, loss, [self.enc, self.dec]))
#  metrics['goalrec_mean'] = rec.mean()
#  metrics['goalrec_std'] = rec.std()
#  return metrics

  def propose_goal(self, start, impl):
    feat = self.feat(start).astype(jnp.float32)
    if impl == 'replay':
      target = random.permutation(nj.rng(), feat, independent=True).astype(jnp.float32)
      skill = self.enc({'goal': target, 'context': feat}).sample(seed=nj.rng())
      return self.dec({'skill': skill, 'context': feat}).mode()
    if impl == 'replay_direct':
      return random.permutation(nj.rng(), feat, independent=True).astype(jnp.float32)
    if impl == 'manager':
      skill = self.manager.actor(start).sample(seed=nj.rng())
      goal = self.dec({'skill': skill, 'context': feat}).mode()
      goal = feat + goal if self.config.manager_delta else goal
      return goal
    if impl == 'prior':
      skill = self.prior.sample(sample_shape=len(start['is_terminal']), seed=nj.rng())
      return self.dec({'skill': skill, 'context': feat}).mode()
    raise NotImplementedError(impl)
  

  def goal_reward(self, traj):
    feat = self.feat(traj).astype(jnp.float32)
    goal = sg(traj['goal'].astype(jnp.float32))
    skill = sg(traj['skill'].astype(jnp.float32))
    context = sg(
        jnp.repeat(feat[0][None], 1 + self.config.imag_horizon, 0))
    if self.config.goal_reward == 'dot':
      return jnp.einsum('...i,...i->...', goal, feat)[1:]
    elif self.config.goal_reward == 'dir':
      return jnp.einsum(
          '...i,...i->...', jnp.linalg.norm(goal, ord=2, axis=-1), feat)[1:]
    elif self.config.goal_reward == 'normed_inner':
      norm = jnp.linalg.norm(goal, axis=-1, keepdims=True)
      return jnp.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
    elif self.config.goal_reward == 'normed_squared':
      norm = jnp.linalg.norm(goal, axis=-1, keepdims=True)
      return -((goal / norm - feat / norm) ** 2).mean(-1)[1:]
    elif self.config.goal_reward == 'cosine_lower':
      gnorm = jnp.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
      fnorm = jnp.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
      fnorm = jnp.maximum(gnorm, fnorm)
      return jnp.einsum('...i,...i->...', goal / gnorm, feat / fnorm)[1:]
    elif self.config.goal_reward == 'cosine_lower_pos':
      gnorm = jnp.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
      fnorm = jnp.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
      fnorm = jnp.maximum(gnorm, fnorm)
      cos = jnp.einsum('...i,...i->...', goal / gnorm, feat / fnorm)[1:]
      return nn.relu(cos)
    elif self.config.goal_reward == 'cosine_frac':
      gnorm = jnp.linalg.norm(goal, axis=-1) + 1e-12
      fnorm = jnp.linalg.norm(feat, axis=-1) + 1e-12
      goal /= gnorm[..., None]
      feat /= fnorm[..., None]
      cos = jnp.einsum('...i,...i->...', goal, feat)
      mag = jnp.minimum(gnorm, fnorm) / jnp.maximum(gnorm, fnorm)
      return (cos * mag)[1:]
    elif self.config.goal_reward == 'cosine_frac_pos':
      gnorm = jnp.linalg.norm(goal, axis=-1) + 1e-12
      fnorm = jnp.linalg.norm(feat, axis=-1) + 1e-12
      goal /= gnorm[..., None]
      feat /= fnorm[..., None]
      cos = jnp.einsum('...i,...i->...', goal, feat)
      mag = jnp.minimum(gnorm, fnorm) / jnp.maximum(gnorm, fnorm)
      return nn.relu(cos * mag)[1:]
    elif self.config.goal_reward == 'cosine_max':
      gnorm = jnp.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
      fnorm = jnp.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
      norm = jnp.maximum(gnorm, fnorm)
      return jnp.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
    elif self.config.goal_reward == 'cosine_max_pos':
      gnorm = jnp.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
      fnorm = jnp.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
      norm = jnp.maximum(gnorm, fnorm)
      cos = jnp.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
      return nn.relu(cos)
    elif self.config.goal_reward == 'normed_inner_clip':
      norm = jnp.linalg.norm(goal, axis=-1, keepdims=True)
      cosine = jnp.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
      return jnp.clip(cosine, -1.0, 1.0)
    elif self.config.goal_reward == 'normed_inner_clip_pos':
      norm = jnp.linalg.norm(goal, axis=-1, keepdims=True)
      cosine = jnp.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
      return jnp.clip(cosine, 0.0, 1.0)
    elif self.config.goal_reward == 'diff':
      goal = jnp.linalg.norm(goal[:-1], ord=2, axis=-1)
      diff = jnp.concatenate([feat[1:] - feat[:-1]], 0)
      return jnp.einsum('...i,...i->...', goal, diff)
    elif self.config.goal_reward == 'norm':
      return -jnp.linalg.norm(goal - feat, axis=-1)[1:]
    elif self.config.goal_reward == 'squared':
      return -((goal - feat) ** 2).sum(-1)[1:]
    elif self.config.goal_reward == 'epsilon':
      return ((goal - feat).mean(-1) < 1e-3).astype(jnp.float32)[1:]
    elif self.config.goal_reward == 'enclogprob':
      return self.enc({'goal': goal, 'context': context}).log_prob(skill)[1:]
    elif self.config.goal_reward == 'encprob':
      return self.enc({'goal': goal, 'context': context}).prob(skill)[1:]
    elif self.config.goal_reward == 'enc_normed_cos':
      dist = self.enc({'goal': goal, 'context': context})
      probs = dist.distribution.probs_parameter()
      norm = jnp.linalg.norm(probs, axis=[-2, -1], keepdims=True)
      return jnp.einsum('...ij,...ij->...', probs / norm, skill / norm)[1:]
    elif self.config.goal_reward == 'enc_normed_squared':
      dist = self.enc({'goal': goal, 'context': context})
      probs = dist.distribution.probs_parameter()
      norm = jnp.linalg.norm(probs, axis=[-2, -1], keepdims=True)
      return -((probs / norm - skill / norm) ** 2).mean([-2, -1])[1:]
    else:
      raise NotImplementedError(self.config.goal_reward)

  def expl_reward(self, traj):
    feat = self.feat(traj).astype(jnp.float32)
    context = jnp.repeat(feat[0][None], 1 + self.config.imag_horizon, 0)
    enc = self.enc({'goal': feat, 'context': context})
    dec = self.dec({'skill': enc.sample(seed=nj.rng()), 'context': context})
    ll = dec.log_prob(feat)
    kl = tfd.kl_divergence(enc, self.prior)
    if self.config.adver_impl == 'abs':
      return jnp.abs(dec.mode() - feat).mean(-1)[1:]
    elif self.config.adver_impl == 'squared':
      return ((dec.mode() - feat) ** 2).mean(-1)[1:]
    elif self.config.adver_impl == 'elbo_scaled':
      return (kl - ll / self.kl.scale())[1:]
    elif self.config.adver_impl == 'elbo_unscaled':
      return (kl - ll)[1:]
    raise NotImplementedError(self.config.adver_impl)

  def split_traj(self, traj):
    traj = traj.copy()
    k = self.config.train_skill_duration
    assert len(traj['action']) % k == 1
    reshape = lambda x: x.reshape((x.shape[0] // k, k) + x.shape[1:])
    for key, val in list(traj.items()):
      val = jnp.concatenate([0 * val[:1], val], 0) if 'reward' in key else val
      # (1 2 3 4 5 6 7 8 9 10) -> ((1 2 3 4) (4 5 6 7) (7 8 9 10))
      val = jnp.concatenate([reshape(val[:-1]), val[k::k][:, None]], 1)
      # N val K val B val F... -> K val (N B) val F...
      val = val.transpose([1, 0] + list(range(2, len(val.shape))))
      val = val.reshape(
          (val.shape[0], np.prod(val.shape[1:3])) + val.shape[3:])
      val = val[1:] if 'reward' in key else val
      traj[key] = val
    # Bootstrap sub trajectory against current not next goal.
    traj['goal'] = jnp.concatenate([traj['goal'][:-1], traj['goal'][:1]], 0)
    traj['weight'] = jnp.cumprod(
        self.config.discount * traj['cont'], axis=0) / self.config.discount
    return traj

  def abstract_traj(self, traj):
    traj = traj.copy()
    traj['action'] = traj.pop('skill')
    k = self.config.train_skill_duration
    reshape = lambda x: x.reshape((x.shape[0] // k, k) + x.shape[1:])
    weights = jnp.cumprod(reshape(traj['cont'][:-1]), axis=1)
    for key, value in list(traj.items()):
      if 'reward' in key:
        traj[key] = (reshape(value) * weights).mean(1)
      elif key == 'cont':
        traj[key] = jnp.concatenate([value[:1], reshape(value[1:]).prod(1)], 0)
      else:
        traj[key] = jnp.concatenate([reshape(value[:-1])[:, 0], value[-1:]], 0)
    traj['weight'] = jnp.cumprod(
        self.config.discount * traj['cont'], axis=0) / self.config.discount
    return traj

  #def abstract_traj_old(self, traj):
  #  traj = traj.copy()
  #  traj['action'] = traj.pop('skill')
  #  mult = tf.math.cumprod(traj['cont'][1:], 0)
  #  for key, value in list(traj.items()):
  #    if 'reward' in key:
  #      traj[key] = (mult * value).mean(0)[None]
  #    elif key == 'cont':
  #      traj[key] = tf.stack([value[0], value[1:].prod(0)], 0)
  #    else:
  #      traj[key] = tf.stack([value[0], value[-1]], 0)
  #  return traj

  def report(self, data):
    metrics = {}
    for impl in ('manager', 'prior', 'replay'):
      for key, video in self.report_worker(data, impl).items():
        metrics[f'impl_{impl}_{key}'] = video
    return metrics

  def report_worker(self, data, impl):

    # Prepare initial state.
    decoder = self.wm.heads['decoder']
    states, _ = self.wm.rssm.observe(
        self.wm.encoder(data)[:6], data['action'][:6], data['is_first'][:6])
    start = {k: v[:, 4] for k, v in states.items()}
    start['is_terminal'] = data['is_terminal'][:6, 4]
    goal = self.propose_goal(start, impl)
    # Worker rollout.
    worker = lambda s: self.worker.actor({
        **s, 'goal': goal, 'delta': goal - self.feat(s).astype(jnp.float32),
    }).sample(seed=nj.rng())
    traj = self.wm.imagine(
        worker, start, self.config.worker_report_horizon)
    # Decoder into images.
    initial = decoder(start)
    target = decoder({'deter': goal, 'stoch': self.wm.rssm.get_stoch(goal)})
    rollout = decoder(traj)
    # Stich together into videos.
    videos = {}
    for k in rollout.keys():
      if k not in decoder.cnn_shapes:
        continue
      length = 1 + self.config.worker_report_horizon
      rows = []
      rows.append(jnp.repeat(initial[k].mode()[:, None], length, 1))
      if target is not None:
        rows.append(jnp.repeat(target[k].mode()[:, None], length, 1))
      rows.append(rollout[k].mode().transpose((1, 0, 2, 3, 4)))
      videos[k] = jaxutils.video_grid(jnp.concatenate(rows, 2))
    return videos
