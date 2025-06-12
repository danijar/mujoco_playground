# import functools

import elements
# import jax
# import mujoco
# import numpy as np

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax.training import orbax_utils
from mujoco_playground import registry
from mujoco_playground import wrapper
# from mujoco_playground._src.gait import draw_joystick_command
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp

env_name = 'G1JoystickFlatTerrain'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)
ppo_params = locomotion_params.brax_ppo_config(env_name)

timestamp = elements.timestamp()
logdir = elements.Path('~/logdir/playground') / str(timestamp)
logdir.mkdir()
elements.print(f'Logdir: {logdir}', color='yellow')

ppo_training_params = dict(ppo_params)
del ppo_training_params['network_factory']
network_factory = ppo_networks.make_ppo_networks

# if 'network_factory' in ppo_params:
#   del ppo_training_params['network_factory']
#   network_factory = functools.partial(
#       ppo_networks.make_ppo_networks,
#       **ppo_params.network_factory
#   )

def progress_fn(num_steps, metrics):
  mean = metrics['eval/episode_reward']
  std = metrics['eval/episode_reward_std']
  print(f'STEP {num_steps} RETURN {mean:.2f} (std={std:.2f})')

def checkpoint_fn(current_step, make_policy, params):
  print(current_step)
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params)
  path = logdir / 'ckpt' / str(current_step)
  orbax_checkpointer.save(path, params, force=True, save_args=save_args)
  print('Wrote checkpoint:', path)

make_inference_fn, params, metrics = ppo.train(
    network_factory=network_factory,
    randomization_fn=registry.get_domain_randomizer(env_name),
    progress_fn=progress_fn,
    policy_params_fn=checkpoint_fn,
    environment=env,
    eval_env=registry.load(env_name, config=env_cfg),
    wrap_env_fn=wrapper.wrap_for_brax_training,
    **dict(ppo_training_params),
)

# env = registry.load(env_name)
# eval_env = registry.load(env_name)
# jit_reset = jax.jit(eval_env.reset)
# jit_step = jax.jit(eval_env.step)
# jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

# rng = jax.random.PRNGKey(1)
#
# rollout = []
# modify_scene_fns = []
#
# x_vel = 1.0
# y_vel = 0.0
# yaw_vel = 0.0
# command = jp.array([x_vel, y_vel, yaw_vel])
#
# phase_dt = 2 * jp.pi * eval_env.dt * 1.5
# phase = jp.array([0, jp.pi])
#
# for j in range(1):
#   print(f'episode {j}')
#   state = jit_reset(rng)
#   state.info['phase_dt'] = phase_dt
#   state.info['phase'] = phase
#   for i in range(env_cfg.episode_length):
#     act_rng, rng = jax.random.split(rng)
#     ctrl, _ = jit_inference_fn(state.obs, act_rng)
#     state = jit_step(state, ctrl)
#     if state.done:
#       break
#     state.info['command'] = command
#     rollout.append(state)
#
#     xyz = np.array(state.data.xpos[eval_env.mj_model.body('torso_link').id])
#     xyz += np.array([0, 0.0, 0])
#     x_axis = state.data.xmat[eval_env._torso_body_id, 0]
#     yaw = -np.arctan2(x_axis[1], x_axis[0])
#     modify_scene_fns.append(
#         functools.partial(
#             draw_joystick_command,
#             cmd=state.info['command'],
#             xyz=xyz,
#             theta=yaw,
#             scl=np.linalg.norm(state.info['command']),
#         )
#     )
#
# render_every = 1
# fps = 1.0 / eval_env.dt / render_every
# print(f'fps: {fps}')
# traj = rollout[::render_every]
# mod_fns = modify_scene_fns[::render_every]
#
# scene_option = mujoco.MjvOption()
# scene_option.geomgroup[2] = True
# scene_option.geomgroup[3] = False
# scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
# scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
# scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
#
# frames = eval_env.render(
#     traj,
#     camera='track',
#     scene_option=scene_option,
#     width=640*2,
#     height=480,
#     modify_scene_fns=mod_fns,
# )

import ipdb; ipdb.set_trace()

# media.save_video(frames, fps=fps, loop=False)
