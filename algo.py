from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
import torch.nn as nn
import torch
import numpy as np
from torch import tensor
from absl import app

#Constants
MAX_SELECT=200
MAX_CARGO=8
MAX_QUEUE=5

torch.set_default_device('cuda')

class ZergAgent(base_agent.BaseAgent, nn.Module):
  def __init__(self):
    base_agent.BaseAgent.__init__(self)
    nn.Module.__init__(self)
    self.feature_image=nn.Sequential(
      nn.Conv2d(27, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),

      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),

      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),

      nn.Flatten(),
      nn.Linear(128 * 10 * 10, 512),
      nn.ReLU()
    )
    self.feature_data=nn.Sequential(
      nn.Linear(249, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU()
    )
    self.classifier=nn.Linear(64, 5)
    self.locater=nn.Linear(64, 8)
    self.mu=nn.Linear(64, 2)
    self.sigma=nn.Parameter(torch.zeros(2))

  def forward(self, images, data):
    return self.feature_image(images), self.feature_data(data)
  
  def step(self, obs):
    super(ZergAgent, self).step(obs)
    feature_screen=obs.observation['feature_screen']
    player_data=torch.tensor(obs.observation['player'])
    ctrl_data=torch.tensor(obs.observation['control_groups']).flatten()
    multi_select=torch.tensor(obs.observation['multi_select'][:MAX_SELECT]) if len(obs.observation['multi_select']) != 0 else torch.zeros(MAX_SELECT)
    cargo_data=torch.tensor(obs.observation['cargo'][:MAX_CARGO]) if len(obs.observation['cargo']) != 0 else torch.zeros(MAX_CARGO)
    prod_queue_data=torch.tensor(obs.observation['production_queue'][:MAX_QUEUE]) if len(obs.observation['production_queue']) != 0 else torch.zeros(MAX_QUEUE)
    build_queue_data=torch.tensor(obs.observation['build_queue'][:MAX_QUEUE]) if len(obs.observation['build_queue']) != 0 else torch.zeros(MAX_QUEUE)

    data=torch.cat((player_data, ctrl_data, multi_select, cargo_data, prod_queue_data, build_queue_data), dim=0)
    data=data.unsqueeze(0)

    feature_screen=tensor(feature_screen, dtype=torch.float32).unsqueeze(0)
    print(self(feature_screen, data))
    return actions.FUNCTIONS.no_op()

def main(unused_argv):
  agent = ZergAgent()
  try:
    env=sc2_env.SC2Env(
          map_name="AbyssalReef",
          players=[sc2_env.Agent(sc2_env.Race.zerg),
                   sc2_env.Bot(sc2_env.Race.random,
                               sc2_env.Difficulty.very_easy)],
          agent_interface_format=features.AgentInterfaceFormat(
              feature_dimensions=features.Dimensions(screen=84, minimap=64)),
          step_mul=16,
          game_steps_per_episode=0,
          visualize=False)
    while True:     
        agent.setup(env.observation_spec(), env.action_spec())
        
        timesteps = env.reset()
        agent.reset()
        
        while True:
          step_actions = [agent.step(timesteps[0])]
          if timesteps[0].last():
            break
          timesteps = env.step(step_actions)
        break
      
  except KeyboardInterrupt:
    pass
  
if __name__ == "__main__":
  app.run(main)