from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
import torch.nn as nn
import torch
from absl import app
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from collections import deque
import numpy as np

torch.set_default_dtype(torch.float16)
torch.set_default_device('cuda')

#Actions
SELECT_POINT=actions.FUNCTIONS.select_point.id
MOVE_CAMERA=actions.FUNCTIONS.move_camera.id
SELECT_CONTROL_GROUP=actions.FUNCTIONS.select_control_group.id
HARVERST_GATHER=actions.FUNCTIONS.Harvest_Gather_screen.id
ATTACK_MINIMAP=actions.FUNCTIONS.Attack_minimap.id
TRAIN_DRONE_QUICK=actions.FUNCTIONS.Train_Drone_quick.id
BUILD_SPAWNPOOL=actions.FUNCTIONS.Build_SpawningPool_screen.id
TRAIN_ZERGLING=actions.FUNCTIONS.Train_Zergling_quick.id
TRAIN_OVERLORD=actions.FUNCTIONS.Train_Overlord_quick.id
HARVEST_RETURN = actions.FUNCTIONS.Harvest_Return_quick.id
BUILD_HATCHERY = actions.FUNCTIONS.Build_Hatchery_screen.id
BUILD_EXTRACTOR = actions.FUNCTIONS.Build_Extractor_screen.id
BUILD_EVOLUTIONCHAMBER = actions.FUNCTIONS.Build_EvolutionChamber_screen.id
BUILD_QUEEN = actions.FUNCTIONS.Train_Queen_quick.id
BUILD_ROACHWARREN = actions.FUNCTIONS.Build_RoachWarren_screen.id
TRAIN_ROACH = actions.FUNCTIONS.Train_Roach_quick.id
BUILD_HYDRALISKDEN = actions.FUNCTIONS.Build_HydraliskDen_screen.id
TRAIN_HYDRALISK = actions.FUNCTIONS.Train_Hydralisk_quick.id
NO_OP=actions.FUNCTIONS.no_op.id

action_ids=[SELECT_POINT, MOVE_CAMERA, SELECT_CONTROL_GROUP, HARVERST_GATHER, ATTACK_MINIMAP, TRAIN_DRONE_QUICK, BUILD_SPAWNPOOL, TRAIN_ZERGLING, TRAIN_OVERLORD, HARVEST_RETURN, BUILD_HATCHERY, BUILD_EXTRACTOR, BUILD_EVOLUTIONCHAMBER, BUILD_QUEEN, BUILD_ROACHWARREN, TRAIN_ROACH, BUILD_HYDRALISKDEN, TRAIN_HYDRALISK, NO_OP]

#Constants
CONV_OUTPUT=256
DATA_OUTPUT=64
COMBINED_OUTPUT=64

N_ACTIONS=len(action_ids)

MAP_SIZE=84
MAX_SELECT=200
MAX_CARGO=8
MAX_QUEUE=5

class Critic(nn.Module):
  def __init__(self):
    super(Critic, self).__init__()
    self.main=nn.Sequential(
      nn.Linear(COMBINED_OUTPUT, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(), 
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 1),
    )
    self.optimizer=torch.optim.Adam(self.parameters(), lr=0.001)

  def forward(self, x):
    return self.main(x)
  
  def step_optimizer(self, loss):
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()


class ZergAgent(base_agent.BaseAgent, nn.Module):
  def __init__(self):
    base_agent.BaseAgent.__init__(self)
    nn.Module.__init__(self)

    self.critic=Critic()

    self.optimizer=torch.optim.Adam(self.parameters(), lr=0.001)

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
      nn.Linear(128 * 10 * 10, CONV_OUTPUT),
      nn.ReLU()
    )
    self.feature_data=nn.Sequential(
      nn.Linear(249, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, DATA_OUTPUT),
      nn.ReLU()
    )
    self.main=nn.Sequential(
      nn.Linear(CONV_OUTPUT + DATA_OUTPUT, 512),
      nn.ReLU(),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(), 
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, COMBINED_OUTPUT),
      nn.ReLU(),
      )
    self.actioner=nn.Sequential(nn.Linear(64, N_ACTIONS), nn.Softmax(dim=1))
    self.screen_mu=nn.Linear(N_ACTIONS + COMBINED_OUTPUT, 2)
    self.screen_sigma=nn.Sequential(
      nn.Linear(N_ACTIONS + COMBINED_OUTPUT, 2),
      nn.Softplus()
    )
    self.queued=nn.Linear(N_ACTIONS + COMBINED_OUTPUT, 2)
    self.ctrl_grp_act=nn.Linear(N_ACTIONS + COMBINED_OUTPUT, 5)
    self.ctrl_grp_id=nn.Linear(N_ACTIONS + COMBINED_OUTPUT, 10)
    self.select_point_act=nn.Linear(N_ACTIONS + COMBINED_OUTPUT, 4)

    #Hyperparameters
    self.buffer = deque(maxlen=100000)
    self.ent_coeff = 0.01
    self.γ = 0.99
    self.λ = 0.95
    self.writer = SummaryWriter("logs/A2C")
    self.unit_score = 0
    self.building_score = 0

  def forward(self, images, data):
    img_feature=self.feature_image(images)
    data_feature=self.feature_data(data)
    combined=torch.cat((img_feature, data_feature), dim=1)
    main_feature=self.main(combined)
    action=self.actioner(main_feature)
    screen_mu= self.screen_mu(torch.cat((action, main_feature), dim=1))
    screen_sigma= self.screen_sigma(torch.cat((action, main_feature), dim=1))
    dist=torch.distributions.Normal(screen_mu, screen_sigma)
    sample=dist.sample()
    x, y = sample[:, 0], sample[:, 1]
    x, y = torch.sigmoid(x) * (63), torch.sigmoid(y) * (63)
    screenx, screeny= torch.sigmoid(x) * (MAP_SIZE-1), torch.sigmoid(y) * (MAP_SIZE-1)
    result={
      "action":self.actioner(main_feature),
      "queued":self.queued(torch.cat((action, main_feature), dim=1)),
      "minimap":(int(y), int(x)),
      "screen":(int(screeny), int(screenx)),
      "control_group_act":torch.argmax(self.ctrl_grp_act(torch.cat((action, main_feature), dim=1))).item(),
      "control_group_id":torch.argmax(self.ctrl_grp_id(torch.cat((action, main_feature), dim=1))).item(),
      "select_point_act":torch.argmax(self.select_point_act(torch.cat((action, main_feature), dim=1))).item(),
      "features":main_feature
    }
    return result
  
  def step_optimizer(self, loss, retain_graph=False):
      self.optimizer.zero_grad()
      loss.backward(retain_graph=retain_graph)
      self.optimizer.step()
  
  def advantage(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0

    for t in reversed(range(T)):
        if dones[t]:  # No bootstrap if episode ends
            next_value = 0
        else:
            next_value = values[t + 1] if t + 1 < T else 0

        # Temporal Difference residual
        delta = rewards[t] + self.γ * next_value - values[t]

        # Recursive GAE formula
        gae = delta + self.γ * self.λ * gae
        advantages[t] = gae

    return advantages
  
  def infer(self):
      r, logit, obs, terminated, entropy = zip(*self.buffer)
      
      r = torch.stack(r)
      logit = torch.stack(logit)
      obs = torch.stack(obs)
      terminated = torch.stack(terminated)
      entropy = torch.stack(entropy)
  
      # Compute values using critic
      with torch.no_grad():  # Don't track gradients here
          val = self.critic(obs).detach()  # Detach to prevent gradient tracking
      
      # Compute advantages
      advantage = self.advantage(r, val, terminated)
      
      # Actor loss
      ent_loss = -entropy.mean() * self.ent_coeff
      actor_loss = -logit * advantage.detach() + ent_loss  # Detach advantage
      
      # Update actor
      self.optimizer.zero_grad()
      actor_loss.mean().backward(retain_graph=True)
      self.optimizer.step()
      
      # Recompute values for critic loss (with gradients)
      val = self.critic(obs)
      # Compute critic loss
      critic_loss = ((r - val.squeeze()) ** 2).mean()      
      # Update critic separately
      self.critic.optimizer.zero_grad()
      critic_loss.backward()
      self.critic.optimizer.step()
  
      return actor_loss.mean().to('cpu').detach().numpy(), critic_loss.to('cpu').detach().numpy(), ent_loss.mean().to('cpu').detach().numpy()
  
  def step(self, obs):
    super(ZergAgent, self).step(obs)
    feature_screen=obs.observation['feature_screen']
    player_data=torch.tensor(obs.observation['player'])
    ctrl_data=torch.tensor(obs.observation['control_groups']).flatten()
    multi_select=torch.tensor(obs.observation['multi_select'][:MAX_SELECT]) if len(obs.observation['multi_select']) != 0 else torch.zeros(MAX_SELECT)
    cargo_data=torch.tensor(obs.observation['cargo'][:MAX_CARGO]) if len(obs.observation['cargo']) != 0 else torch.zeros(MAX_CARGO)
    prod_queue_data=torch.tensor(obs.observation['production_queue'][:MAX_QUEUE]) if len(obs.observation['production_queue']) != 0 else torch.zeros(MAX_QUEUE)
    build_queue_data=torch.tensor(obs.observation['build_queue'][:MAX_QUEUE]) if len(obs.observation['build_queue']) != 0 else torch.zeros(MAX_QUEUE)

    reward=(obs.observation['score_cumulative'][5]-self.unit_score)+ (obs.observation['score_cumulative'][6]-self.building_score)
    self.unit_score=obs.observation['score_cumulative'][5]
    self.building_score = obs.observation['score_cumulative'][6]
    reward = reward + obs.reward*100

    done=obs.last()

    data=torch.cat((player_data, ctrl_data, multi_select, cargo_data, prod_queue_data, build_queue_data), dim=0)
    data=data.unsqueeze(0)
    feature_screen=torch.tensor(feature_screen, dtype=torch.float16).unsqueeze(0)

    result=self(feature_screen, data)
    
    args=[]
    
    dist=torch.distributions.Categorical(result['action'])
    action=dist.sample()
    entropy=dist.entropy()
    logit=dist.log_prob(action)
    self.buffer.append((torch.tensor(reward), logit, result['features'], torch.tensor(done), entropy))

    action=action.item()

    if done:
      self.infer()

    if int(actions.FUNCTIONS[action_ids[action]].id) in obs.observation['available_actions']:
      for i in actions.FUNCTIONS[action_ids[action]].args:
        try:
          val=result[i.name]
          if isinstance(val, int):
            val=[val,]
          args.append(val)
        except KeyError:
          print(i, action_ids[action])
          raise KeyError
      return actions.FunctionCall(action_ids[action], args)
    else:
      return actions.FunctionCall(NO_OP, [])

def main(unused_argv):
  agent = ZergAgent()
  try:
    env=sc2_env.SC2Env(
          map_name="AbyssalReef",
          players=[sc2_env.Agent(sc2_env.Race.zerg),
                   sc2_env.Bot(sc2_env.Race.random,
                               sc2_env.Difficulty.very_easy)],
          agent_interface_format=features.AgentInterfaceFormat(
              feature_dimensions=features.Dimensions(screen=MAP_SIZE, minimap=64)),
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
          print(step_actions)
        break

      
  except KeyboardInterrupt:
    pass
  
if __name__ == "__main__":
  app.run(main)