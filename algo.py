from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
import torch.nn as nn
import torch
from absl import app
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from collections import deque
from torchvision.models import resnet18 as resnet, ResNet18_Weights as weights
import numpy as np

#Precision
P_CRITIC=torch.float32
P_ACTOR=torch.float16

torch.set_default_dtype(P_ACTOR)
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
CONV_OUTPUT=512
DATA_OUTPUT=64
COMBINED_OUTPUT=128

N_ACTIONS=len(action_ids)

MAP_SIZE=84
MAX_SELECT=200
MAX_CARGO=8
MAX_QUEUE=5

class Critic(nn.Module):
  def __init__(self, size):
    super(Critic, self).__init__()
    self.main=nn.Sequential(
      nn.Linear(size, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(), 
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 1),
    ).to(P_CRITIC)
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

    self.critic=Critic(COMBINED_OUTPUT)

    self.args_critic=Critic(N_ACTIONS + COMBINED_OUTPUT)

    self.optimizer=torch.optim.Adam(self.parameters(), lr=0.001)

    self.feature_image=resnet(weights=weights.DEFAULT)
    self.feature_image.conv1 = nn.Conv2d(27, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.feature_image=nn.Sequential(*list( self.feature_image.children())[:-1])
    self.feature_data=nn.Sequential(
      nn.Linear(249, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, DATA_OUTPUT),
      nn.ReLU()
    )
    self.main=nn.Sequential(
      nn.Linear(CONV_OUTPUT + DATA_OUTPUT, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(), 
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, COMBINED_OUTPUT),
      nn.ReLU(),
      )
    self.actioner=nn.Sequential(nn.Linear(COMBINED_OUTPUT, 128),
                                nn.Linear(128, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, N_ACTIONS),
                                nn.Softmax(dim=1))
    self.screen_mu=nn.Sequential(
      nn.Linear(N_ACTIONS + COMBINED_OUTPUT, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 2)
    )
    self.screen_sigma=nn.Sequential(
      nn.Linear(N_ACTIONS + COMBINED_OUTPUT, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 2),
      nn.Softplus()
    )
    self.queued=nn.Sequential(
      nn.Linear(N_ACTIONS + COMBINED_OUTPUT, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 2),
      nn.Softmax(dim=1)
    )
    self.ctrl_grp_act=nn.Sequential(
      nn.Linear(N_ACTIONS + COMBINED_OUTPUT, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 5),
      nn.Softmax(dim=1)
    )
    self.ctrl_grp_id=nn.Sequential(
      nn.Linear(N_ACTIONS + COMBINED_OUTPUT, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 10),
      nn.Softmax(dim=1)
    )
    
    self.select_point_act=nn.Sequential(
      nn.Linear(N_ACTIONS + COMBINED_OUTPUT, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, 4),
      nn.Softmax(dim=1)
    )

    #Hyperparameters
    self.buffer = deque(maxlen=10000)
    self.buffers = {
    "queued" : deque(maxlen=10000),
    "screen" : deque(maxlen=10000),
    "minimap" : deque(maxlen=10000),
    "control_group_act" : deque(maxlen=10000),
    "control_group_id" : deque(maxlen=10000),
    "select_point_act" : deque(maxlen=10000)
    }
    self.ent_coeff = 0.01
    self.γ = 0.99
    self.λ = 0.95
    self.writer = SummaryWriter("logs/A2C")
    self.unit_score = 0
    self.building_score = 0
    self.eval()

  def forward(self, images, data):
    img_feature=self.feature_image(images)
    img_feature=torch.flatten(img_feature, 1)
    data_feature=self.feature_data(data)
    combined=torch.cat((img_feature, data_feature), dim=1)
    main_feature=self.main(combined)
    actions=self.actioner(main_feature)
    screen_mu= self.screen_mu(torch.cat((actions, main_feature), dim=1)).float()
    screen_sigma= self.screen_sigma(torch.cat((actions, main_feature), dim=1)).float()

    screen_dist=torch.distributions.Normal(screen_mu, screen_sigma)
    sample=screen_dist.sample()
    screen_logit=screen_dist.log_prob(sample)
    screen_entropy=screen_dist.entropy()
    x, y = sample[:, 0], sample[:, 1]
    x, y = torch.sigmoid(x) * (63), torch.sigmoid(y) * (63)
    screenx, screeny= torch.sigmoid(x) * (MAP_SIZE-1), torch.sigmoid(y) * (MAP_SIZE-1)

    action_dist=torch.distributions.Categorical(actions)
    action=action_dist.sample()
    action_logit=action_dist.log_prob(action)
    action_entropy=action_dist.entropy()

    queued_dist=self.queued(torch.cat((actions, main_feature), dim=1)).float()
    queued_dist=torch.distributions.Categorical(queued_dist)
    queued=queued_dist.sample()
    queued_logit=queued_dist.log_prob(queued)
    queued_entropy=queued_dist.entropy()

    ctrl_act=self.ctrl_grp_act(torch.cat((actions, main_feature), dim=1)).float()
    ctrl_act_dist=torch.distributions.Categorical(ctrl_act)
    ctrl_act=ctrl_act_dist.sample()
    ctrl_act_logit=ctrl_act_dist.log_prob(ctrl_act)
    ctrl_act_entropy=ctrl_act_dist.entropy()

    ctrl_id=self.ctrl_grp_id(torch.cat((actions, main_feature), dim=1)).float()
    ctrl_id_dist=torch.distributions.Categorical(ctrl_id)
    ctrl_id=ctrl_id_dist.sample()
    ctrl_id_logit=ctrl_id_dist.log_prob(ctrl_id)
    ctrl_id_entropy=ctrl_id_dist.entropy()

    select_point_act=self.select_point_act(torch.cat((actions, main_feature), dim=1)).float()
    select_point_act_dist=torch.distributions.Categorical(select_point_act)
    select_point_act=select_point_act_dist.sample()
    select_point_act_logit=select_point_act_dist.log_prob(select_point_act)
    select_point_act_entropy=select_point_act_dist.entropy()

    result={
      "action":action.item(),
      "queued":queued.item(),
      "minimap":(int(y), int(x)),
      "screen":(int(screeny), int(screenx)),
      "control_group_act":ctrl_act.item(),
      "control_group_id":ctrl_id.item(),
      "select_point_act":select_point_act.item(),
      "features":(main_feature.squeeze(0), torch.cat((actions, main_feature), dim=1).squeeze(0)),
      "logits":{"action" : (action_logit, action_entropy), "queued": (queued_logit, queued_entropy),"screen" : (screen_logit, screen_entropy), "minimap" : (screen_logit, screen_entropy), "control_group_act" : (ctrl_act_logit, ctrl_act_entropy), "control_group_id" : (ctrl_id_logit, ctrl_id_entropy), "select_point_act" : (select_point_act_logit, select_point_act_entropy)}
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
  
  def infer(self, buffer=None, critic=None):
      self.train()
      if buffer is None:
          buffer = self.buffer
      if critic is None:
          critic = self.args_critic

      r, logit, obs, terminated, entropy = zip(*buffer)
      
      r = torch.stack(r)
      logit = torch.stack(logit)
      obs = torch.stack(obs)
      terminated = torch.stack(terminated)
      entropy = torch.stack(entropy)
      
      obs = DataLoader(obs, batch_size=64)
      val = []
      with autocast("cuda"):
        with torch.no_grad():
            for i in obs:
                val.append(critic(i.to("cuda")))
      val = torch.cat(val, dim=0)

      # Compute advantages
      advantage = self.advantage(r, val, terminated)
      advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
      
      assert not torch.isnan(val).any(), "Critic returned NaN values!"
      assert not torch.isnan(advantage).any(), "Advantage has NaN!"
      
      # Actor loss
      ent_loss = (-entropy.mean() * self.ent_coeff)
      actor_loss = (-logit.squeeze(0) * advantage.reshape(-1,1).detach() + ent_loss)
      
      # Update actor
      self.optimizer.zero_grad()
      actor_loss.mean().backward(retain_graph=True)
      self.optimizer.step()
      
      val = []
      with autocast("cuda"):
        for i in obs:
            val.append(critic(i))
      val = torch.cat(val, dim=0)
      critic_loss = ((r - val.squeeze()) ** 2).mean()      
      # Update critic separately
      critic.optimizer.zero_grad()
      critic_loss.backward(retain_graph=True)
      critic.optimizer.step()

      self.eval()

      torch.cuda.empty_cache()
  
      return actor_loss.mean().to('cpu').detach().numpy(), critic_loss.to('cpu').detach().numpy(), ent_loss.mean().to('cpu').detach().numpy()
  
  def step(self, obs, iter):
    super(ZergAgent, self).step(obs)
    feature_screen=obs.observation['feature_screen']
    player_data=torch.tensor(obs.observation['player'])
    ctrl_data=torch.tensor(obs.observation['control_groups']).flatten()
    multi_select=torch.tensor(obs.observation['multi_select'][:MAX_SELECT]) if len(obs.observation['multi_select']) != 0 else torch.zeros(MAX_SELECT)
    cargo_data=torch.tensor(obs.observation['cargo'][:MAX_CARGO]) if len(obs.observation['cargo']) != 0 else torch.zeros(MAX_CARGO)
    prod_queue_data=torch.tensor(obs.observation['production_queue'][:MAX_QUEUE]) if len(obs.observation['production_queue']) != 0 else torch.zeros(MAX_QUEUE)
    build_queue_data=torch.tensor(obs.observation['build_queue'][:MAX_QUEUE]) if len(obs.observation['build_queue']) != 0 else torch.zeros(MAX_QUEUE)

    killed_now=obs.observation['score_cumulative'][5]-self.unit_score
    razed_now=obs.observation['score_cumulative'][6]-self.building_score
    reward=(razed_now) + (killed_now)
    reward += 1.5 * np.sum(obs.observation['score_by_vital'][0]) - np.sum(obs.observation['score_by_vital'][1]) + 1.5 * np.sum(obs.observation['score_by_vital'][2])
    if reward == 0:
      reward-=15
    self.unit_score=obs.observation['score_cumulative'][5]
    self.building_score = obs.observation['score_cumulative'][6]
    reward = reward + obs.reward*100

    reward = torch.tensor(reward)

    done=torch.tensor(obs.last())

    data=torch.cat((player_data, ctrl_data, multi_select, cargo_data, prod_queue_data, build_queue_data), dim=0)
    data=data.unsqueeze(0)
    feature_screen=torch.tensor(feature_screen, dtype=P_ACTOR).unsqueeze(0)

    result=self(feature_screen, data)

    action=result['action']
    
    args=[]
    args_losses=[]

    self.buffer.append((reward, result['logits']['action'][0], result['features'][0], done, result['logits']['action'][1]))

    if done:
      print(reward)
      main_losses=self.infer(critic=self.critic)
      for i in self.buffers.values():
        if len(i) != 0:
          key = next((k for k, v in self.buffers.items() if v == i), None)
          args_losses.append((key, self.infer(i)))
          i.clear()      
      self.buffer.clear()
      self.writer.add_scalar("Losses/actioner/actor_loss", main_losses[0], iter)
      self.writer.add_scalar("Losses/actioner/critic_loss", main_losses[1], iter)
      self.writer.add_scalar("Losses/actioner/entropy_loss", main_losses[2], iter)
      for key, losses in args_losses:
        self.writer.add_scalar(f"Losses/{key}/actor_loss", losses[0], iter)
        self.writer.add_scalar(f"Losses/{key}/critic_loss", losses[1], iter)
        self.writer.add_scalar(f"Losses/{key}/entropy_loss", losses[2], iter)
      self.writer.add_scalar("Rewards/total_reward", reward, iter)
      self.writer.add_scalar("Rewards/damage_recieved", sum(obs.observation['score_by_vital'][1]), iter)
      self.writer.add_scalar("Rewards/damage_dealt", sum(obs.observation['score_by_vital'][0]), iter)
      self.writer.add_scalar("Rewards/units_killed", killed_now, iter)
      self.writer.add_scalar("Rewards/buildings_killed", razed_now, iter)
      self.writer.add_scalar("Rewards/damage_healed", sum(obs.observation['score_by_vital'][2]), iter)

    if int(actions.FUNCTIONS[action_ids[action]].id) in obs.observation['available_actions']:
      for i in actions.FUNCTIONS[action_ids[action]].args:
        try:
          val=result[i.name]
          if isinstance(val, int):
            val=[val,]
          self.buffers[i.name].append((reward, result['logits'][i.name][0], result['features'][1], done, result['logits'][i.name][1]))
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
          step_actions = [agent.step(timesteps[0], env._episode_count)]
          if timesteps[0].last():
            break
          timesteps = env.step(step_actions)
      
  except KeyboardInterrupt:
    pass
  
if __name__ == "__main__":
  app.run(main)