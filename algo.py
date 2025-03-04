from pysc2.agents import base_agent
from pysc2.lib import actions

import torch
import torch.nn as nn
from torch import tensor

SELECT_POINT=actions.FUNCTIONS.select_point.id
SELECT_RECT=actions.FUNCTIONS.select_rect.id
SELECT_ARMY=actions.FUNCTIONS.select_army.id
MOVE_CAMERA=actions.FUNCTIONS.move_camera.id
ATTACK_MINIMAP=actions.FUNCTIONS.Attack_minimap.id
TRAIN_DRONE_QUICK=actions.FUNCTIONS.Train_Drone_quick.id
BUILD_SPAWNPOOL=actions.FUNCTIONS.Build_SpawningPool_screen.id
TRAIN_ZERGLING=actions.FUNCTIONS.Train_Zergling_quick.id
TRAIN_OVERLORD=actions.FUNCTIONS.Train_Overlord_quick.id
NO_OP=actions.FUNCTIONS.no_op.id

class AgentZero(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.stack=torch.nn.Sequential(
            nn.Conv2d()
        )
    def step(self, obs):
        super(AgentZero, self).step(obs)
        return actions.FUNCTIONS.no_op()