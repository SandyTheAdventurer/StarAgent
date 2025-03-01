from pysc2.agents import base_agent
from pysc2.lib import actions, features
from pysc2.env import sc2_env
from absl import app

class AgentZero(base_agent.BaseAgent):
    def step(self, obs):
        super(AgentZero, self).step(obs)
        return actions.FUNCTIONS.no_op()

def main(args):
    agent = AgentZero()
    try:
        env= sc2_env.SC2Env(
            map_name="Simple128",
            players=[sc2_env.Agent(sc2_env.Race.zerg),
                    sc2_env.Bot(sc2_env.Race.random,
                    sc2_env.Difficulty.very_easy)],
            agent_interface_format=features.AgentInterfaceFormat(feature_dimensions=features.Dimensions(screen=84, minimap=64)),
            step_mul=16, game_steps_per_episode=0, visualize=True)
        
        agent.setup(env.observation_spec(), env.action_spec())
        
        timesteps = env.reset()
        agent.reset()
        
        while True:
          step_actions = [agent.step(timesteps[0])]
          if timesteps[0].last():
            break
          timesteps = env.step(step_actions)
      
    except KeyboardInterrupt:
        pass
  
if __name__ == "__main__":
  app.run(main)