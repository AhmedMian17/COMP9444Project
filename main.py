import game.wrapped_flappy_bird as game
import keyboard

from utils import get_gamestate_info
from utils import get_input_layer

# from Models.NEATModel import NEATModel

# neatModel = NEATModel()
# neatModel.run(100)


# game_state = game.GameState()
# while True:
#     if keyboard.is_pressed(" "):
#         _, reward, _ = game_state.frame_step(True)
#         print(reward)
#         # currently 0.1 reward by default, 1 reward if bird passes through pipe, -1 reward if bird hits pipe or ground
#         # keyboard seems to press jump multiple times but whatever.
#     else:
#         _, reward, _ = game_state.frame_step(False)
#     if keyboard.is_pressed("q"):
#         break
#     pipes_obj = get_gamestate_info_tensor(game_state)

from Models.DQL.agent import test

test(game.GameState())