import game.wrapped_flappy_bird as game
import keyboard

from Models.NEATModel import NEATModel

neatModel = NEATModel()
# neatModel.run(300, "neat-checkpoint-27")
neatModel.loadBest()
neatModel.playGame()
# neatModel.testBest(1000)

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

