import game.wrapped_flappy_bird as game
import keyboard

from utils import get_gamestate_info
from utils import get_gamestate_info_tensor

game_state = game.GameState()
while True:
    if (keyboard.is_pressed(" ")):
        _, reward, _ = game_state.frame_step(True)
        print(reward)
    else:
        _, reward, _ = game_state.frame_step(False)
    if (keyboard.is_pressed("q")):
        break
    pipes_obj = get_gamestate_info_tensor(game_state)

