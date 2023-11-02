import game.wrapped_flappy_bird as game
import keyboard

game_state = game.GameState()
while True:
    game_state.frame_step(True)
    if (keyboard.is_pressed("q")):
        break

