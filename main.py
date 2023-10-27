import game.wrapped_flappy_bird as game

game_state = game.GameState()
while True:
    game_state.frame_step(True)

