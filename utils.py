import torch

def get_gamestate_info(game_state):
    """
    gets coordinates of the two pipes
    usage:          pipe_info = get_pipes_info(game_state)
                    pipe_info["pipe0"]["upper"]["x"] = x coordinate of the upper pipe of the first pipe
    @args:          game_state
    @returns:         
        "pipe0": {
            "upper": {
                "x":
                "y": 
            },
            "lower": {
                "x": 
                "y": 
            }
        },
        "pipe1": {
            "upper": {
                "x": ,
                "y": 
            },
            "lower": {
                "x": 
                "y": 
            }
        }, 
        "player": {
            "x": 
            "y": 
            "VelY":
            "AccY": 
            "Flapped": 
        }
    """
    return {
        "pipe0": {
            "upper": {
                "x": game_state.upperPipes[0]['x'],
                "y": game_state.upperPipes[0]['y']
            },
            "lower": {
                "x": game_state.lowerPipes[0]['x'],
                "y": game_state.lowerPipes[0]['y']
            }
        },
        "pipe1": {
            "upper": {
                "x": game_state.upperPipes[1]['x'],
                "y": game_state.upperPipes[1]['y']
            },
            "lower": {
                "x": game_state.lowerPipes[1]['x'],
                "y": game_state.lowerPipes[1]['y']
            }
        },
        "player": {
            "x": game_state.playerx,
            "y": game_state.playery,
            "VelY": game_state.playerVelY,
            "AccY": game_state.playerAccY,
            "Flapped": game_state.playerFlapped,
        }
    }

def get_gamestate_info_tensor(game_state):
    """
    gets gamestate but returns it as a tensor. Use when feeding into ML algorithm
    Arguments: game_state
    Returns: tensor of shape (3, 4) containing same information as get_gamestate_info, but without the dictionary.
    """
    return torch.tensor([game_state.lowerPipes[0]['x'], game_state.lowerPipes[0]['y'], 
                         game_state.lowerPipes[1]['x'], game_state.lowerPipes[1]['y'], 
                        game_state.playery, game_state.playerVelY])