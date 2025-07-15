import pygame
import time

def world_step(world, action, key_idx_list):
    if time.time() - action["last call time"] > 0.250 / (action["steps in row"] + 1):
        if world.autoplay:
            world.autoplay = False
            print("world autoplay disabled")
        else:
            if action["steps in row"] < 50:
                world.step()
            else:
                print("making steps")
                world.multiple_steps(action["steps in row"])
                world.step()
        action["last call time"] = time.time()
        action["steps in row"] = min(action["steps in row"] + 1, 100)

def world_autoplay(world, action, pressed_key_idx_list):
    action["last call time"] = time.time()
    if time.time() - action["last release time"] < 0.300:
        world.autoplay = True
        print("world autoplay enabled")
    action["steps in row"] = min(action["steps in row"] + 1, 30)

def world_autoplay_single_key_press(world, action, pressed_key_idx_list):
    if time.time() - action["last call time"] > 0.500 and not world.autoplay:
        world.autoplay = True
        print("world autoplay enabled")
        action["steps in row"] = min(action["steps in row"] + 1, 2)
        action["last call time"] = time.time()


def camera_move(world, action, pressed_key_idx_list):
    if time.time() - action["last call time"] > 0.150 / (action["steps in row"] + 1):
        for action_idx in pressed_key_idx_list:
            world.camera_move(action["extra"][action_idx])
        action["last call time"] = time.time()
        action["steps in row"] = min(action["steps in row"] + 1, 30)

def camera_fit(world, action, pressed_key_idx_list):
    action["last call time"] = time.time()
    if time.time() - action["last release time"] < 0.500:
        world.camera_fit_view()
    action["steps in row"] = min(action["steps in row"] + 1, 30)

def switch_drawing_enabled(world, action, pressed_key_idx_list):
    if time.time() - action["last call time"] > 0.500:
        world.enable_visualization = not world.enable_visualization
        action["last call time"] = time.time()
        action["steps in row"] = min(action["steps in row"] + 1, 2)
        if world.enable_visualization:
            print("Visualization enabled")
        else:
            print("Visualization disabled")

keys_list = [
    {
        "keys": [pygame.K_SPACE],
        "action": world_step,
        "last call time": time.time(),
        "last release time": time.time() - 5,
        "steps in row": 0,
    },
    {
        "keys": [pygame.K_SPACE],
        "action": world_autoplay,
        "last call time": time.time(),
        "last release time": time.time() - 5,
        "steps in row": 0,
    },
    {
        "keys": [pygame.K_p],
        "action": world_autoplay_single_key_press,
        "last call time": time.time(),
        "last release time": time.time() - 5,
        "steps in row": 0,
    },
    {
        "keys": [pygame.K_w,
                 pygame.K_s,
                 pygame.K_a,
                 pygame.K_d,
                 pygame.K_x,
                 pygame.K_z,],
        "action": camera_move,
        "last call time": time.time(),
        "steps in row": 0,
        "last release time": time.time() - 5,
        "extra": ["up",
                  "down",
                  "left",
                  "right",
                  "zoom in",
                  "zoom out",
        ]
    },
    {
        "keys": [pygame.K_z],
        "action": camera_fit,
        "last call time": time.time(),
        "last release time": time.time() - 5,
        "steps in row": 0,
    },
    {
        "keys": [pygame.K_c],
        "action": switch_drawing_enabled,
        "last call time": time.time(),
        "last release time": time.time() - 5,
        "steps in row": 0,
    },
]

def parse_actions(world):
    keys = pygame.key.get_pressed()
    for action in keys_list:
        pressed_key_idx_list = []
        for idx, single_key in enumerate(action["keys"]):
            if keys[single_key]:
                pressed_key_idx_list.append(idx)
        if not pressed_key_idx_list:
            if action["steps in row"] > 0:
                action["last release time"] = time.time()
                action["steps in row"] = 0
        else:
            action["action"](world, action, pressed_key_idx_list)









