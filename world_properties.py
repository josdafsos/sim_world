from dataclasses import dataclass

# world constants
MEAT = 'meat'

COW_ID = 1
WOLF_ID = -1

@dataclass(frozen=True)
class WorldProperties:
    body_disappear_chance = 0.10  # probability that a lying body on a tile will disappear (only one, not all)


soil_types = {
    "dirt": {
        "color": (148, 85, 49, 255),
        "water absorption": 0.00015,
        "nutritional value": 1,
    },
    "rock": {
        "color": (128, 128, 128, 255),
        "water absorption": 4e-7,
        "nutritional value": 0.1,
    },
    "sand": {
        "color": (200, 170, 80, 255),
        "water absorption": 0.00075,
        "nutritional value": 0.1,
    }
}


world_generation_properties = {
    "generation probabilities": [
        ["water source", 0.0015],  # 0.0015  # TODO make exponential decay probability for value of the source
        ["grass", 0.3]
    ]

}


tile_format = {  # just a hint how an orbitrary tile might look like
    "type": {
        "dirt": 10,
        "rock": 20,
    },
    "modifiers": [
        ["water source", 0.1],  # second parameter is water generation

    ]
}

