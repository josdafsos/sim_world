soil_types = {
    "dirt": {
        "color": (148, 85, 49, 255),
        "water absorption": 0.0001,
    },
    "rock": {
        "color": (128, 128, 128, 255),
        "water absorption": 0.00001,
    },
    "sand": {
        "color": (223, 197, 94, 255),
        "water absorption": 0.0005,
    }
}
WATER_MOISTURE_LEVEL_TO_RISE_HEIGHT = 0.4  # if moisture reaches this level, water level starts to rise
WATER_FLOW_RATE_TIME_CONSTANT = 0.5  # defines the speed at which water propagates into nearby tiles
WATER_MAX_SOURCE_OUTPUT = 0.05
WATER_MIN_SOURCE_OUTPUT = 0.001

world_generation_properties = {
    "generation probabilities": [
        ["water source", 0.005]
    ]

}


tile_format = {  # just a hint how an orbitrary tile might look like
    "type": [
        ["dirt", 10],
        ["rock", 20],
             ],
    "modifiers": [
        ["water source", 0.1],  # second parameter is water generation

    ]
}