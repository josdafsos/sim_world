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


world_generation_properties = {
    "generation probabilities": [
        ["water source", 0.0015]
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