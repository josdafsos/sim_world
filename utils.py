agents_folder = 'agents'


# --- vegetation setting ---

# used to map plant type into position in vegetation array of each tile. Plant group is used to define position
plant_group_index_mapping_dict = {
"base class group": -1,
"grass": 0,
}

# maps certain
plant_type_group_mapping_dict = {
    "grass": "grass",
    "cactus": 'grass',
}

VEGETATION_GROUPS = ['grass', 'tree']  # list of all usable vegetation groups

# list containing None for each vegetation type. To be used in initialization of tile's vegetation list
VEGETATION_LIST = [ None for _ in range(len(VEGETATION_GROUPS))]