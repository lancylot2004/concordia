class CleanupGame:
    """
    A mixed-motive multi-agent game where seven players coexist on a 2D grid
    world. There exists a river and an orchard area. The orchard area spawns
    apples at some rate 'r', which is inversely proportional to the overall
    pollution level in the river area.

    In each turn, players can choose one action from the set
    {move_{wasd}, pick_up, clean_up_{wasd}, zap_{wasd}}.
    """
