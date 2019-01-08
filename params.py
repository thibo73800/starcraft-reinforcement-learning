

class PooParams(object):
    cpu = 4 # Number of fork (cpu) to use
    gamma = 0.99 # Gamma
    epochs = 50 # Number of epoch to run
    seed = 42 # Random seed
    exp_name = "PooExperience" # Default name of the experience
    clip_ratio = 0.2 # POO clip ratio
    pi_lr = 0.0001 # pi learning rate
    vf_lr = 0.001 # Value learning rate
    target_kl = 0.01
    log_target_kl = target_kl * 1.5

class BeaconParams(PooParams):
    # Size of the map and the minimap
    map_size = (64,64)
    minimap_size = 64
    action_space = 64*64 # then convert to x, y

    pi_lr = 0.0001 # pi learning rate
    vf_lr = 0.0001 # Value learning rate

    cpu = 1
    epochs = 50
    exp_name = "BeaconEnv"
