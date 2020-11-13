import numpy as np

def chi2_distance(target,obs,normalize):
    obs = np.absolute(obs)
    if normalize:
        obs = obs / sum(obs)
        assert abs(sum(obs)-1)<1e-5
    if isinstance(target,np.array()):
        assert len(target)==len(obs)
        distance = 0
        for t, o in zip(target,obs):
            if abs(t-o)<1e-16:
                distance += 0
            else:
                distance += np.power(t-o,2)/(t+o)
    elif isinstance(target,dict):
        distance = 0
        for o_idx, o in enumerate(obs):
            if o_idx in target:
                t = target[o_idx]
            else:
                t = 0
            if abs(t-o)<1e-16:
                distance += 0
            else:
                distance += np.power(t-o,2)/(t+o)
    else:
        raise Exception('Illegal target type:',type(target))
    return distance

def fidelity(target,obs):
    assert len(target)==len(obs)
    epsilon = 1e-20
    obs = np.absolute(obs)
    obs = obs / sum(obs)
    # assert abs(sum(obs)-1)<1e-5
    fidelity = 0
    for t,o in zip(target,obs):
        if t > 1e-16:
            fidelity += o
    return fidelity