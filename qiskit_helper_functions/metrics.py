import numpy as np

def chi2_distance(target,obs,normalize):
    obs = np.absolute(obs)
    if normalize:
        obs = obs / sum(obs)
        assert abs(sum(obs)-1)<1e-5
    if isinstance(target,np.ndarray):
        assert len(target)==len(obs)
        distance = 0
        for t, o in zip(target,obs):
            if abs(t-o)>1e-10:
                distance += np.power(t-o,2)/(t+o)
    elif isinstance(target,dict):
        distance = 0
        for o_idx, o in enumerate(obs):
            if o_idx in target:
                t = target[o_idx]
                if abs(t-o)>1e-10:
                    distance += np.power(t-o,2)/(t+o)
            else:
                distance += o
    else:
        raise Exception('Illegal target type:',type(target))
    return distance

def MSE(target,obs):
    if isinstance(target,np.ndarray):
        mse = (target-obs)**2
        mse = np.mean(mse)
    elif isinstance(target,dict):
        mse = 0
        for o_idx, o in enumerate(obs):
            if o_idx in target:
                t = target[o_idx]
                mse += (t-o)**2
            else:
                mse += o**2
        mse /= len(obs)
    else:
        raise Exception('target type : %s'%type(target))
    return mse

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

def cross_entropy(target,obs):
    obs = np.clip(obs,a_min=1e-16,a_max=None)
    if isinstance(target,np.ndarray):
        CE = np.sum(-target*np.log(obs))
        return CE
    elif isinstance(target,dict):
        CE = 0
        for t_idx in target:
            t = target[t_idx]
            CE -= -t*np.log(obs[t_idx])
        return CE