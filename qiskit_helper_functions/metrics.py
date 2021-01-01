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
    if isinstance(target,dict):
        mse = 0
        for t_idx in target:
            t = target[t_idx]
            o = obs[t_idx]
            mse += (t-o)**2
        mse /= len(obs)
    elif isinstance(target,np.ndarray) and isinstance(obs,np.ndarray):
        target = target.reshape(-1,1)
        obs = obs.reshape(-1,1)
        mse = (target-obs)**2
        mse = np.mean(mse)
    elif isinstance(target,np.ndarray) and isinstance(obs,dict):
        mse = 0
        for o_idx in obs:
            o = obs[o_idx]
            t = target[o_idx]
            mse += (t-o)**2
        mse /= len(obs)
    else:
        raise Exception('target type : %s'%type(target))
    return mse

def fidelity(target,obs):
    if isinstance(target,np.ndarray):
        assert len(target)==len(obs)
        fidelity = 0
        for t,o in zip(target,obs):
            if t > 1e-16:
                fidelity += o
    elif isinstance(target,dict):
        fidelity = 0
        for t_idx in target:
            t = target[t_idx]
            o = obs[t_idx]
            if t > 1e-16:
                fidelity += o
    else:
        raise Exception('target type : %s'%type(target))
    return fidelity

def cross_entropy(target,obs):
    if isinstance(target,dict):
        CE = 0
        for t_idx in target:
            t = target[t_idx]
            o = obs[t_idx]
            o = o if o>1e-16 else 1e-16
            CE += -t*np.log(o)
        return CE
    elif isinstance(target,np.ndarray) and isinstance(obs,np.ndarray):
        obs = np.clip(obs,a_min=1e-16,a_max=None)
        CE = np.sum(-target*np.log(obs))
        return CE
    elif isinstance(target,np.ndarray) and isinstance(obs,dict):
        CE = 0
        for o_idx in obs:
            o = obs[o_idx]
            t = target[o_idx]
            o = o if o>1e-16 else 1e-16
            CE += -t*np.log(o)
        return CE
    else:
        raise Exception('target type : %s, obs type : %s'%(type(target),type(obs)))

def relative_entropy(target,obs):
    return cross_entropy(target=target,obs=obs) - cross_entropy(target=target,obs=target)