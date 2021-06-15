import numpy as np
from sklearn.linear_model import LinearRegression

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
    '''
    Mean Square Error
    '''
    if isinstance(target,dict):
        se = 0
        for t_idx in target:
            t = target[t_idx]
            o = obs[t_idx]
            se += (t-o)**2
        mse = se/len(obs)
    elif isinstance(target,np.ndarray) and isinstance(obs,np.ndarray):
        target = target.reshape(-1,1)
        obs = obs.reshape(-1,1)
        squared_diff = (target-obs)**2
        se = np.sum(squared_diff)
        mse = np.mean(squared_diff)
    elif isinstance(target,np.ndarray) and isinstance(obs,dict):
        se = 0
        for o_idx in obs:
            o = obs[o_idx]
            t = target[o_idx]
            se += (t-o)**2
        mse = se/len(obs)
    else:
        raise Exception('target type : %s'%type(target))
    return mse, se

def MAPE(target,obs):
    '''
    Mean absolute percentage error
    abs(target-obs)/target
    '''
    if isinstance(target,dict):
        mape = 0
        for t_idx in target:
            t = target[t_idx]
            o = obs[t_idx]
            mape += abs((t-o)/t)
        mape /= len(obs)
    elif isinstance(target,np.ndarray) and isinstance(obs,np.ndarray):
        target = target.reshape(-1,1)
        obs = obs.reshape(-1,1)
        mape = np.abs((target-obs)/target)
        mape = np.mean(mape)
    elif isinstance(target,np.ndarray) and isinstance(obs,dict):
        mape = 0
        for o_idx in obs:
            o = obs[o_idx]
            t = target[o_idx]
            mape += abs((t-o)/t)
        mape /= len(obs)
    else:
        raise Exception('target type : %s'%type(target))
    return mape*100

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

def correlation(target,obs):
    '''
    Measure the linear correlation between `target` and `obs`
    '''
    target = target.reshape(-1, 1)
    obs = obs.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(X=obs,y=target)
    score = reg.score(X=obs,y=target)
    return score

def nearest_probability_distribution(quasiprobability):
    '''Takes a quasiprobability distribution and maps
    it to the closest probability distribution as defined by
    the L2-norm.
    Parameters:
        return_distance (bool): Return the L2 distance between distributions.
    Returns:
        ProbDistribution: Nearest probability distribution.
        float: Euclidean (L2) distance of distributions.
    Notes:
        Method from Smolin et al., Phys. Rev. Lett. 108, 070502 (2012).
    '''
    sorted_probs, states = zip(*sorted(zip(quasiprobability, range(len(quasiprobability)))))
    num_elems = len(sorted_probs)
    new_probs = np.zeros(num_elems)
    beta = 0
    diff = 0
    for state, prob in zip(states,sorted_probs):
        temp = prob + beta / num_elems
        if temp < 0:
            beta += prob
            num_elems -= 1
            diff += prob * prob
        else:
            diff += (beta / num_elems) * (beta / num_elems)
            new_probs[state] = prob + beta / num_elems
    return new_probs, np.sqrt(diff)