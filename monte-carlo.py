from copy import deepcopy
from random import gauss, uniform

from numpy.linalg import norm
from numpy import subtract, linspace
from scipy.optimize import minimize

def min_func(q, data, y0, space, param_keys, parser):
    """
        Functional for minimization problems

        Keyword arguments:
        q           -- array of parameter values from problem solution on i-th iteration
        data        -- array of inverse data with which we will be comparing solution
        y0          -- initial data
        space       -- time points in which we will solve ODE with solution parameters 'q'
        param_keys  -- array of corresponding to 'q' parameter names
        parser      -- ODE from text parser
    """
    res = 0.0
    params_dict = get_dict(param_keys, q)
    sol = odeint(model_eval, y0, space, args=(params_dict, parser))
    step = (len(space) - 1) / float(space[-1])
    for  points, idx in zip(data['data'], range(len(data['data']))):
        for time, point in zip(data['t'], points):
            res += (point - sol[int(time * step), idx]) ** 2
    return res

def get_solution_subspace(sol, count, space):
    if sol is None or len(sol) == 0:
        raise Exception('Error while trying generate synthetic data', 'No direct problem solution')
    subspace = []
    data = []
    for i in range(len(sol[0])):
        data.append([])
    step = (len(sol) - 1) // count
    for i in range(count):
        point = step * (i + 1)
        for j in range(len(sol[0])):
            data[j].append(sol[point][j])
        subspace.append(space[point])
    return {'t': subspace, 'data': data}

def get_synth_data(y0, params, points_count, space, parser):
    sol = odeint(model_eval, y0, space, args=(params, parser))
    return get_solution_subspace(sol, points_count, space)

def get_random(rand_type='uniform'):
    """Return random value of given distribution.

    Args:
        rand_type: 'uniform' or 'gauss' distribution

    Returns: double
    """
    if rand_type == 'uniform':
        return uniform(-1, 1)
    elif rand_type == 'gauss':
        return gauss(0, 1)

def add_error(value, percent, rand_type='uniform'):
    """Add noise with given percentage and distribution to value and return result"""
    return value + value * (percent / 100) * get_random(rand_type)

def add_arr_error(arr, percent, rand_type='uniform'):
    """Add noise to array of doubles"""
    return [add_error(x, percent, rand_type) for x in arr]

def add_data_noise(data, percent, rand_type='uniform'):
    """
        Add noise to the inverse data and return result of it.

        Args:
        data -- tuple of type {'t': double[], 'data': double[][]}
    """
    new_data = deepcopy(data)
    new_data['data'] = [add_arr_error(x, percent, rand_type) for x in new_data['data']]
    return new_data

def get_rel_error(init_val, modified_val):
    """Get relative error of two vectors"""
    return norm(subtract(init_val, modified_val)) / norm(init_val)

def get_avg_estimation_error(model, inverse_data):
    """Find average relative estimation error of model"""
    parser          = ModelParser(model['model'])
    params_dict     = model['parameters']
    y_0             = model['initialValues']
    options         = model['options']
    batches_num     = 10                # TODO: options['dataOptions']['batchesNumber']
    method          = 'nelder-mead'     # TODO: options['dataOptions']['inverseMethod']
    percent         = 5                 # TODO: options['dataOptions']['percent']
    space           = get_space(options)
    method_options  = {'xtol': 1e-8}

    error = 0
    param_values = list(params_dict.values())
    for _ in range(batches_num):
        result = minimize(min_func, param_values, method=method,
                          args=(add_data_noise(inverse_data, percent), y_0,
                                space, params_dict.keys(), parser),
                          options=method_options)
        error += get_rel_error(param_values, result.x.tolist())
    return error * 100 / batches_num

def get_space(options):
    """
        Get discrete linear space from given interval and number of points
    """
    return linspace(0, options['interval'], options['points'])

def solve_monte_carlo(model):
    """
        Identifiability analysis of model by monte carlo simulation method
    """
    parser = ModelParser(model['model'])
    options = model['options']

    y_0 = model['initialValues']
    params_dict = model['parameters']

    space = get_space(options)
    synth_data_points_count = 10 # TODO: options['dataOptions']['dataPoints']
    inverse_data = get_synth_data(y_0, params_dict, synth_data_points_count, space, parser)

    return {'solution': get_avg_estimation_error(model, inverse_data)}
