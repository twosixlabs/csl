import time

import numpy as np
import sympy as sym
from functools import reduce

def L1_norm(terms):
    return reduce(lambda x, y: x + y, terms, 0)

def L2_norm(terms):
    sq_terms = [t**2 for t in terms]
    summation = reduce(lambda x, y: x + y, sq_terms, 0)
    return sym.sqrt(summation)

def dot_product(xs, ys):
    return L1_norm([x * y for x, y in zip(xs, ys)])

def build_weights(layer_num, input_size, width, all_weights):
    weights = []
    for i in range(width):
        w = [sym.Symbol(f'w{layer_num}_{i}_{j}') for j in range(input_size)]
        weights.append(w)
        all_weights.extend(w)
    return weights

def run_layer(inputs, weights):
    return [dot_product(inputs, w_row) for w_row in weights]

class Relu(sym.Function):
    def fdiff(self, x):
        if x > 0:
            return 1
        else:
            return 0

def run_relu(inputs):
    return [Relu(x) for x in inputs]

sym.init_printing(use_unicode=True)

print('constructing')
start_time = time.process_time()

# 2 features
inputs = [sym.Symbol(f'x_{n}') for n in range(2)]

# 3 layers, one hidden layer
# generate the weights
all_weights = []
layer1_weights = build_weights(1, 2, 2, all_weights)
layer2_weights = build_weights(2, 2, 2, all_weights)
layer3_weights = build_weights(3, 2, 1, all_weights)

# run the network
layer1_output = run_layer(inputs, layer1_weights)
layer1_relu   = run_relu (layer1_output)
layer2_output = run_layer(layer1_relu, layer2_weights)
layer2_relu   = run_relu (layer2_output)
layer3_output = run_layer(layer2_relu, layer3_weights)

# calculate the loss
label = sym.Symbol('y')
pred_exp = L1_norm(layer3_output)
loss = (pred_exp - label)**2

# calculate immediate sensitivity
inner_gradient = [sym.diff(loss, w) for w in all_weights]
inner_norm = L1_norm(inner_gradient)
outer_gradient = [sym.diff(inner_norm, x) for x in inputs]
outer_norm = L1_norm(outer_gradient)
immediate_sensitivity = outer_norm

print('done constructing, time:', time.process_time() - start_time)

print('substituting')
start_time = time.process_time()

# substitute in actual values for the weights
weight_vals = [1.0 for w in all_weights]
subst = immediate_sensitivity
for w_name, w_val in zip(all_weights, weight_vals):
    subst = subst.subs(w_name, w_val)

subst = subst.subs(label, 1)

ivl = sym.Interval(-1,1)

# print(sym.log(subst))
# print(sym.srepr(subst))

print('done substituting, time:', time.process_time() - start_time)


def log(m):
    #pass
    print(m)

# recursive analysis for sympy ASTs
def analyze(e):
    if e.func in [sym.Float, sym.Integer,
                  sym.numbers.NegativeOne,
                  sym.numbers.Half
    ]:
        log('found constant')

    elif e.func == sym.Symbol:
        log('found symbol ' + str(e))

    elif e.func == sym.Add:
        log('found add')
        [analyze(a) for a in e.args]

    elif e.func == sym.Mul:
        log('found mul')
        [analyze(a) for a in e.args]

    elif e.func == sym.Pow:
        log('found pow')
        [analyze(a) for a in e.args]

    elif e.func == Relu:
        log('found relu')
        [analyze(a) for a in e.args]

    else:
        log('found unknown type' + str(e.func))

print('analyzing')
start_time = time.process_time()

analyze(subst)
#analyze(sym.log(subst))
print('done analyzing, time:', time.process_time() - start_time)

def add_envs(dict1, dict2):
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = value + dict1[key]
    return dict3

def scale_env(n, dict1):
    return {k : n * v for k, v in dict1.items()}

def sens(e, i_env):
    if e.func in [sym.Float, sym.Integer,
                  sym.numbers.NegativeOne,
                  sym.numbers.Half,
                  sym.numbers.One
    ]:
        return {}, (e, e)

    elif e.func == sym.Symbol:
        if e in i_env:
            return {e : 1}, i_env[e]
        else:
            return {e : 1}, (0, 1) # dangerous default

    elif e.func == sym.Add:
        all_ss, all_is = zip(*[sens(a, i_env) for a in e.args])
        s = reduce(add_envs, all_ss)

        all_rls, all_rhs = zip(*all_is)
        rl = np.sum(all_rls)
        rh = np.sum(all_rhs)

        return s, (rl, rh)

    elif e.func == sym.Mul:
        e1, e2 = e.args

        s1, i1 = sens(e1, i_env)
        s2, i2 = sens(e2, i_env)

        rl1, rh1 = i1
        rl2, rh2 = i2

        ss1 = scale_env(max(abs(rl2), abs(rh2)), s1)
        ss2 = scale_env(max(abs(rl1), abs(rh1)), s2)
        s = add_envs(ss1, ss2)

        rl = min(rl1*rl2, rl1*rh2, rh1*rl2, rh1*rh2)
        rh = max(rl1*rl2, rl1*rh2, rh1*rl2, rh1*rh2)

        return s, (rl, rh)


    elif e.func == sym.Pow:
        log('found pow')
        print(e)
        raise 5
        [analyze(a) for a in e.args]

    elif e.func == Relu:
        assert len(e.args) == 1
        s, i = sens(e.args[0], i_env)
        rl, rh = i

        if rl < 0:
            s_prime = {k : 0 for k in s.keys()}
        else:
            s_prime = s

        rl_prime = max(rl, 0)
        rh_prime = max(rh, 0)

        return s_prime, (rl_prime, rh_prime)

    else:
        log('found unknown type' + str(e.func))
        raise 5


tx = sym.Symbol('x')
ty = sym.Symbol('y')

print(sens(tx * ty, {tx : (0, 1), ty : (0, 1)}))
print(sens(subst, {}))
        
# for x in inputs:
#     print(f'immediate sensitivity wrt {x}: {subst}')
#     print(f'log of immediate sensitivity wrt {x}: {sym.simplify(sym.log(subst))}')
#     print(f'1st derivative of immediate sensitivity wrt {x}: {sym.diff(subst, x)}')
#     print(f'2nd derivative of immediate sensitivity wrt {x}: {sym.diff(sym.diff(subst, x), x)}')

# OUTPUT (all weights are 0.1):
# 1st derivative of immediate sensitivity wrt x_0: 0.0280800000000000
# 2nd derivative of immediate sensitivity wrt x_0: 0
# 1st derivative of immediate sensitivity wrt x_1: 0.0280800000000000
# 2nd derivative of immediate sensitivity wrt x_1: 0

# OUTPUT (all weights are 1.0):
# 1st derivative of immediate sensitivity wrt x_0: 2808.00000000000
# 2nd derivative of immediate sensitivity wrt x_0: 0
# 1st derivative of immediate sensitivity wrt x_1: 2808.00000000000
# 2nd derivative of immediate sensitivity wrt x_1: 0
