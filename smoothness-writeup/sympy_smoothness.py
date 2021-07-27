import time

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
layer1_weights = build_weights(1, 10, 10, all_weights)
layer2_weights = build_weights(2, 10, 10, all_weights)
layer3_weights = build_weights(3, 10, 1, all_weights)

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
inner_norm = L2_norm(inner_gradient)
outer_gradient = [sym.diff(inner_norm, x) for x in inputs]
outer_norm = L2_norm(outer_gradient)
immediate_sensitivity = outer_norm

print('done constructing, time:', time.process_time() - start_time)

print('substituting')
start_time = time.process_time()

# substitute in actual values for the weights
weight_vals = [1.0 for w in all_weights]
subst = immediate_sensitivity
# for w_name, w_val in zip(all_weights, weight_vals):
#     subst = subst.subs(w_name, w_val)

# subst = subst.subs(label, 1)

ivl = sym.Interval(-1,1)

# print(sym.log(subst))
# print(sym.srepr(subst))

print('done substituting, time:', time.process_time() - start_time)


def log(m):
    pass
    #print(m)

# recursive analysis for sympy ASTs
def analyze(e):
    if e.func in [sym.Float, sym.Integer,
                  sym.core.numbers.NegativeOne,
                  sym.core.numbers.Half]:
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

analyze(sym.log(subst))
print('done analyzing, time:', time.process_time() - start_time)

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
