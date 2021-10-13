import numpy
import aesara
from aesara import tensor as aet
from aesara import pp
# x = aet.dscalar('x')
# y = x ** 2
# gy = aet.grad(y, x)
# print(pp(gy))  # print out the gradient prior to optimization
# f = aesara.function([x], gy)

# print(f(4))


x = aet.dvector('x')
y = aet.dvector('y')

print(aet.grad(x.norm(2), x))
print(pp(aet.grad(x.norm(2), x)))

# layer1_weights = aet.dmatrix('l1w')
# print(aet.dot(x, layer1_weights))
# print(aet.grad(aet.dot(x, layer1_weights), x))
