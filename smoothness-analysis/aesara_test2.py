import numpy
import aesara
import aesara.tensor as aet
from aesara import pp
rng = numpy.random

N = 400                                   # training sample size
feats = 784                               # number of input variables

# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 10000

# Declare Aesara symbolic variables
x = aet.dmatrix("x")
y = aet.dvector("y")

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
w = aesara.shared(rng.randn(feats), name="w")

# initialize the bias term
b = aesara.shared(0., name="b")

print("Initial model:")
print(w.get_value())
print(b.get_value())

# Construct Aesara expression graph
p_1 = 1 / (1 + aet.exp(-aet.dot(x, w) - b))        # Probability that target = 1
prediction = p_1 > 0.5                          # The prediction thresholded
xent = -y * aet.log(p_1) - (1-y) * aet.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()      # The cost to minimize
gw, gb = aet.grad(cost, [w, b])                  # Compute the gradient of the cost
                                                # w.r.t weight vector w and
                                                # bias term b (we shall
                                                # return to this in a
                                                # following section of this
                                                # tutorial)

# Compile
# train = aesara.function(
#           inputs=[x,y],
#           outputs=[prediction, xent],
#           updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = aesara.function(inputs=[x], outputs=prediction)

# Train
# for i in range(training_steps):
#     pred, err = train(D[0], D[1])

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))

print('my stuff')
print(pp(gw))

print('gradient wrt inputs')
grad_norm = gw.norm(1)
imm_sens = aet.grad(grad_norm, x)
print(pp(imm_sens))
