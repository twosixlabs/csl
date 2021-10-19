import sympy as sym

class ReluDiff(sym.Function):
    @classmethod
    def eval(cls, x):
        if x.is_Number:
            if x > 0:
                return 1
            else:
                return 0

    def fdiff(self, x):
        return 0


class AbsDiff(sym.Function):
    @classmethod
    def eval(cls, x):
        if x.is_Number:
            if x > 0:
                return 1
            else:
                return -1

    def fdiff(self, x):
        return 0


class Relu(sym.Function):
    @classmethod
    def eval(cls, x):
        if x.is_Number:
            if x > 0:
                return x
            else:
                return 0
        

    def fdiff(self, x):
        return ReluDiff(x)

# TODO: this might be broken because it doesn't have an "eval" method
class MyAbs(sym.Function):
    @classmethod
    def eval(cls, x):
        if x.is_Number:
            if x > 0:
                return x
            else:
                return -x

    def fdiff(self, x):
        return AbsDiff(x)

w0, w1, x, b0, b1, y = sym.symbols('w0 w1 x b0 b1 y')

inter = Relu(w0*x + b0)

expr = (y - Relu(w1*inter + b1))**2
print(Relu(1))

print(f"single neruon: \n{expr} \n")

dw0 = sym.diff(expr, w0)
dw1 = sym.diff(expr, w1)
db0 = sym.diff(expr, b0)
db1 = sym.diff(expr, b1)
print(f"dw: \n{dw0}", f"\ndb \n{db0}\n")

di = MyAbs(dw0) + MyAbs(dw1) + MyAbs(db0) + MyAbs(db1)
print(f"l1 norm of dw and db:\n {di}\n")

d2 = sym.diff(di, x)
print(f"IS:\n {d2}\n")

d3 = sym.diff(MyAbs(d2), x)
print(f"dIS:\n {d3}\n")


