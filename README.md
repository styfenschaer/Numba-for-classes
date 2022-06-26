# Numba-for-classes
Just-in-time compile class methods with Numba 🚀

With *Numbas* you can now accelerate class methods with Numba without writing overly verbose code. 
If your class is suitable to be accelerated with *Numbas*, then there is good news: 
You will have to make very few changes to your code.
Obviously, there are many reasons why this is not directly supported by *Numba*. So don't expect this package to solve all your problems. 
It works in many cases, but has some clear limitations imposed by *Numba* itself, and also introduces new ones.

However, to stop boring you and get you started quickly, I'll give you a small example of what *Numba* is good for.
Let's scale all the elements of an array to the interval [0,1], similarly to scikit-learn's
[MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).


```python
import numpy as np
import numbas as nbs

class MinMaxScaler:
    def fit(self, X):
        data_min_, data_max_ = X.min(), X.max()
        self.scale_ = 1 / (data_max_ - data_min_)
        self.min_ = -data_min_ * self.scale_

    @nbs.jit(parallel=True)
    def jit_transform(self, X):
        return X * self.scale_ + self.min_

    def np_transform(self, X):
        return X * self.scale_ + self.min_
        
 
X = np.random.randint(-42, 42, size=42_000_000)

scaler = MinMaxScaler()
scaler.fit(X)

# the first call is slow because it includes the compilation time
%timeit -r 1 -n 1 scaler.jit_transform(X) 
# subsequent calls are much faster 
%timeit -r 10 -n 3 scaler.jit_transform(X)
# let's compare to a pure Numpy implementation
%timeit -r 10 -n 3 scaler.np_transform(X)

650 ms ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
84.5 ms ± 2.27 ms per loop (mean ± std. dev. of 10 runs, 3 loops each)
271 ms ± 3.63 ms per loop (mean ± std. dev. of 10 runs, 3 loops each) 
```

When the time is right, I will give more examples of what *Numbas* can and cannot do. 
You may be surprised how much you can actually do with such a simple solution as *Numbas*, 
but also how limited it is, although many limitations could be fixed with a little love and time.

One limitation I won't deprive you of is the need to reset your compiled function before changing the 
value of class attributes.
In our example, this means we need to be careful when fitting our scaler to new data. 
You can do it this way:


```python
X_new = np.random.randint(-1, 1, size=42_000_000)

nbs.reset(scaler.jit_transform)
scaler.fit(X_new)
```

Because it is not uncommon that the change of class attributes is associated with another method call, 
*Numbas* provides a more convenient way to do this.
In our case, we want to reset our decorated function every time we refit the scaler. We can achieve this 
by informing the ``fit`` method about the class methods we want to reset:

```python
@nbs.reset("jit_transform")
def fit(self, X):
    data_min_, data_max_ = X.min(), X.max()
    self.scale_ = 1 / (data_max_ - data_min_)
    self.min_ = -data_min_ * self.scale_
```

In this sense I wish you a lot of fun with troubleshooting.