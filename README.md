### milligrad (micrograd clone)

This is a little project to build a tiny automatic differentiation (autograd) engine from scratch. it is a bit of a clone of [micrograd](https://github.com/karpathy/micrograd).

my goal is just to understand at a deep level the mechanics of backpropagation. 

it is ongoing, although some of the modifications I have implemented include transforming micrograd from a scalar-only toolkit into a tensor-aware one. it extends the engine to arbitrary shapes, reduces duplication and has some slight, basic modifications to optimize runtime and memory usage.

### credits and huge thanks

needless to say that this project (among others) would not exist without the work of **Andrej Karpathy**. it is a direct result of following his great ["The spelled-out intro to neural networks and backpropagation: bulding micrograd"](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1). all the core ideas and implementation details are his. 

if you haven't, please go watch all of his videos. they are the best, clear, intuitive and hands-on explanations  of just about anything happening under the hood of deep learning. 

**thank you andrej** :goat: :goat:
