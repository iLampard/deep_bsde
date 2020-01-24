# Deep Learning based Backward Stochastic Differential Equation

The repo is a Tensorflow implementation of deep learning based BSDE.
- Two versions of algorithms, based on paper [Solving high-dimensional partial differential equations using deep learning](https://arxiv.org/abs/1707.02568) 
and [Machine Learning for Semi Linear PDEs](https://arxiv.org/abs/1809.07609v1) respectively, have been implemented.
- A Black-Schole Call option BSDE has been added as an example. 
- A *ModelRunner* class is added to control the pipeline of model 
training and evaluation: evaluate the performance on testset only when the lowest-by-far
 validation loss has been achieved.


# How to Run

By default we use method mentioned in [Solving high-dimensional partial differential equations using deep learning](https://arxiv.org/abs/1707.02568)
```bash
python main.py 
```

If we want to use the method mentioned in [Machine Learning for Semi Linear PDEs](https://arxiv.org/abs/1809.07609v1),
```bash
python main.py --network_type merged
```



# Requirement

```bash
tensorflow==1.13.1
```






# Reference
- [DeepBSDE](https://github.com/frankhan91/DeepBSDE)
