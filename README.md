# torchKNN
This is a fork from https://github.com/foolyc/torchKNN, a KNN implement in Pytorch including both cpu version and gpu version.

I modify it to support Pytorch>=1.11.

# dependency

tested with ubuntu20.04 + cuda11.3 + python3.9 + pytorch1.12

## Installation
```
git clone https://github.com/HaoyiZhu/torchKNN.git
cd torchKNN
sudo sudo python3 setup.py install
```

## Usage

ref [batch, dim, num_ref]

query [batch, dim, num_query]

inds [batch, k, num_query]

```
knn(ref, query, inds)
```