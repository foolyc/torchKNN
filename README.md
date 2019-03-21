# torchKNN
KNN implement in Pytorch 1.0 including both cpu version and gpu version

# dependency

tested with ubuntu16.04 + cuda9.0 + python3 + pytorch nightly

## Installation
```
git clone https://github.com/foolyc/torchKNN.git
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