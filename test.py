import unittest
import torch
from torch.autograd import Variable, Function

from vision_sandbox import _C
knn = _C.knn

class TestKNearestNeighbor(unittest.TestCase):

    def test_forward(self):
        D, N, M = 3, 100, 500
        ref = Variable(torch.rand(2, D, N))
        query = Variable(torch.rand(2, D, M))
        ref = ref.float()
        query = query.float()

        inds = torch.empty(query.shape[0], 2, query.shape[2]).long()

        knn(ref, query, inds)


        print(query)
        print(ref)

        print("CPU")
        print(inds.shape)
        print(inds)

        print("CUDA")
        ref = ref.cuda()
        query = query.cuda()
        inds = inds.cuda()
        knn(ref, query, inds)
        print(inds)



if __name__ == '__main__':
  unittest.main()
