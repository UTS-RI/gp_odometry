import numpy as np
from sklearn.metrics import pairwise_distances
from pykeops.numpy import LazyTensor



def kernelSE(X1, X2, length_scale, with_grad = False):
    l2 = length_scale*length_scale
    k = np.exp(-pairwise_distances(X1,X2,'sqeuclidean')/(2*l2))
    if with_grad:
        dim = X1.shape[1]
        grad = np.empty((dim, X1.shape[0], X2.shape[0]))
        for i in range(dim):
            grad[i,:,:] = -k*((X1[:,i][:,np.newaxis])-(X2[:,i][np.newaxis,:])) / (l2)
        return k, grad
    else:
        return k

# Unit test for the gradients of kernelSE
def testKernelSE():
    print("Testing kernelSE")
    dim = 2
    X1 = np.random.rand(5,dim)
    X2 = np.random.rand(10,dim)
    length_scale = 0.1
    k, grad = kernelSE(X1, X2, length_scale, with_grad = True)
    quantum = 0.0001
    num_grad = np.empty_like(grad)
    for i in range(dim):
        X1_shift = X1.copy()
        X1_shift[:,i] += quantum
        k_shift = kernelSE(X1_shift, X2, length_scale, with_grad = False)
        num_grad[i,:,:] = (k_shift-k)/quantum

    print("Max error ", np.max(np.abs(grad-num_grad)))
    print(np.stack((grad.ravel(), num_grad.ravel()), axis = 1))

def keopsSE(x,y,length_scale, with_grad = False):
    l2 = length_scale*length_scale
    x_i = LazyTensor(x[:, None, :])  # (M, 1, 1)
    y_j = LazyTensor(y[None, :, :])  # (1, N, 1)
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) symbolic matrix of squared distances
    K = (-D_ij / (2*l2)).exp()  # (M, N) symbolic Gaussian kernel matrix
    if with_grad:
        dim = x.shape[1]
        #grad = LazyTensor(np.empty((x.shape[0], y.shape[0], 1, dim)))
        grad = []
        for i in range(dim):
            #temp_grad = -K*((x_i[:,:,i])-(y_j[:,:,i])) / (l2)
            grad.append(-K*((x_i[:,:,i])-(y_j[:,:,i])) / (l2))
        return K, LazyTensor.concatenate(tuple(grad))
    else:
        return K


# Unit test for the gradients of keopsSE
def testKeopsSE():
    import pykeops
    pykeops.clean_pykeops()
    print("Testing keopsSE")
    dim = 2
    x = np.random.rand(5,dim)
    y = np.random.rand(10,dim)
    length_scale = 0.1
    k_keops, grad_keops = keopsSE(x,y,length_scale, with_grad = True)
    k = np.empty((x.shape[0], y.shape[0]))
    for i in range(y.shape[0]):
        selector = np.zeros((y.shape[0],1))
        selector[i] = 1
        k[:,i] = (k_keops @ selector).squeeze()
    grad = np.empty((dim, x.shape[0], y.shape[0]))
    for i in range(dim):
        for j in range(y.shape[0]):
            selector = np.zeros((y.shape[0],1))
            selector[j] = 1
            grad[i,:,j] = (grad_keops[:,:,i] @ selector).squeeze()

    k_np, grad_np = kernelSE(x,y,length_scale, with_grad = True)

    print("Max K diff with Numpy ", np.max(np.abs(k-k_np)))
    print("Max grad diff with Numpy ", np.max(np.abs(grad-grad_np)))




def revertKernelSE(occ, length_scale, with_grad = False):
    l2 = length_scale*length_scale
    occ_temp = occ.copy()
    occ_temp[occ_temp>1] = 1
    dist = np.empty_like(occ_temp)
    dist[occ_temp>0] = np.sqrt(-np.log(occ_temp[occ_temp>0])*(2*l2))
    dist[occ_temp<=0] = 1000
    if with_grad:
        grad = np.empty_like(occ_temp)
        mask = np.logical_and(occ_temp>0, occ_temp<1)
        grad[mask] = -l2/(occ_temp[mask]*dist[mask])
        grad[occ_temp<=0] = 0
        grad[occ_temp>=1] = 0
        return dist, grad
    else:
        return dist

# Unit test for the gradients of revertKernelSE
def testRevertKernelSE():
    print("Testing revertKernelSE")
    occ = np.linspace(-0.2, 1.2, 100)
    quantum = 0.0001
    length_scale = 0.1
    dist, grad = revertKernelSE(occ, length_scale, with_grad = True)
    
    dist_shift = revertKernelSE(occ+quantum, length_scale, with_grad = False)

    num_grad = (dist_shift-dist)/quantum
    print("Max error ", np.max(np.abs(grad-num_grad)))
    print(np.stack((grad, num_grad), axis = 1))


def revertKeopsSE(occ, length_scale, with_grad = False):
    l2 = length_scale*length_scale
    occ_temp = occ.copy()
    mask_sup = occ_temp>1
    occ_temp[mask_sup] = 1
    mask = occ_temp<=0
    occ_temp[mask] = 1
    occ_i = LazyTensor(occ_temp, axis=0)
    dist_i = (-(occ_i.log()*(2*l2))).sqrt()
    # From Keops to numpy
    dist = dist_i.sum(1)
    dist[mask] = 1000
    if with_grad:
        grad_i = -l2/(occ_i*dist_i)
        grad = grad_i.sum(1)
        grad[mask_sup] = 0
        grad[mask] = 0


        #grad = np.empty_like(occ)
        #mask = np.logical_and(occ>0, occ<1)
        #grad[mask] = -l2/(occ[mask]*dist[mask])
        #grad[occ<=0] = 0
        #grad[occ>=1] = 0
        return dist, grad
    else:
        return dist

# Unit test for the gradients of revertKeopsSE
def testRevertKeopsSE():
    import pykeops
    pykeops.clean_pykeops()
    print("Testing revertKeopsSE")
    occ = np.linspace(-0.2, 1.2, 100)
    quantum = 0.0001
    length_scale = 0.1
    dist, grad = revertKeopsSE(occ[:,None], length_scale, with_grad = True)
    
    dist_shift = revertKeopsSE((occ+quantum)[:,None], length_scale, with_grad = False)

    num_grad = (dist_shift-dist)/quantum
    print("Max error ", np.max(np.abs(grad-num_grad)))
    print(np.stack((grad, num_grad), axis = 1))


class gpDistFieldSE:

    # Please note that measurement noise is hardcoded here. It is to speedup/robustify the GPU solver for the GP
    def __init__(self, map_pts, length_scale, sz = 0.01, use_keops = False, lumped_matrix = False):
        self.length_scale = length_scale
        self.map_pts = map_pts
        nb_map_pts = map_pts.shape[0]

        if use_keops:
            self.use_keops = True

            self.dtype = 'float64'
            self.map_pts = map_pts.astype(self.dtype)
            map_float = map_pts.astype('float32')
            K = keopsSE(map_float, map_float, length_scale)
            # Alpha (sz*sz) hardcoded for numerical reasons
            if lumped_matrix:
                self.alpha = (1.0/(K.sum(1))).reshape(-1,1).astype(self.dtype)
            else:
                self.alpha = K.solve(np.ones((nb_map_pts,1), dtype = 'float32'), alpha = 0.1).astype(self.dtype)


        else:
            self.use_keops = False

            K = kernelSE(map_pts, map_pts, length_scale) + (sz*sz)*np.eye(map_pts.shape[0])
            if lumped_matrix:
                self.alpha = (1.0/np.sum(K,axis=0)).reshape(-1,1)
            else:
                self.alpha = np.linalg.solve(K+(sz*sz*np.eye(nb_map_pts)), np.ones((map_pts.shape[0],1)))



    def query(self, pts):

        if self.use_keops:
            ks = keopsSE(pts.astype(self.dtype),self.map_pts,self.length_scale)
            occ = ks @ self.alpha
            return revertKeopsSE(occ, self.length_scale)
        
        else:
            ks = kernelSE(pts, self.map_pts, self.length_scale)
            occ = ks @ self.alpha
            return revertKernelSE(occ, self.length_scale).squeeze()

    def queryWithGrad(self, pts):
        if self.use_keops:
            ks, ks_grad = keopsSE(pts.astype(self.dtype),self.map_pts,self.length_scale, with_grad = True)
            occ = ks @ self.alpha
            dist, grad = revertKeopsSE(occ, self.length_scale, with_grad = True)
            dim = pts.shape[1]
            d_occ_d_pts = np.empty((dim, pts.shape[0], 1))
            for i in range(dim):
                d_occ_d_pts[i,:,:] = ks_grad[:,:,i] @ self.alpha
            return dist, (grad * d_occ_d_pts).squeeze().T
        
        else:
            ks, ks_grad = kernelSE(pts, self.map_pts, self.length_scale, with_grad = True)
            occ = ks @ self.alpha
            dist, grad = revertKernelSE(occ, self.length_scale, with_grad = True)
            d_occ_d_pts = ks_grad @ self.alpha
            return dist.squeeze(), (grad*d_occ_d_pts).squeeze().T
        

# Unit test for the gradients of gpDistFieldSE
def testGpDistFieldSE():
    print("Testing gpDistFieldSE")
    dim = 3
    map_pts = np.random.rand(100,dim)
    length_scale = 0.1
    sz = 0.01
    gp = gpDistFieldSE(map_pts, length_scale, sz = sz, use_keops = False)
    pts = np.random.rand(10,dim)
    dist, grad = gp.queryWithGrad(pts)
    quantum = 0.0001
    num_grad = np.empty_like(grad)
    for i in range(dim):
        pts_shift = pts.copy()
        pts_shift[:,i] += quantum
        dist_shift, _ = gp.queryWithGrad(pts_shift)
        num_grad[:,i] = (dist_shift-dist)/quantum

    print("Max error ", np.max(np.abs(grad-num_grad)))
    print(np.stack((grad.ravel(), num_grad.ravel()), axis = 1))