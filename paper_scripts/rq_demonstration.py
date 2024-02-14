import numpy as np
import scipy.linalg as linalg
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors as nn
from matplotlib import pyplot as plt


def kernelRQ(X1, X2, alpha, hyper):
    return (hyper['sf']**2)*np.power(1+(pairwise_distances(X1,X2,'sqeuclidean')/(2*alpha*(hyper['l']**2))), -alpha)

def kernelRQDiff(X1, X2, alpha, hyper, axis=0):
    l2 = hyper['l']**2
    sf2 = hyper['sf']**2
    dist = pairwise_distances(X1,X2,'sqeuclidean')
    temp = (1+(dist/(2*alpha*l2)))**(-alpha-1)
    return -(sf2/l2)*temp*((X1[:,axis][:,np.newaxis])-(X2[:,axis][np.newaxis,:]))

def occRQToDist(occ, alpha, hyper):
    temp_occ = np.array(occ)
    if temp_occ.shape == ():
        temp_occ = temp_occ[np.newaxis]
    temp_occ[temp_occ>1] = 1
    temp = np.power(hyper['sf']**2/temp_occ, 1/alpha)-1
    temp[temp<0] = 0
    dist = np.sqrt(2*alpha*(hyper['l']**2)*temp)
    return dist

# Generate simulated data
# Inputs:
#    - nb_pts_obs : number of points for use later in the GP
#    - res_inference : resolution of the inference grid (used to generate the ground truth)
#    - x_range : range of the x axis
#    - y_range : range of the y axis
#    - nb_pts_gt : number of points to generate for the ground truth
# Outputs:
#    - X_out : nb_pts_obs*2 numpy array of observed points
#    - inf_X : inference grid
#    - gt : ground truth distance field
def genData(nb_pts_obs = 200, res_inference = 0.03, x_range=[-3,2], y_range=[-2.5,2.8], nb_pts_gt = int(1e6)):

    x_inf_steps = np.arange(x_range[0], x_range[1], res_inference)
    y_inf_steps = np.arange(y_range[0], y_range[1], res_inference)

    temp_x, temp_y = np.meshgrid(x_inf_steps, y_inf_steps)
    inf_X = np.column_stack((temp_x.flatten(), temp_y.flatten()))


    #t = 2.2*np.arange(0,1, 1/kNbPointsGt, dtype=np.float128)

    #x = np.sin(t*0.7*np.pi) + np.sin(t*1.32*np.pi) + np.sin(t*0.83*np.pi)
    #y = np.sin(t*1.56*np.pi) + np.sin(t*1.23*np.pi) + np.sin(t*0.2*np.pi)
    #t = 2*np.linspace(0.2,0.7, nb_pts_obs)
    t = 2.2*np.linspace(0,1, nb_pts_obs)
    coeff = np.random.randn(6)
    x = np.sin(t*coeff[0]*np.pi) + np.sin(t*coeff[1]*np.pi) + np.sin(t*coeff[2]*np.pi)
    y = np.sin(t*coeff[3]*np.pi) + np.sin(t*coeff[4]*np.pi) + np.sin(t*coeff[5]*np.pi)

    X_out = np.column_stack((x, y))

    t_gt = 2.2*np.linspace(0,1, nb_pts_gt)
    x_gt = np.sin(t_gt*coeff[0]*np.pi) + np.sin(t_gt*coeff[1]*np.pi) + np.sin(t_gt*coeff[2]*np.pi)
    y_gt = np.sin(t_gt*coeff[3]*np.pi) + np.sin(t_gt*coeff[4]*np.pi) + np.sin(t_gt*coeff[5]*np.pi)
    X_gt = np.column_stack((x_gt, y_gt))



    knn = nn(n_neighbors=1)
    knn.fit(X_gt)
    dist_mat, nn_mat = knn.kneighbors(inf_X)
    
    gt = dist_mat.reshape(temp_x.shape)


    return X_out, inf_X, gt




# Object to query distance measurements from map points
class gpDistField:

    # Constructor
    # INputs:
    #    - map_pts : N*3 numpy array of map points
    #    - lengthscale : lengthscale of the GP kernel 
    #    - kernel ('matern' | 'se') : type of GP kernel
    #    - nu : nu parameter in the Matern kernel (only used when kernel='matern')
    def __init__(self, map_pts, lengthscale, rq_alpha = 100, add_data_noise = False, precompute_inverse = False, sz=0.1):
        self.hyper = {'l': lengthscale, 'sf': 1.0, 'sz': sz}
        self.map_pts = np.copy(map_pts)
        self.rq_alpha = rq_alpha
        self.add_data_noise = add_data_noise
        self.alpha = {}
        self.dim = np.size(map_pts,1)


        self.computeAlpha(precompute_inverse)

    def getCovVec(self, pts, map_pts=None):
        if map_pts is None:
            map_pts = self.map_pts
        ks = kernelRQ(pts,map_pts, self.rq_alpha, self.hyper)
        return ks

    def getCovVecDiff(self, pts, axis=0, map_pts=None):
        if map_pts is None:
            map_pts = self.map_pts
        ks = kernelRQDiff(pts,map_pts, self.rq_alpha, self.hyper, axis)
        return ks

    def occToDist(self, occ):
        dist = occRQToDist(occ, self.rq_alpha, self.hyper)
        return dist

    def computeAlpha(self, precompute_inverse = False):
        nb_pts = np.size(self.map_pts,0)
        lengthscale = self.hyper['l']
        Y = np.ones(nb_pts)

        temp_X = self.map_pts + (lengthscale/20)*np.random.randn(np.size(self.map_pts, 0), np.size(self.map_pts,1)) if self.add_data_noise else self.map_pts
        self.map_pts = temp_X
        K = kernelRQ(temp_X, temp_X, self.rq_alpha, self.hyper)
        
        if precompute_inverse:
            self.K_inv = linalg.inv(K + (self.hyper['sz']**2)*np.eye(nb_pts))
            self.alpha = self.K_inv@Y
        else:
            self.alpha = linalg.solve( K + (self.hyper['sz']**2)*np.eye(nb_pts),Y, assume_a='sym')



    # Query the distance for a given set of points
    def query(self, pts, occ_out=False, smooth_min_fusion = False):
        ks = self.getCovVec(pts)
        occ = ks@self.alpha
        if occ_out:
            return occ
        dist = self.occToDist(occ)
        return dist
    
    def queryDistGrad(self, pts):
        occ_grad, _ = self.queryOccGrad(pts)
        # Normalise each row
        grad_norm = np.linalg.norm(occ_grad, axis=1, ord=2)
        grad = -occ_grad/grad_norm[:,np.newaxis]
        return grad

    # Query the gradient of the occ field
    def queryOccGrad(self, pts):
        # Get the gradient of the occ field
        if not hasattr(self, 'K_inv'):
            self.computeAlpha(True)
        
        occ_grad = np.empty((np.size(pts,0), self.dim))
        occ_grad_var = np.empty((np.size(pts,0), self.dim))
        for i in range(self.dim):
            ks = self.getCovVecDiff(pts, i)
            occ_grad[:,i] = ks @ self.alpha
            for j in range(np.size(pts,0)):
                occ_grad_var[j,i] = -ks[j,:] @ self.K_inv @ ks[j,:].T + (self.hyper['sf']**2/(self.hyper['l']**2))
                
        return occ_grad, occ_grad_var


    # Query the uncertainty proxy of the distance field
    def queryUncertainty(self, pts):

        occ = self.query(pts, occ_out=True)
        dist = self.occToDist(occ)
        occ_grad, occ_grad_var = self.queryOccGrad(pts)

        mask = np.isnan(dist)
        dist[mask] = self.hyper['l']
        model_grad_norm = np.abs(self.getCovVecDiff(dist.reshape(-1,1), 0, np.zeros((1,1))))

        grad_norm = np.linalg.norm(occ_grad, axis=1, ord=2)

        # Propagate the variance of the occ grad through the norm function
        norm_var = np.empty(np.size(pts,0))
        for i in range(np.size(pts,0)):
            if(grad_norm[i] == 0):
                norm_var[i] = np.mean(occ_grad_var[i,:])
            else:
                jacobian = occ_grad[i,:]/grad_norm[i]
                norm_var[i] = jacobian@np.diag(occ_grad_var[i,:])@(jacobian.T)

        uncertainty = np.sqrt((model_grad_norm.squeeze() - grad_norm.squeeze())**2 / norm_var)

        return uncertainty




if __name__ == "__main__":
    np.random.seed(21)
    X, inf_X, gt = genData()

    lengthscale = 1.5*np.mean(np.sqrt(np.sum((X[:-1,:]-X[1:,:])**2, axis=1)))

    field = gpDistField(X, lengthscale=lengthscale, rq_alpha=50, precompute_inverse=False, sz=0.1)

    dist = field.query(inf_X)
    uncertainty = field.queryUncertainty(inf_X)
    error = np.abs(gt - dist.reshape(gt.shape))
    rmse = np.sqrt(np.mean(error**2))

    # Query some gradients (normalised)
    grid_size = 0.3
    x = np.arange(np.min(inf_X[:,0]), np.max(inf_X[:,0]), grid_size)
    y = np.arange(np.min(inf_X[:,1]), np.max(inf_X[:,1]), grid_size)
    temp_x, temp_y = np.meshgrid(x, y)
    grad_X = np.column_stack((temp_x.flatten(), temp_y.flatten()))
    dist_grad = field.queryDistGrad(grad_X)



    # Display the ground truth
    extent = [np.min(inf_X[:,0]), np.max(inf_X[:,0]), np.min(inf_X[:,1]), np.max(inf_X[:,1])]

    fig, axs = plt.subplots(2,2, figsize=(8,8))
    # Plot the ground truth
    im = axs[0,0].imshow(gt, extent = extent, origin='lower')
    axs[0,0].scatter(X[:,0], X[:,1], s=1, c='r', label='Obs.')
    axs[0,0].legend()
    axs[0,0].set_title("Ground truth")
    fig.colorbar(im, ax=axs[0,0])

    # Plot the estimated distance field
    im = axs[0,1].imshow(dist.reshape(gt.shape), extent = extent, origin='lower')
    axs[0,1].set_title("Estimated dist field")
    fig.colorbar(im, ax=axs[0,1])
    # Show the gradient as arrows
    axs[0,1].quiver(grad_X[:,0], grad_X[:,1], dist_grad[:,0], dist_grad[:,1], color='w', label='Grad', width=0.002, headwidth=5, headlength=5)
    axs[0,1].legend()

    # Plot the uncertainty
    im = axs[1,0].imshow(uncertainty.reshape(gt.shape), extent = extent, origin='lower')
    axs[1,0].set_title("Uncertainty")
    fig.colorbar(im, ax=axs[1,0])

    # Plot the error
    im = axs[1,1].imshow(error, extent = extent, origin='lower')
    axs[1,1].set_title("Error")
    axs[1,1].text(0.5, 0.1, "RMSE: "+str(rmse.round(4)), fontsize=12, color='white', ha='center', va='center', transform=axs[1,1].transAxes)
    fig.colorbar(im, ax=axs[1,1])


    plt.show()







