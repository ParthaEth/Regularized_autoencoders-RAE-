import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

model_path = '/is/ps2/pghosh/repos/high_res_vae (copy)/logs/0/VAE_11/GMM_10_mdl.pkl'
test_zs = np.load('/is/ps2/pghosh/repos/high_res_vae (copy)/logs/0/VAE_11/std_vae_MNIST._VAE__test_embedding.npz')['zs']

with open(model_path, 'rb') as f:
    clf = pickle.load(f)

# 10K smpls
zs, label = clf.sample(10000)

np.savez('10k_laten_smpls.npy', zs=zs, label=label)

# display predicted scores by the model as a contour plot
x = np.linspace(-6., 6.)
y = np.linspace(-6., 6.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = np.exp(clf.score_samples(XX))
Z = Z.reshape(X.shape)

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
CS = plt.contour(X, Y, Z)
# ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,)
# CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
#                  levels=np.logspace(0, 3, 10))
# CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(test_zs[0:2000, 0], test_zs[0:2000, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()