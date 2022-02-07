import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from elastic_desc1 import elastic_desc
from elastic_flow import elastic_flow
from matplotlib import pyplot as plt

scaler = StandardScaler()
np.set_printoptions(suppress=True)
X, y = datasets.load_diabetes(return_X_y=True)
p=X.shape[1]

##Select a random subset of variables
#n_vars = np.random.randint(p)+1
#ds=np.random.choice(p,n_vars, replace=False)
#print(ds)
#X=X[:,ds]

X=scaler.fit_transform(X)

ALPHA=0.5
STEP_SIZE=1e-2
betas, grads=elastic_desc(X, y, ALPHA, STEP_SIZE=STEP_SIZE)

ts=np.arange(0,betas.shape[0],1)*STEP_SIZE

f, axs = plt.subplots(2, 1)
for d in range(betas.shape[1]):
  axs[0].plot(ts, betas[:,d])
  axs[1].plot(ts, grads[:,d]/np.max(np.abs(grads),1))

axs[0].set_ylabel('$\\beta$')
axs[1].set_ylabel('$g/||g||_\infty$')

plt.savefig('figures/elastic_desc.pdf')
plt.close()


betas, grads, _, _=elastic_flow(X, y, ALPHA, STEP_SIZE=STEP_SIZE, PRINT=True)

ts=np.arange(0,betas.shape[0],1)*STEP_SIZE
f, axs = plt.subplots(2, 1)
for d in range(betas.shape[1]):
  axs[0].plot(ts, betas[:,d])
  axs[1].plot(ts, grads[:,d]/np.max(np.abs(grads),1))

axs[0].set_ylabel('$\\beta$')
axs[1].set_ylabel('$g/||g||_\infty$')

plt.savefig('figures/elastic_flow.pdf')
plt.close()
