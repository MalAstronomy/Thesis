import numpy as np
import h5py 
import matplotlib.pyplot as plt

size_ratio=[]
mass_ratio=[]

for i in [19,25,26,27,28]:

    f= h5py.File('/Users/malavikavijayendravasist/Desktop/mt2/mergers_identified/mergers_'+str(i)+'.hdf5', 'r')

    size_ratio=np.append(size_ratio,f.get('Size Ratio').value)
    mass_ratio=np.append(mass_ratio,f.get('Mass Ratio').value)


plt.subplot(1,2,1)
plt.hist(size_ratio,bins=np.arange(0,1.25,0.1))
plt.title('Size Ratio' )

plt.subplot(1,2,2)
plt.hist(mass_ratio,bins=np.arange(0,1,0.1))
plt.title('Mass Ratio' )
plt.show()


