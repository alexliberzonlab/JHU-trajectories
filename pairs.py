
import numpy as np
import pyJHTDB
from pyJHTDB.dbinfo import isotropic1024coarse
info = isotropic1024coarse
# 20 seed nodes in 3D, x,y,z
ind = np.random.randint(45,1024-45,size=(20,3))

eta = 0.002 # Kolmogorov
positions = []

for i in range(20):
    tmp = 128*eta*(np.random.random((400,3))-0.5) # 20 particles around the selected node within 128 \eta
    positions.append((info['xnodes'][ind[i,0]],info['ynodes'][ind[i,1]],info['znodes'][ind[i,2]]) + tmp)
    
positions = np.vstack(positions).astype(np.float32)
lJHTDB = pyJHTDB.libJHTDB()

lJHTDB.initialize()

u = lJHTDB.getData(
               0.0,
               positions,
               sinterp = 4,
               getFunction='getVelocity')


x, t = lJHTDB.getPosition(
    starttime = 0.0,
    endtime = 4.0,
    dt = 0.0004,
    point_coords = positions,
    steps_to_keep = 150)

lJHTDB.finalize()

np.savez('trajectories',u=u,x=x,t=t)
