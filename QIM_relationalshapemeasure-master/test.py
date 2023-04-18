import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from RandomRotations import RandomRotationMatrix
from src import measure_interior, measure_exterior
from mayavi import mlab
import plotly.graph_objs as go


r = np.linspace(0, 2, 100, endpoint=True)

mesh = np.array([#[0,3,1,4],
                 #[0,3,2,6]], dtype=np.int32)
                 [0, 1, 2, 3],
                 [1, 2, 3, 0],
                 [2, 3, 0, 1]], dtype=np.int32)

vertices = np.array([[0,0,0],
                [0,0,1],
                [0,1,0],
                [1,0,0],
                [1,1,0],
                [1,0,1],
                [0,1,1],
                [1,1,1]], dtype=float)

data = []
for e in mesh:
    tet_vertices = vertices[e]
    x, y, z = tet_vertices.T
    data.append(go.Mesh3d(
        x=x, y=y, z=z,
        #i=[0, 1, 2, 3], j=[1, 2, 3, 0], k=[2, 3, 0, 1],
        opacity=0.5,
        color='blue'
    ))

# layout = go.Layout(

# )

# Create the plotly figure and show it
fig = go.Figure(data=data)
fig.update_layout(
    #margin=dict(l=0, r=0, t=0, b=0),
    scene=dict(aspectmode='data'))
fig.update_traces(name='INT-F',visible=True,showlegend=True)
fig.show()


D = vertices[:,0] + 0.5

print(D.shape)
ra, volume1, area1 = measure_interior(vertices, mesh, D, r, adaptive=True)


fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


ax1.plot(ra, volume1, label='type 1')
ax2.plot(ra, area1, label='type 2')

plt.show()

