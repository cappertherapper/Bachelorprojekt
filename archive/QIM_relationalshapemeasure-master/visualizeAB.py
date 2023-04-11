import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from RandomRotations import RandomRotationMatrix
from src import measure_interior, measure_exterior
from mayavi import mlab
import plotly.graph_objs as go

def Test_Cube_Integration_Interior(do_plots):

    r = np.linspace(0, 2, 100, endpoint=True)

    T1 = np.array([[0,2,3,7],
                   [0,2,6,7],
                   [0,4,6,7],
                   [0,4,5,7],
                   [0,1,5,7],
                   [0,1,3,7]], dtype=np.int32)

    T2 = np.array([[2,4,6,7],
                   [0,1,2,4],
                   [1,2,4,7],
                   [1,4,5,7],
                   [1,2,3,7]], dtype=np.int32)


    if do_plots:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        # ax_V = fig.add_subplot(211,projection='3d')
        # ax_T1 = fig.add_subplot(212)
        # ax_D = fig.add_subplot(221)        
        # ax_r = fig.add_subplot(222)

    V = np.array([[0,0,0],
                    [0,0,1],
                    [0,1,0],
                    [0,1,1],
                    [1,0,0],
                    [1,0,1],
                    [1,1,0],
                    [1,1,1]], dtype=float)
    # V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1]])

    D = V[:,0] + 0.5
    
    if do_plots:
        for i,T in enumerate([T1,T2]):
            data = []
            for e in T:
                tet_vertices = V[e]
                x, y, z = tet_vertices.T
                data.append(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=[0, 1, 2, 3], j=[1, 2, 3, 0], k=[2, 3, 0, 1],
                    opacity=0.5,
                    color='blue'
                ))

            # layout = go.Layout(

            # )

            # Create the plotly figure and show it
            fig = go.Figure(data=data)
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                scene=dict(aspectmode='data'))
            fig.update_traces(name=f'T{i+1}',visible=True,showlegend=True)
            fig.show()


    
    if do_plots:
        for i in range(2):
            R = RandomRotationMatrix()
            V = np.dot(V, R.T)
            for i,T in enumerate([T1,T2]): # plot T1 and T2
                '''
                In this example, we use the Mesh3d trace type in plotly to plot
                each tetrahedron as a single 3D shape. The i, j, and k lists 
                define the indices of the vertices for each of the four 
                triangular faces of the tetrahedron.
                '''
                data = []
                for e in T:
                    tet_vertices = V[e] #(3,4) array, 3 rækker 4 kolonner
                    x, y, z = tet_vertices.T
                    data.append(go.Mesh3d(
                        x=x, y=y, z=z,
                        i=[0, 1, 2, 3], j=[1, 2, 3, 0], k=[2, 3, 0, 1],
                        opacity=0.5,
                        color='blue'
                    ))

                layout = go.Layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    scene=dict(aspectmode='data'),
                    title=f'Plot {i+1}'
                )
                # Create the plotly figure and show it
                fig = go.Figure(data=data, layout=layout)
                fig.update_traces(name=f'T{i+1}',visible=True,showlegend=True)
                fig.show()
                ra, volume1, area1 = measure_interior(V, T, D, r, adaptive=True)
                # Define the plotly data and layout
                data1 = [
                    go.Scatter(
                        x=ra, y=volume1,
                        mode='lines',
                        line=dict(color='blue')
                    )
                ]

                layout1= go.Layout(
                    xaxis=dict(title='x'),
                    yaxis=dict(title='y'),
                    title=f'T{i+1} of Interior volume '
                )

                # Create the plotly figure and show it
                fig1 = go.Figure(data=data1, layout=layout1)
                fig1.show()
                
    
    ra, volume1, area1 = measure_interior(V, T1, D, r, adaptive=True)
    ra, volume2, area2 = measure_interior(V, T2, D, r, adaptive=True)
    
    # T1 = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [0, 3, 6, 7], [0, 1, 3, 7]])
    if do_plots:
        plt.show()

# def Test_Cube_Integration_Exterior(do_plots):

#     r = np.linspace(0, 2, 100, endpoint=True)

#     F1 = np.array([[0,1,2],
#                    [1,2,3],
#                    [0,1,4],
#                    [1,4,5],
#                    [0,2,4],
#                    [2,4,6],
#                    [1,3,5],
#                    [3,5,7],
#                    [2,3,6],
#                    [3,6,7],
#                    [4,5,6],
#                    [5,6,7]], dtype=np.int32)

#     F2 = np.array([[0,2,3],
#                    [0,1,3],
#                    [0,1,5],
#                    [0,4,5],
#                    [0,2,6],
#                    [0,4,6],
#                    [1,3,7],
#                    [1,5,7],
#                    [2,3,7],
#                    [2,6,7],
#                    [4,5,7],
#                    [4,6,7]], dtype=np.int32)

#     if do_plots:
#         fig = plt.figure()
#         ax1 = fig.add_subplot(121)
#         ax2 = fig.add_subplot(122)

#     for i in range(100):
#         V = np.array([[0,0,0],
#                       [0,0,1],
#                       [0,1,0],
#                       [0,1,1],
#                       [1,0,0],
#                       [1,0,1],
#                       [1,1,0],
#                       [1,1,1]], dtype=float)

#         D = V[:,0] + 0.5

#         R = RandomRotationMatrix()
#         V = np.dot(V, R.T)

#         ra, area1, circ1 = measure_exterior(V.copy(), F1, D.copy(), r, adaptive=True)
#         ra, area2, circ2 = measure_exterior(V.copy(), F2, D.copy(), r, adaptive=True)

#         if do_plots:
#             ax1.plot(ra, area1)
#             ax1.plot(ra, area2)

#             ax2.plot(ra, circ1)
#             ax2.plot(ra, circ2)

#     if do_plots:
#         plt.show()

def Test_Random_Integrations(do_plots):

    np.random.seed(42)

    N = 25    # num vertices
    M = N//3  # num faces / tetrahedra
    R = 15    # max r value (distance) to measure
    K = 100   # num reading points

    ## EXTERIOR TEST

    V = np.random.uniform(0, 10, (N, 3))
    F = np.random.randint(0, N, (M, 3))
    D = np.random.uniform(0, 10, N)
    r = np.linspace(0, R, K)

    if do_plots:
        data = []
        for e in F:
            tet_vertices = V[e]
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
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(aspectmode='data'))
        fig.update_traces(name='EXT-F',visible=True,showlegend=True)
        fig.show()
            
            
    _, mu10, mu11 = measure_exterior(V, F, D, r)

    r_adapt, mu10_adapt, mu11_adapt = measure_exterior(V, F, D, r, adaptive=True)

    # if do_plots:
    #     plt.subplot(1,2,1)
    #     plt.title('volume')
    #     plt.plot(r, mu10, label='fixed r')
    #     plt.plot(r_adapt, mu10_adapt, label='adaptive r')
    #     plt.legend()

    #     plt.subplot(1,2,2)
    #     plt.title('inner surface cut')
    #     plt.plot(r, mu11, label='fixed r')
    #     plt.plot(r_adapt, mu11_adapt, label='adaptive r')
    #     plt.legend()

        #plt.show()

    ## INTERIOR TEST

    V = np.random.uniform(0, 10, (N, 3))
    T = np.random.randint(0, N, (M, 4))
    D = np.random.uniform(0, 10, N)
    r = np.linspace(0, R, K)
    
    if do_plots:
        data = []
        for e in F:
            tet_vertices = V[e]
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
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(aspectmode='data'))
        fig.update_traces(name='INT-F',visible=True,showlegend=True)
        fig.show()

    _, mu00, mu01 = measure_interior(V, T, D, r)

    r_adapt, mu00_adapt, mu01_adapt = measure_interior(V, T, D, r, adaptive=True)

    if do_plots:
        plt.subplot(1,2,1)
        plt.title('volume')
        plt.plot(r, mu00, label='fixed r')
        plt.plot(r_adapt, mu00_adapt, label='adaptive r')
        plt.legend()

        plt.subplot(1,2,2)
        plt.title('inner surface cut')
        plt.plot(r, mu01, label='fixed r')
        plt.plot(r_adapt, mu01_adapt, label='adaptive r')
        plt.legend()

        plt.show()

if __name__ == '__main__':
    do_plots = True

    import time

    t0 = time.time()
    #Test_Cube_Integration_Interior(do_plots)
    # Test_Cube_Integration_Exterior(do_plots)
    Test_Random_Integrations(do_plots)
    t1 = time.time()
    print(t1 - t0, 's')