import base64

import plotly.graph_objects as go
#from stl import mesh
import stl
import numpy as np


def atmFreeStreamX(avVel, z):
    # re-compute free-stream vel in cutplanes vgl. atmBoundaryLayerInletVelocity
    # https://www.openfoam.com/documentation/guides/latest/doc/guide-bcs-inlet-atm-atmBoundaryLayerInletVelocity.html
    # vertikale Cutplane in internalProbes_1
    # Zref    10.0;
    if z > 0.01:
        vX = avVel * np.log((z - 0.1) / 0.1) / np.log(10 / .01)
    else:
        vX = 0.0
    return vX


# find the max dimensions, so we can know the bounding box, getting the height,
# width, length (because these are the step size)...
def find_mins_maxs(obj):
    minx = maxx = miny = maxy = minz = maxz = None
    for p in obj.points:
        # p contains (x, y, z)
        if minx is None:
            minx = p[stl.Dimension.X]
            maxx = p[stl.Dimension.X]
            miny = p[stl.Dimension.Y]
            maxy = p[stl.Dimension.Y]
            minz = p[stl.Dimension.Z]
            maxz = p[stl.Dimension.Z]
        else:
            maxx = max(p[stl.Dimension.X], maxx)
            minx = min(p[stl.Dimension.X], minx)
            maxy = max(p[stl.Dimension.Y], maxy)
            miny = min(p[stl.Dimension.Y], miny)
            maxz = max(p[stl.Dimension.Z], maxz)
            minz = min(p[stl.Dimension.Z], minz)
    return minx, maxx, miny, maxy, minz, maxz


def stl2mesh3d(stl_mesh):
    # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles (i.e. three 3d points)
    # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
    p, q, r = stl_mesh.vectors.shape #(p, 3, 3)
    # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
    # extract unique vertices from all mesh triangles
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(p*q, r), return_inverse=True, axis=0)
    I = np.take(ixr, [3*k for k in range(p)])
    J = np.take(ixr, [3*k+1 for k in range(p)])
    K = np.take(ixr, [3*k+2 for k in range(p)])
    return vertices, I, J, K


def genArrow(minx, maxx, miny, maxy, minz, maxz,vel,s):
    vector = go.Scatter3d(x=[minx-(abs(minx)+abs(maxx))*2, minx-(abs(minx)+abs(maxx))*2+s*atmFreeStreamX(vel,10)],
                          y=[.5*(maxy+miny), .5*(maxy+miny)],
                          z=[10., 10.],
                          marker=dict(size=1,color='rgb(0,176,240)'),
                          line=dict(color='rgb(0,176,240)',
                                    width=6), showlegend=False, name = "Wind speed at 10m"
                          )
    cone = go.Cone(x=[minx-(abs(minx)+abs(maxx))*2+s*atmFreeStreamX(vel,10)], y=[.5*(maxy+miny)], z=[10.],
                   u=[vel/5], v=[0], w=[0], anchor="tip", colorscale=[[0, 'rgb(0,176,240)'], [1, 'rgb(0,176,240)']], showscale=False)

    return vector, cone


def genBuilding(x,y,z,I,J,K, opa, rgb):
    colorscale = [[0, rgb], [1, rgb]]
    mesh3D = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=I,
        j=J,
        k=K,
        flatshading=True,
        colorscale=colorscale,
        intensity=z,
        name='building',
        showscale=False,
        opacity=opa)
    return mesh3D


def genPlotlyMesh(stl_mesh_building, name, vel, s):
    vertices, I, J, K = stl2mesh3d(stl_mesh_building)
    x, y, z = vertices.T
    # find min and max from building dimensions
    minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(stl_mesh_building)
    # Plot Building
    mesh3D = genBuilding(x,y,z,I,J,K, 1.0, 'rgb(232,65,119)')

    # Plot wind direction vector
    vector, cone = genArrow(minx, maxx, miny, maxy, minz, maxz, vel, s)

    title = "%s" %str(name)
    layout = go.Layout(paper_bgcolor='lightgray',
                       title_text=title, title_x=0.4,
                       font_color='white',
                       width=1200,
                       scene_camera=dict(eye=dict(x=-1., y=-1.8, z=1)),
                       scene_xaxis_visible=False,
                       scene_yaxis_visible=False,
                       scene_zaxis_visible=True,
                       )
    windPoints = np.arange(0, maxz+10, 0.5)
    wx = np.zeros(len(windPoints))
    wy = np.zeros(len(windPoints))

    for i in range(len(windPoints)):
        wy[i] = .5*(maxy+miny)
        wx[i] = (minx-(abs(minx)+abs(maxx))*2) + s*atmFreeStreamX(vel, windPoints[i])
    windPoints=np.append(windPoints, windPoints[-1])
    windPoints=np.append(windPoints, windPoints[0])
    wx=np.append(wx, wx[0])
    wx=np.append(wx, wx[0])
    wy=np.append(wy, wy[0])
    wy=np.append(wy, wy[0])

    windProfile = go.Scatter3d(x=wx, y=wy, z=windPoints, marker=dict(size=1,color='rgb(0,176,240)'), line=dict(color='rgb(0,176,240)',
                                    width=3), name="ABL" )
    fig2 = go.Figure(data=[mesh3D, vector, cone, windProfile], layout=layout)
    fig2.data[0].update(lighting=dict(ambient=0.18,
                                     diffuse=1,
                                     fresnel=.1,
                                     specular=1,
                                     roughness=.1,
                                     facenormalsepsilon=0))
    fig2.data[0].update(lightposition=dict(x=3000,
                                          y=3000,
                                          z=10000))
    fig2.update_scenes(aspectmode="data")

    return fig2


def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img width="150" height="150" src="data:image/svg+xml;base64,%s"/>' % b64
    return html


def find_absmax(data, use_targets, x):
    maxval = 0
    for i in range(data.totalLength):
        if use_targets == 0:
            temp_tensor = data.inputs[i]
        else:
            temp_tensor = data.targets[i]
        temp_max = np.max(np.abs(temp_tensor[x]))
        if temp_max > maxval:
            maxval = temp_max
    return maxval
