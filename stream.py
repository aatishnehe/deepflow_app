import io

import streamlit as st
import time
import math
import trimesh
from statistics import mean
from utility_functions import *
import plotly.figure_factory as ff
import predict



# [0] pressure output vert profile
# [1] velocity X output vert profile
# [2] velocity Z output vert profile
# [3] pressure output hor profile
# [4] velocity X output hor profile
# [5] velocity Z output hor profile

def genQuiverPlot(vx, vy, freq):
    filter_freq = freq

    res = int(128 / filter_freq)

    filtered_vx = np.empty((res, res))
    filtered_vy = np.empty((res, res))

    counter_x = -1
    counter_y = -1

    for j in range(128):
        if j % filter_freq == 0:
            for i in range(128):
                if i % filter_freq == 0:
                    filtered_vx[counter_x][counter_y] = vx[j][i]
                    filtered_vy[counter_x][counter_y] = vy[j][i]
                    counter_y += 1
            counter_y = -1
            counter_x += 1

    max_x = np.max(filtered_vx)
    min_x = np.min(filtered_vx)

    max_y = np.max(filtered_vy)
    min_y = np.min(filtered_vy)

    x_axis = np.linspace(min_x, max_x, res)
    y_axis = np.linspace(min_y, max_y, res)

    X, Y = np.meshgrid(x_axis, y_axis)

    fig1 = ff.create_quiver(x_axis, Y[0], filtered_vx[0], filtered_vy[0], arrow_scale=.05, line=dict(color="#0000ff"))

    for i in range(res - 1):
        #figname = 'fig_' + str(i)
        fig = ff.create_quiver(x_axis, Y[i + 1], filtered_vx[i + 1], filtered_vy[i + 1], arrow_scale=.05,
                               line=dict(color="#0000ff"))
        fig1.add_traces(data=fig.data)
        fig1.update(layout_showlegend=False)

    return fig1.data


def genPlotlyResult(stl_mesh_building, ptX, ptY, ptZ, height, vel, nn_res, p_v_toggle, planeToggle, op, s, op_b):

    #np.savez_compressed('D:\\DeepFlow_local\\DeepFlowPrediction\\train\\saved_results_web.npz', a=np.nan_to_num(nn_res))

    p_vert = nn_res[0, :, :]
    v_vert = nn_res[1, :, :]
    p_hor = nn_res[2, :, :]
    v_hor = nn_res[3, :, :]
    #v_hor_y = nn_res[4, :, :]
    #v_vert_y = nn_res[5, :, :]

    #p_vert = p_vert.transpose()
    #v_vert = v_vert.transpose()
    #p_hor = p_hor.transpose()
    #v_hor = v_hor.transpose()
    #v_hor_y = v_hor_y.transpose()
    #v_vert_y = v_vert_y.transpose()

    vertices, I, J, K = stl2mesh3d(stl_mesh_building)
    x, y, z = vertices.T
    minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(stl_mesh_building)
    # Plot Building
    mesh3D = genBuilding(x, y, z, I, J, K, op_b, 'rgb(64,64,64)')


    # Plot wind direction vector
    vector, cone = genArrow(minx, maxx, miny, maxy, minz, maxz, vel, windProfileScaleFactor)

    windPoints = np.arange(0, np.max(ptZ), 0.5)
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
    Z = np.zeros(ptX.size*ptY.size)
    #st.write('Height: ', height)
    Y0 = np.full(ptX.size, height[0])
    for p, item in enumerate(Z):
        Z[p] = height[1]
    #st.write(nn_res[0, :, :])
    #horizontales Profil
    if p_v_toggle == 1:
        res_nn_h = go.Surface(x=ptX, y=ptY, z=Z.reshape(ptX.size, ptY.size), surfacecolor=p_hor, opacity=op)
    elif p_v_toggle == 2:
        res_nn_h = go.Surface(x=ptX, y=ptY, z=Z.reshape(ptX.size, ptY.size), surfacecolor=v_hor, opacity=op)
    #st.write("horizontal", nn_res[1])
    #vertikales Profil
    Zp = np.tile(ptZ, (1, ptX.size)).reshape(ptX.size, ptY.size)

    #st.write(ptX, Y0)
    #st.write(Zp)

    if p_v_toggle == 1:
        res_nn_v = go.Surface(x=ptX, y=Y0, z=np.transpose(Zp), surfacecolor=p_vert, opacity=op)
    elif p_v_toggle == 2:
        res_nn_v = go.Surface(x=ptX, y=Y0, z=np.transpose(Zp), surfacecolor=v_vert, opacity=op)
    #st.write("vertikal", nn_res[0])

    title = ""
    layout = go.Layout(paper_bgcolor='lightgray',
                       title_text=title, title_x=0.5,
                       font_color='white',
                       width=1200,
                       scene_camera=dict(eye=dict(x=-1., y=-1.8, z=1)),
                       scene_xaxis_visible=False,
                       scene_yaxis_visible=False,
                       scene_zaxis_visible=True,
                       )

    if planeToggle == 1:
        plotData = [mesh3D, vector, cone, res_nn_v, windProfile]
    elif planeToggle == 2:
        plotData = [mesh3D, vector, cone, res_nn_h, windProfile]
    elif planeToggle == 3:
        plotData = [mesh3D, vector, cone, res_nn_h, res_nn_v, windProfile]

    figure = go.Figure(data=plotData, layout=layout)
    figure.data[0].update(lighting=dict(ambient=0.18,
                                     diffuse=1,
                                     fresnel=.1,
                                     specular=1,
                                     roughness=.1,
                                     facenormalsepsilon=0))
    figure.data[0].update(lightposition=dict(x=3000,
                                          y=3000,
                                          z=10000))
    #figure.data[3].update_traces(visible="False")
    figure.update_scenes(aspectmode="data")

    return figure


# @st.cache
def preProcessingData(trimeshData):
    st.session_state.loader = True
    # get the bounding box
    try:
        boundingBox = trimeshData.bounds
        #st.write(boundingBox)

    except:
        st.write("No mesh input loaded! Aborting...")
        st.stop()
    # calculate the centre of the bounding box in y-dir
    yCentre = mean(boundingBox[:, 1])
    #st.write("bbox center in y-dir", yCentre)
    # scale factors
    sx = 3.
    sy = 2.
    sz = 1.5
    ScaleMatrix = np.array([[sx, 0., 0.], [0., sy, 0.], [0., 0., sz]])
    # scale the bounding box
    boundingBox = np.transpose(np.dot(ScaleMatrix, np.transpose(boundingBox)))
    #st.write(boundingBox)
    # move the bounding box
    # get dim in x for translation of bounding box
    bbDimx = abs(boundingBox[0, 0]) + abs(boundingBox[1, 0])
    boundingBox = np.add(boundingBox, np.array([[0.2 * bbDimx, -yCentre, 0], [0.2 * bbDimx, -yCentre, 0]]))
    #st.write(boundingBox)
    bbox_centres = np.zeros(3)
    for i in range(3):
        bbox_centres[i] = np.mean(boundingBox[:, i])
    #st.write("bbox center", bbox_centres)
    st.session_state.cutH = [bbox_centres[1], bbox_centres[2] / sz]
    #st.write(st.session_state.cutH)


    # get enlarged bounding box edge lengths
    l = np.array([abs(boundingBox[1, 0] - boundingBox[0, 0]), abs(boundingBox[1, 1] - boundingBox[0, 1]),
                  abs(boundingBox[1, 2] - boundingBox[0, 2])])
    pointsX = np.arange(boundingBox[0, 0], boundingBox[1, 0], l[0] / N)
    pointsY = np.arange(boundingBox[0, 1], boundingBox[1, 1], l[1] / N)
    pointsZ = np.arange(boundingBox[0, 2], boundingBox[1, 2], l[2] / N)

    st.session_state.ptx = pointsX
    st.session_state.pty = pointsY
    st.session_state.ptz = pointsZ

    #st.write(pointsZ)
    ptCloudH = []
    ptCloudV = []

    for pt_x in pointsX:
        for pt_y in pointsY:
            ptCloudH.append([pt_x, pt_y, bbox_centres[2] / sz])
    ptCloudH = np.array(ptCloudH)
    #st.write(ptCloudH.shape)

    #st.write(trimeshData.contains(ptCloudH[pt].reshape(1, 3)))

    for pt_x in pointsX:
        for pt_z in pointsZ:
            ptCloudV.append([pt_x, bbox_centres[1], pt_z + 0.5])
    ptCloudV = np.array(ptCloudV)

    #st.write(len(ptCloudV))

    # input layout channels for Deep Flow:
    # [0] freestream field X + boundary vert profile
    # [1] freestream field X + boundary hor profile
    # [2] binary mask for boundary in Z-dir vert profile
    # [3] binary mask for boundary in Y-dir hor profile

    input_df = np.zeros((4, N, N))

    #st.write(N)

    curIndex = 0
    # horizontal pt Cloud
    for x in range(N):

        for y in range(N):
            if trimeshData.contains(ptCloudH[curIndex].reshape(1, 3)):
                # fill bin mask
                input_df[3][y][x] = 1.
            else:
                input_df[3][y][x] = 0.
                # fill mask
                val = atmFreeStreamX(avVel, bbox_centres[2] / sz)
                input_df[1][y][x] = val
            curIndex += 1
            #st.write(curIndex)

    #st.write(curIndex)
    # vertical pt Cloud
    curIndex = 0
    for x in range(N):
        for z in range(N):
            if trimeshData.contains(ptCloudV[curIndex].reshape(1, 3)):
                # fill bin mask
                input_df[2][z][x] = 1.
            else:
                input_df[2][z][x] = 0.
                # fill mask
                input_df[0][z][x] = atmFreeStreamX(avVel, ptCloudV[curIndex][2])
            curIndex += 1


    return input_df


def normalisation(data):

    max_inputs_0 = np.max(np.abs(data[0]))
    max_inputs_1 = np.max(np.abs(data[1]))
    max_inputs_2 = np.max(np.abs(data[2]))
    max_inputs_3 = np.max(np.abs(data[3]))

    if max_inputs_0 != 0.0:
        data[0, :, :] *= (1.0 / max_inputs_0)
    if max_inputs_1 != 0.0:
        data[1, :, :] *= (1.0 / max_inputs_1)
    if max_inputs_2 != 0.0:
        data[2, :, :] *= (1.0 / max_inputs_2)
    if max_inputs_3 != 0.0:
        data[3, :, :] *= (1.0 / max_inputs_3)

    return data


# @st.cache
def strFLOWPrediction_ver(DF):

    res_pressure = predict.predict_ver_pressure(DF)
    res_velocity = predict.predict_ver_vel(DF)
    np.copyto(inputArray, DF)

    return [res_pressure.detach().numpy(), res_velocity.detach().numpy()]

def strFLOWPrediction_hor(DF):

    res_pressure = predict.predict_hor_pressure(DF)
    res_velocity = predict.predict_hor_vel(DF)
    np.copyto(inputArray, DF)

    return [res_pressure.detach().numpy(), res_velocity.detach().numpy()]


def denormalise_velocity(DF, v_norm):
    denormalised_df = DF.copy()

    max_v1 = np.max(np.abs(denormalised_df[0, :, :]))
    max_v2 = np.max(np.abs(denormalised_df[1, :, :]))

    # a[0, :, :] /= (1.0/self.max_targets_0_p)
    denormalised_df[0, :, :] /= (1.0 / max_v1)
    denormalised_df[1, :, :] /= (1.0 / max_v2)

    # a[0, :, :] *= v_norm ** 2
    denormalised_df[0, :, :] *= v_norm
    denormalised_df[1, :, :] *= v_norm

    #st.write('denormalised_df shape: ', denormalised_df.shape)

    return denormalised_df


def denormalise_pressure(DF, v_norm):
    denormalised_df = DF.copy()

    max_p = np.max(np.abs(denormalised_df[0, :, :]))

    denormalised_df[0, :, :] /= (1.0/max_p)

    denormalised_df[0, :, :] *= v_norm ** 2

    denormalised_df[0, :, :] += np.mean(denormalised_df[0, :, :])

    #st.write('denormalised_df shape: ', denormalised_df.shape)

    return denormalised_df


# PAGE cfig
st.set_page_config(
    page_title="str.FLOWer",
    page_icon="🌬"
)

# Containers
header = st.container()
user_input = st.container()
building_vis = st.container()
compute = st.container()
viewer = st.container()

# Global variables

N = 128
windProfileScaleFactor = 0.5

#prediction = np.empty((6, N, N))
inputArray = np.empty((2, N, N))
#ptx = np.empty(N)
#pty = np.empty(N)
#ptz = np.empty(N)
#cutH = np.empty(2)

if 'prediction' not in st.session_state:
    st.session_state.prediction = np.empty((6, N, N))
if 'predicted' not in st.session_state:
    st.session_state.predicted = False
if 'count' not in st.session_state:
    st.session_state.count = 0
if 'calcTime' not in st.session_state:
    st.session_state.calcTime = 0.0
if 'ptx' not in st.session_state:
    st.session_state.ptx = np.empty(N)
if 'pty' not in st.session_state:
    st.session_state.pty = np.empty(N)
if 'ptz' not in st.session_state:
    st.session_state.ptz = np.empty(N)
if 'cutH' not in st.session_state:
    st.session_state.cutH = np.empty(2)

result = False
# header
#page header
with header:
    col1, col2 = st.columns([3, 1])
    with col1:
    #st.image('https://www.clipartmax.com/png/middle/319-3190374_big-image-vector-free-dandelion-icon-transparent.png', width=60)
        #st.write('Current Dir: ', os.getcwd())
        f = open("./Images/Logo_one3d_by_structure_rgb.svg", "r")
        lines = f.readlines()
        line_string = ''.join(lines)
        st.write(render_svg(line_string), unsafe_allow_html=True)

    with col2:
        f = open("./Images/FLOWer_dandi.svg", "r")
        lines = f.readlines()
        line_string = ''.join(lines)
        st.write(render_svg(line_string), unsafe_allow_html=True)
        #st.image('./app/logo_dandi.png', width=120)

    st.subheader("Preliminaries")
    #about app
with header.expander("What is str.FLOWer?", expanded=False):
    st.markdown("""
    This is v.alpha of the **steady wind flow generator** based on an artificial neural network and so on ...
    ...
    Thanks to
    ...
    """)

with header.expander("CNN Model and Training Data", expanded=False):
    st.markdown("""
    The CNN Network is similar to Thurey et al. 2020 ... Models are...
    """)
# input
with user_input:

    st.subheader("Inputs")
    with st.form("Geometry"):
        build_mesh = st.file_uploader(label="Place your building as a closed *stl mesh here", type='stl')

        col1, col2 = st.columns([1,2])
        with col1:
            avVel = st.slider(label="Define the wind velocity at 10m above ground", min_value=10., max_value=40., step=1.,
                          value=25.)
        # Input rotation angle
        with col2:
            rotAngle = st.slider(
                label="Define the wind flow direction by rotating your building w.r.t the blue arrow (wind flow direction)",
                min_value=-180., max_value=180., step=1., value=0.)
        # Define rot Matrix and rotate stl
        rotMatrix = stl.mesh.Mesh.rotation_matrix([0.0, 0.0, 0.5], math.radians(rotAngle))
        submitted = st.form_submit_button("Submit")
    if build_mesh is not None:
        bytes_data = io.BytesIO(build_mesh.getvalue())
        # Building as numpy-stl
        building = stl.mesh.Mesh.from_file("tmp.stl", fh=bytes_data)

    # Input the basis vel. at 10m hight.


    if build_mesh is not None:
        building.rotate_using_matrix(rotMatrix)
        volume, cog, inertia = building.get_mass_properties()
        building.translate([-1.*cog[0], -1.*cog[1], 0])
        #save stl for trimesh and import to trimesh for easy pt checking
        building.save("stl_tmp.stl")
        trimeshdata = trimesh.load_mesh("stl_tmp.stl", file_type='stl')
        #st.write(trimeshdata)


# visualisation
with building_vis:
    if build_mesh is not None:

        # Load the STL files and add the vectors to the plot
        fig = genPlotlyMesh(building, build_mesh.name, avVel, windProfileScaleFactor)

        # stream the plotly chart
        st.plotly_chart(fig, use_container_width=True)
        if trimeshdata.is_watertight:
            st.write('Nice, your mesh is watertight. That should work fine!')
        else:
            st.write('Oh, your mesh is not watertight. This will cause problems. Please repair the mesh and come back later!')
            st.stop()

# Compute button
with compute:
    if build_mesh is not None:
        st.subheader("Prediction")
        result = st.button("Update Prediction using str.FLOWer")

    #result_str= st.select_slider("Slide to predict the steady wind flow field using str.FLOWer", options=["Wait...", "Do it!"])
    #if result_str == "Wait...":
    #    result = False
    #elif result_str == "Do it!":
    #    result = True

    #
    # generate point clouds NxN for Deep Flow evaluation

    oldSessionStateCount = st.session_state.count

    if result:
        st.session_state.count += 1
        #st.write(st.session_state.count)


    if not st.session_state.predicted or (st.session_state.count - oldSessionStateCount) != 0:

        with st.spinner('Processing...'):

            tic = time.perf_counter()
            #st.write(trimeshdata)
            #inputDF = preProcessingData(trimeshdata)
            try:
                inputDF = preProcessingData(trimeshdata)
                #np.savez_compressed('D:\\DeepFlow_local\\DeepFlowPrediction\\train\\preprocessed_input.npz', a=np.nan_to_num(inputDF))
                #st.write('Input Dimensions: ', inputDF.shape)
            except:
                st.write("You did not upload a geometry... Aborting!")
                st.stop()
            #toc = time.perf_counter()

            # st.write(st.session_state.prediction)
            # NN to go

            #tic2 = time.perf_counter()

            detachedPrediction = np.zeros((6, N, N))

            inputDF = normalisation(inputDF)
            #st.write('Vertical cuts dim: ', inputDF[[0, 2], :, :].shape)

            [detachedPrediction_hor_pressure, detachedPrediction_hor_velocity] = strFLOWPrediction_hor(inputDF[[1,3], :, :])
            [detachedPrediction_ver_pressure, detachedPrediction_ver_velocity] = strFLOWPrediction_ver(inputDF[[0,2], :, :])

            #st.write('Prediction Dims 1 : ', detachedPrediction_hor_velocity.shape)

            detachedPrediction[0] = detachedPrediction_ver_pressure
            detachedPrediction[1] = detachedPrediction_ver_velocity[0][0]
            detachedPrediction[2] = detachedPrediction_hor_pressure
            detachedPrediction[3] = detachedPrediction_hor_velocity[0][0]

            v_norm = (np.max(np.abs(detachedPrediction[0, :, :])) ** 2 +
                      np.max(np.abs(detachedPrediction[1, :, :])) ** 2 +
                      np.max(np.abs(detachedPrediction[2, :, :])) ** 2 +
                      np.max(np.abs(detachedPrediction[3, :, :])) ** 2) ** 0.5

            detachedPrediction[[1, 3]] = denormalise_velocity(detachedPrediction[[1,3]], v_norm)
            detachedPrediction[[0, 2]] = denormalise_velocity(detachedPrediction[[0, 2]], v_norm)


            #st.write('Prediction Dims: ', detachedPrediction.shape)

            st.session_state.prediction = detachedPrediction
            #st.write(st.session_state.prediction)
            #np.copyto(prediction, detachedPrediction)

            toc = time.perf_counter()
            st.session_state.calcTime = toc - tic
            st.session_state.predicted = True

if 'vChecker' not in st.session_state:
    st.session_state.vChecker = True
if 'pChecker' not in st.session_state:
    st.session_state.pChecker = False


def ClickpChecker():
    if not st.session_state.pChecker:
        st.session_state.pChecker = True
    elif st.session_state.pChecker:
        st.session_state.pChecker = False



with viewer:
    #st.write("Calculation finished.")
    st.write("""Job done. Preprocessing the input data took and evaluating the artificial neurol network took %0.4s seconds.""" % st.session_state.calcTime)
    if st.session_state.predicted:
        st.subheader("Steady wind flow visualization")
        with st.form("Parameters"):
            col1, col2 = st.columns([1, 2])
            with col1:
                varSlider = st.select_slider("Visualise the ...", options=['pressure field', 'velocity field'])
            with col2:
                cutPlanesVis = st.multiselect("Select horizontal and/or vertical cut planes",
                                          ['vertical cutplane', 'horizontal cutplane'], ['vertical cutplane'])
                opacity = st.slider(label="Plane opacity", min_value=0., max_value=100., step=5., value=100.)
                opacity_building = st.slider(label="Building opacity", min_value=0., max_value=100., step=5., value=100.)
            submitted = st.form_submit_button("Submit")
        cutPlaneToggle = 0
        if 'vertical cutplane' in cutPlanesVis and 'horizontal cutplane' in cutPlanesVis:
            cutPlaneToggle = 3
        elif 'vertical cutplane' in cutPlanesVis:
            cutPlaneToggle = 1
        elif 'horizontal cutplane' in cutPlanesVis:
            cutPlaneToggle = 2
        if cutPlaneToggle == 0:
            st.write('At least one cutplane needs to be selected... Aborting!')
            st.stop()

        if varSlider == 'pressure field':
            varSliderToggle = 1
        elif varSlider == 'velocity field':
            varSliderToggle = 2


        figResult = genPlotlyResult(building, st.session_state.ptx, st.session_state.pty, st.session_state.ptz,
                                    st.session_state.cutH, avVel, st.session_state.prediction, varSliderToggle,
                                    cutPlaneToggle, opacity/100., windProfileScaleFactor, opacity_building/100.)

            # stream the plotly chart

        st.plotly_chart(figResult, use_container_width=True)

        #fig = genQuiverPlot(st.session_state.prediction, freq = 6)

        #st.plotly_chart(fig, use_container_width=True)


def main():
    # Your Streamlit app code goes here

    # Add company contact info at the bottom
    st.markdown("---")
    st.markdown("## Contact Information")
    st.write("For any inquiries, please contact:")
    st.markdown("*str.ucture GmbH*: Lightweight Design. Made in Stuttgart. ")
    st.markdown("Lindenspürstr. 32")
    st.markdown('70176 Stuttgart')
    st.markdown("Germany")
    st.markdown("Email: [info@str-ucture.com](mailto:info@str-ucture.com)")
    st.markdown("Phone: [+49 (0)711 286937-0](tel:+49 (0)711 286937-0)")

main()

