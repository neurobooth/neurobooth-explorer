import neurobooth_terra

import psycopg2
from sshtunnel import SSHTunnelForwarder

import dash_auth
import configparser

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash import callback_context
import plotly.graph_objects as go
import dash_table

import numpy as np
import pandas as pd

import os.path as op
import glob

from h5io import read_hdf5

from datetime import datetime
from datetime import date

from scipy import signal


# ###########################################
# ### Comment out section depending on os ###
# ###########################################

### WINDOWS Legion ###

# --- ssh, db cred, variable assignment for Windows 11 on Legion --- #

### setting data file locations ###
file_loc = 'C:\\Users\\siddh\\Desktop\\lab_projects\\Neurobooth_Explorer\\data'
face_landmark_filename = 'C:\\Users\\siddh\\Desktop\\repos\\neurobooth-explorer\\facial_landmark_file\\100001_2022-02-28_08h-55m-00s_passage_obs_1_R001-FLIR_blackfly_1-FLIR_rgb_1_face_landmarks.hdf5'
###

auth_config_file_loc = 'C:\\Users\\siddh\\.db_secrets\\users.txt'
auth_config = configparser.ConfigParser()
auth_config.read(auth_config_file_loc)

USERNAME_PASSWORD_PAIRS = dict()
for ky in auth_config[auth_config.sections()[0]]:
    USERNAME_PASSWORD_PAIRS[ky] = auth_config[auth_config.sections()[0]][ky]

db_config_file_loc = 'C:\\Users\\siddh\\.db_secrets\\db_secrets.txt'
config = configparser.ConfigParser()
config.read(db_config_file_loc)

# Setting db access args #
ssh_args = dict(
        ssh_address_or_host=config['windows']['ssh_address_or_host'],
        ssh_username=config['windows']['ssh_username'],
        host_pkey_directories=config['windows']['host_pkey_directories'],
        remote_bind_address=(config['windows']['remote_bind_address'], int(config['windows']['remote_bind_address_port'])),
        local_bind_address=(config['windows']['local_bind_address'], int(config['windows']['local_bind_address_port'])),
        allow_agent=False
)

db_args = dict(
    database=config['windows']['database'], user=config['windows']['user'], password=config['windows']['password'],
    # host='localhost'
)


# ### LINUX P620 and Neurodoor ###

# # --- ssh, db cred, variable assignment for Ubuntu on P620 Workstation --- #

# ### setting data file locations ###
# file_loc = '/home/sid/data'
# face_landmark_filename = '/home/sid/Desktop/repos/neurobooth-explorer/facial_landmark_file/100001_2022-02-28_08h-55m-00s_passage_obs_1_R001-FLIR_blackfly_1-FLIR_rgb_1_face_landmarks.hdf5'
# auth_config_file_loc = '/home/sid/.db_secrets/users.txt'
# db_config_file_loc = '/home/sid/.db_secrets/db_secrets.txt'
# ###

# ### setting data file locations for Neurodoor ###
# file_loc = '/autofs/nas/neurobooth/data'
# face_landmark_filename = '/homes/5/sp1022/repos/neurobooth-explorer/facial_landmark_file/100001_2022-02-28_08h-55m-00s_passage_obs_1_R001-FLIR_blackfly_1-FLIR_rgb_1_face_landmarks.hdf5'
# auth_config_file_loc = '/homes/5/sp1022/.db_secrets/users.txt'
# db_config_file_loc = '/homes/5/sp1022/.db_secrets/db_secrets.txt'
# ###

# auth_config = configparser.ConfigParser()
# auth_config.read(auth_config_file_loc)

# USERNAME_PASSWORD_PAIRS = dict()
# for ky in auth_config[auth_config.sections()[0]]:
#     USERNAME_PASSWORD_PAIRS[ky] = auth_config[auth_config.sections()[0]][ky]

# config = configparser.ConfigParser()
# config.read(db_config_file_loc)

# # Setting db access args #
# ssh_args = dict(
#         ssh_address_or_host=config['linux']['ssh_address_or_host'],
#         ssh_username=config['linux']['ssh_username'],
#         ssh_pkey=config['linux']['ssh_pkey'],
#         remote_bind_address=(config['linux']['remote_bind_address'], int(config['linux']['remote_bind_address_port'])),
#         local_bind_address=(config['linux']['local_bind_address'], int(config['linux']['local_bind_address_port'])),
#         allow_agent=False
# )

# db_args = dict(
#     database=config['linux']['database'], user=config['linux']['user'], password=config['linux']['password'],
#     # host='localhost'
# )


# ### Control Machine ###

# # --- ssh, db cred, variable assignment for Control @ Neurobooth --- #

# from neurobooth_os.secrets_info import secrets

# ### setting data file locations ###
# file_loc = 'Z:\\data'
# face_landmark_filename = 'C:\\neurobooth\\neurobooth-explorer\\facial_landmark_file\\100001_2022-02-28_08h-55m-00s_passage_obs_1_R001-FLIR_blackfly_1-FLIR_rgb_1_face_landmarks.hdf5'
# ###

# auth_config_file_loc = 'C:\\Users\\CTR\\.db_secrets\\users.txt'
# auth_config = configparser.ConfigParser()
# auth_config.read(auth_config_file_loc)

# USERNAME_PASSWORD_PAIRS = dict()
# for ky in auth_config[auth_config.sections()[0]]:
#     USERNAME_PASSWORD_PAIRS[ky] = auth_config[auth_config.sections()[0]][ky]
# #######################


# --- Function to compute age from date of birth --- #
def calculate_age(dob):
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


###### Building Master Data Table ######

sql_query_cmd = """
SELECT subject.subject_id, subject.gender_at_birth, subject.date_of_birth, tech_obs_log.date_times, tech_obs_log.tech_obs_log_id,
       tech_obs_log.tech_obs_id, sensor_file_log.sensor_file_path
FROM ((tech_obs_log
INNER JOIN sensor_file_log ON tech_obs_log.tech_obs_log_id = sensor_file_log.tech_obs_log_id)
INNER JOIN subject ON tech_obs_log.subject_id = subject.subject_id);
"""

def rebuild_master_data_table(sql_query_cmd):
    
    # # --- Querying on control --- #
    # database = secrets['database']['dbname']
    # host = secrets['database']['host']
    # port = 5432
    # with psycopg2.connect(database=database, 
    #                         user=secrets['database']['user'],
    #                         password=secrets['database']['pass'],
    #                         host=host,
    #                         port=port) as conn:
    #     nb_data_df = neurobooth_terra.query(conn,sql_query_cmd, ['subject_id', 'gender_at_birth', 'dob', 'session_datetime', 'task_log', 'tasks', 'file_names'])
    
    # --- Querying Neurobooth Terra Database --- #
    with SSHTunnelForwarder(**ssh_args) as tunnel:
        with psycopg2.connect(port=tunnel.local_bind_port,
                            host=tunnel.local_bind_host,
                            **db_args) as conn:
            nb_data_df = neurobooth_terra.query(conn,sql_query_cmd, ['subject_id', 'gender_at_birth', 'dob', 'session_datetime', 'task_log', 'tasks', 'file_names'])

    nb_data_df.dropna(inplace=True)
    nb_data_df['age'] = nb_data_df.dob.apply(calculate_age)
    nb_data_df['session_date'] = [i[0].date() for i in nb_data_df.session_datetime]
    nb_data_df['gender'] = ['M' if i=='1.0' else 'F' for i in nb_data_df.gender_at_birth]
    col_reorder = ['subject_id', 'gender', 'dob', 'age', 'task_log', 'session_date', 'session_datetime', 'tasks', 'file_names']
    nb_data_df = nb_data_df[col_reorder]

    # --- Generating dropdown lists --- #
    sub_id_list = [str(j) for j in np.sort([int(i) for i in nb_data_df.subject_id.unique()])]
    session_date_list = [str(i) for i in np.sort(nb_data_df.session_date.unique())]
    task_list = nb_data_df.tasks.unique()
    clinical_list = ['Ataxia-Telangiectasia','Spino Cerebellar Ataxia', 'Parkinsonism']

    return nb_data_df, sub_id_list, session_date_list, task_list, clinical_list

# Fetching data at app launch
nb_data_df, sub_id_list, session_date_list, task_list, clinical_list = rebuild_master_data_table(sql_query_cmd)


# --- Function to extract all file list from dataframe --- #
def get_file_list(fdf):
    file_list=[]
    for file_array in fdf.file_names:
        for file in file_array:
            if file[-5:] == '.hdf5':
                file_list.append(file)
    return file_list


# --- Function to extract task session file list from datafarme --- #
def get_task_session_files(fdf):
    sl = fdf.subject_id.tolist() #subject list
    dtl=[] #datetime list
    for i in fdf.session_datetime:
        j = str(i[0]).replace(' ','_')
        j = j.replace(':','-')
        j = j[:13]+'h'+j[13:16]+'m'+j[16:]+'s'
        dtl.append(j)
    tl = fdf.tasks.tolist() #task list
    fnl=[] #filename list
    for i in range(len(sl)):
        fnl.append(sl[i]+'_'+dtl[i]+'_'+tl[i])
    return list(set(fnl))


# --- Function to parse task session files and return traces --- #
def parse_files(task_files):

    timeseries_data=[]
    specgram_data=[]

    if len(task_files)==0 or len(task_files)==1:
        ts_rand_trace = go.Scatter(
            x=np.arange(10),
            y=np.random.randint(0,5,size=(10)),
            name='Random Data',
            mode='lines'
            )
        timeseries_data.append(ts_rand_trace)

        random_y = np.random.randint(0,5,size=(10))
        audio_rand_trace = go.Scatter(
            x=np.arange(10),
            y=random_y,
            name='Random Data',
            mode='lines'
            )
        vert_trace = go.Scatter(
            x=[0]*10,
            y=random_y,
            name='x position',
            mode='lines'
            )
        specgram_data.append(audio_rand_trace)
        specgram_data.append(vert_trace)

        return timeseries_data, specgram_data
    
    else:
        for file in task_files:
            if 'Eyelink' in file:
                try:
                    fname = glob.glob(op.join(file_loc, file))[0]
                    fdata = read_hdf5(fname)['device_data']

                    et_df = pd.DataFrame(fdata['time_series'][:, [0,1,3,4]], columns=['R_gaze_x','R_gaze_y','L_gaze_x','L_gaze_y'])
                    et_df['timestamps'] = fdata['time_stamps']
                    #print(et_df.head(n=2))
                    et_datetime = [datetime.fromtimestamp(timestamp) for timestamp in et_df.timestamps]

                    trace1 = go.Scatter(
                                x=et_datetime,
                                y=et_df['R_gaze_x'],
                                name='Right Eye Gaze X',
                                mode='lines'
                            )
                    timeseries_data.append(trace1)
                    #print(trace1)

                    trace2 = go.Scatter(
                                x=et_datetime,
                                y=et_df['R_gaze_y'],
                                name='Right Eye Gaze Y',
                                mode='lines'
                            )
                    timeseries_data.append(trace2)
                    #print(trace2)

                    trace3 = go.Scatter(
                                x=et_datetime,
                                y=et_df['L_gaze_x'],
                                name='Left Eye Gaze X',
                                mode='lines',
                                visible='legendonly',
                            )
                    timeseries_data.append(trace3)
                    #print(trace3)

                    trace4 = go.Scatter(
                                x=et_datetime,
                                y=et_df['L_gaze_y'],
                                name='Left Eye Gaze Y',
                                mode='lines',
                                visible='legendonly',
                            )
                    timeseries_data.append(trace4)
                    #print(trace4)

                    #try:
                    mdata = read_hdf5(fname)['marker']
                    #print(mdata.keys())
                    
                    ts_ix = []
                    x_coord = []
                    y_coord = []
                    for ix, txt in enumerate(mdata['time_series']):
                        if '!V TARGET_POS' in txt[0]:
                            ts_ix.append(ix)
                            l = txt[0].split(' ')
                            x_coord.append(int(l[3][:-1]))
                            y_coord.append(int(l[4]))

                    ctrl_ts = []
                    for ix in ts_ix:
                        ctrl_ts.append(mdata['time_stamps'][ix])
                        
                    target_pos_df = pd.DataFrame()
                    target_pos_df['ctrl_ts'] = ctrl_ts
                    target_pos_df['x_pos'] = x_coord
                    target_pos_df['y_pos'] = y_coord
                    #print(target_pos_df.head(n=2))
                    target_datetime = [datetime.fromtimestamp(timestamp) for timestamp in target_pos_df.ctrl_ts]

                    target_x_trace = go.Scatter(
                                x=target_datetime,
                                y=(target_pos_df['x_pos']),
                                name='Target X',
                                line={'shape': 'hv'},
                                mode='lines',
                                visible='legendonly',
                            )
                    timeseries_data.append(target_x_trace)

                    target_y_trace = go.Scatter(
                                x=target_datetime,
                                y=(target_pos_df['y_pos']),
                                name='Target Y',
                                line={'shape': 'hv'},
                                mode='lines',
                                visible='legendonly',
                            )
                    timeseries_data.append(target_y_trace)
                    # except:
                    #     pass

                    #print(timeseries_data)
                except:
                    eye_rand_trace = go.Scatter(
                                        x=np.arange(10),
                                        y=[0]*10,
                                        name='Eye Trace file not found or parsed',
                                        mode='lines'
                                        )
                    timeseries_data.append(eye_rand_trace)

            if 'Mouse' in  file:
                try:
                    fname = glob.glob(op.join(file_loc, file))[0]
                    fdata = read_hdf5(fname)['device_data']

                    mouse_df = pd.DataFrame(fdata['time_series'][:,:], columns=['mouse_x','mouse_y','clicks'])
                    mouse_df['timestamps'] = fdata['time_stamps']
                    #print(mouse_df.head(n=2))
                    mouse_datetime = [datetime.fromtimestamp(timestamp) for timestamp in mouse_df.timestamps]

                    trace5 = go.Scatter(
                                x=mouse_datetime,
                                y=mouse_df['mouse_x'],
                                name='Mouse X',
                                mode='lines',
                                visible='legendonly'
                            )
                    timeseries_data.append(trace5)

                    trace6 = go.Scatter(
                                x=mouse_datetime,
                                y=mouse_df['mouse_y'],
                                name='Mouse Y',
                                mode='lines',
                                visible='legendonly'
                            )
                    timeseries_data.append(trace6)
                except:
                    mouse_rand_trace = go.Scatter(
                                        x=np.arange(10),
                                        y=[0.5]*10,
                                        name='Mouse Trace file not found or parsed',
                                        mode='lines',
                                        visible='legendonly'
                                        )
                    timeseries_data.append(mouse_rand_trace)

            if 'Mic' in file:
                try:
                    fname = glob.glob(op.join(file_loc, file))[0]
                    fdata = read_hdf5(fname)['device_data']

                    audio_tstmp = fdata['time_stamps']

                    audio_ts = fdata['time_series']
                    chunk_len = audio_ts.shape[1]
                    # chunk timestamps from end of chunck, add beginning
                    audio_tstmp = np.insert(audio_tstmp, 0, audio_tstmp[0] - np.diff(audio_tstmp).mean())
                    tstmps = []
                    for i in range(audio_ts.shape[0]):
                        tstmps.append(np.linspace(audio_tstmp[i], audio_tstmp[i+1], chunk_len))
                    audio_tstmp_full = np.hstack(tstmps)

                    audio_ts_full = np.hstack(audio_ts)

                    audio_df = pd.DataFrame(audio_ts_full, columns=['amplitude'])
                    audio_df['timestamps'] = audio_tstmp_full
                    #print(audio_df.head(n=2))
                    audio_df = audio_df.iloc[::20,:]

                    trace7 = go.Scatter(
                                x=[datetime.fromtimestamp(timestamp) for timestamp in audio_df.timestamps],
                                y=audio_df['amplitude'],
                                name='Audio Trace',
                                mode='lines',
                                visible='legendonly',
                                yaxis="y2"
                            )
                    timeseries_data.append(trace7)

                    # fs = 44100
                    # N_pts = 2**11 # Number of points in window 2048
                    # w = signal.windows.hann(N_pts)
                    # freqs, bins, Pxx = signal.spectrogram(audio_df['amplitude'],fs,window = w,nfft=N_pts)
                    #print(freqs, bins, Pxx)

                    # specgram_trace = go.Heatmap(
                    #             x= bins,
                    #             y= freqs,
                    #             z= 10*np.log10(Pxx),
                    #             colorscale='plasma',
                    #             )
                    #print(specgram_trace)

                    audio_trace = go.Scatter(
                                x=[timestamp-audio_df.timestamps[0] for timestamp in audio_df.timestamps],
                                y=audio_df['amplitude'],
                                name='Audio Trace',
                                mode='lines',
                            )
                    specgram_data.append(audio_trace)

                    vert_trace = go.Scatter(
                                x=[0]*len(audio_df['timestamps']),
                                y=audio_df['amplitude'],
                                name='Frame',
                                mode='lines'
                                )
                    specgram_data.append(vert_trace)

                except:
                    mic_ts_rand_trace = go.Scatter(
                                        x=np.arange(10),
                                        y=[1]*10,
                                        name='Mic trace file not found or parsed',
                                        mode='lines'
                                        )
                    timeseries_data.append(mic_ts_rand_trace)

                    random_y = np.random.randint(0,5,size=(10))
                    mic_audio_rand_trace = go.Scatter(
                                        x=np.arange(10),
                                        y=random_y,
                                        name='Random Data',
                                        mode='lines'
                                        )
                    vert_trace = go.Scatter(
                        x=[0]*10,
                        y=random_y,
                        name='x position',
                        mode='lines'
                        )
                    specgram_data.append(mic_audio_rand_trace)
                    specgram_data.append(vert_trace)

            if 'Mbient_RH' in file:
                try:
                    fname = glob.glob(op.join(file_loc, file))[0]
                    fdata = read_hdf5(fname)['device_data']

                    imu_df = pd.DataFrame(fdata['time_series'][:,[1,2,3,4,5,6]], columns=['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z'])
                    imu_df['timestamps'] = fdata['time_stamps']
                    #print(imu_df.head(n=2))
                    imu_datetime = [datetime.fromtimestamp(timestamp) for timestamp in imu_df.timestamps]

                    trace8 = go.Scatter(
                                x=imu_datetime,
                                y=imu_df['acc_x'],
                                name='Acceleration X',
                                mode='lines',
                                visible='legendonly',
                                yaxis="y2"
                            )
                    timeseries_data.append(trace8)

                    trace9 = go.Scatter(
                                x=imu_datetime,
                                y=imu_df['acc_y'],
                                name='Acceleration Y',
                                mode='lines',
                                visible='legendonly',
                                yaxis="y2"
                            )
                    timeseries_data.append(trace9)

                    trace10 = go.Scatter(
                                x=imu_datetime,
                                y=imu_df['acc_z'],
                                name='Acceleration Z',
                                mode='lines',
                                visible='legendonly',
                                yaxis="y2"
                            )
                    timeseries_data.append(trace10)

                    trace11 = go.Scatter(
                                x=imu_datetime,
                                y=imu_df['gyr_x'],
                                name='Gyroscope X',
                                mode='lines',
                                visible='legendonly',
                                yaxis="y2"
                            )
                    timeseries_data.append(trace11)

                    trace12 = go.Scatter(
                                x=imu_datetime,
                                y=imu_df['gyr_y'],
                                name='Gyroscope Y',
                                mode='lines',
                                visible='legendonly',
                                yaxis="y2"
                            )
                    timeseries_data.append(trace12)

                    trace13 = go.Scatter(
                                x=imu_datetime,
                                y=imu_df['gyr_z'],
                                name='Gyroscope Z',
                                mode='lines',
                                visible='legendonly',
                                yaxis="y2"
                            )
                    timeseries_data.append(trace13)
                except:
                    imu_rand_trace = go.Scatter(
                                        x=np.arange(10),
                                        y=[1.5]*10,
                                        name='IMU Trace file not found or parsed',
                                        mode='lines'
                                        )
                    timeseries_data.append(imu_rand_trace)
        
        print(len(timeseries_data),len(specgram_data))
        return timeseries_data, specgram_data


# --- Creating all_files list --- #
all_file_list = get_file_list(nb_data_df)

# --- Reading face landmark file --- #
face_landmark_data = read_hdf5(face_landmark_filename)['device_data']
face_landmark_timestamps = face_landmark_data['time_stamps']
face_landmark_points = face_landmark_data['time_series']
face_landmark_x = face_landmark_points[::100,:,0]
face_landmark_y = face_landmark_points[::100,:,1]


app = dash.Dash(__name__)
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)


app.layout = html.Div([
                        html.H1('Neurobooth Explorer', style={'textAlign':'center'}),
                        html.H4('A web-app to browse Neurobooth data', style={'textAlign':'center'}),
                        html.Hr(),
                        html.Div(
                                dcc.Markdown('''
                                * Use the dropdown menus to select subject_id, session_date, task or clinical indication
                                * The table displays context specific information depending on dropdown selection
                                '''),style={'padding-left':'8%'},
                                ),
                        html.Div([
                                dcc.Markdown('''Select Subject ID''', style={'textAlign':'center'}),
                                dcc.Dropdown(id="subject_id_dropdown",
                                            options=[ {'label': x, 'value': x} for x in sub_id_list],
                                            value=sub_id_list[0],
                                            clearable=False),
                                ], className="three columns", style={'width':'20%', 'display':'inline-block', 'padding-left':'7%', 'padding-right':'2%'}),
                        html.Div([
                                dcc.Markdown('''Select Date''', style={'textAlign':'center'}),
                                dcc.Dropdown(id="session_date_dropdown",
                                            options=[ {'label': x, 'value': x} for x in session_date_list],
                                            value=session_date_list[0],
                                            clearable=False),
                                ], className="three columns", style={'width':'20%', 'display':'inline-block', 'padding-right':'2%'}),
                        html.Div([
                                dcc.Markdown('''Select Task''', style={'textAlign':'center'}),
                                dcc.Dropdown(id="task_dropdown",
                                            options=[ {'label': x, 'value': x} for x in task_list],
                                            value=task_list[0],
                                            clearable=False),
                                ], className="three columns", style={'width':'20%', 'display':'inline-block', 'padding-right':'2%'}),
                        html.Div([
                                dcc.Markdown('''Select Clinical Indication''', style={'textAlign':'center'}),
                                dcc.Dropdown(id="clinical_dropdown",
                                            options=[ {'label': x, 'value': x} for x in clinical_list],
                                            value=clinical_list[0],
                                            clearable=False),
                                ], className="three columns", style={'width':'20%', 'display':'inline-block'}),
                        dash_table.DataTable(
                                id='datatable',
                                style_table={'maxHeight': '400px','overflowY': 'scroll', 'overflowX': 'auto'},
                                style_data={'whiteSpace': 'normal', 'height': 'auto',}
                                ),
                        html.Hr(),
                        html.H3('Task Session Information', style={'textAlign':'center'}),
                        html.Div(
                                dcc.Markdown('''
                                * Use dropdown to select a task session
                                * The table shows data files associated with the selected task session
                                '''),style={'padding-left':'8%'},
                                ),
                        html.Div([
                                html.Div([
                                        dcc.Dropdown(id="task_session_dropdown",
                                                    #options=[ {'label': x, 'value': x} for x in clinical_list],
                                                    #value=clinical_list[0],
                                                    clearable=False),
                                        ], className="three columns", style={'width':'30%', 'display':'inline-block', 'padding-left':'5%', 'padding-right':'2%'}),
                                html.Div([
                                        dash_table.DataTable(
                                                id='task_session_file_datatable',
                                                style_table={'maxHeight': '150px','overflowY': 'scroll', 'overflowX': 'auto'},
                                                ),
                                        ], className="nine columns", style={'width':'55%', 'display':'inline-block', 'padding-left':'3%', 'verticalAlign':'top'}),
                                ]),
                        html.Hr(),
                        html.H3('Explore Time Series', style={'textAlign':'center'}),
                        html.Div([
                                dcc.Markdown('''
                                * Click legend to toggle trace visibility
                                * Double click any legend to toggle all traces
                                ''',style={'padding-left':'8%'}),
                                dcc.Graph(id="timeseries_graph")
                                ]),
                        html.Hr(),
                        html.H3('Explore Audio and Facial Landmarks', style={'textAlign':'center'}),
                        html.Div([
                                #dcc.Markdown('''* Lorem Ipsum''',style={'padding-left':'8%'}),
                                html.Div([
                                        dcc.Graph(id="specgram_graph"),
                                        html.Div([
                                                dcc.Slider(
                                                    id="frame_slider",
                                                    min=0,
                                                    max=len(face_landmark_x)-1,
                                                    value=0,
                                                    step=1,
                                                    marks={i: str(i) for i in range(len(face_landmark_x))[::10]},
                                                    updatemode='drag')
                                            ], style={'width':'79%', 'padding-left':'8%', 'padding-right':'5%', 'padding-bottom':'1%'})
                                        ], className="nine columns", style={'width':'70%', 'display':'inline-block'}),#, 'padding-left':'3%', 'padding-right':'2%'}),
                                html.Div([
                                        dcc.Graph(id="face_landmarks_with_slider", style={'horizontalAlign':'left'}),
                                        ], className="three columns", style={'width':'30%', 'display':'inline-block', 'verticalAlign':'center', 'horizontalAlign':'left'})#, 'padding-left':'2%', 'padding-right':'3%'
                                ]),
                        html.Hr(),
                        html.Div([
                                dcc.Markdown('''Click button to retrieve new data from Neurobooth database'''),
                                html.Button('Get New Data', id='db_button', n_clicks_timestamp=0),
                                html.Div(id='button_container', children='Click button to retreive new data', style={'padding-top':'2%'})
                                ], style={'textAlign':'center', 'width': '50%', 'margin':'auto', 'verticalAlign': 'middle'}),
                        html.Hr(),
                        html.Div([dcc.Markdown('''
                                                Hints:
                                                * Double click anywhere in the plot area to reset view
                                                * Use control buttons at top right corner of plot area to interact with the plots
                                                * You can zoom in/out, pan, select area etc.
                                                * Refreshing the page also retrieves new data from the database
                                                * Email spatel@phmi.partners.org for bug reports and feedback''', style={'padding-left':'8%'})]),
                        html.Hr(),
                        html.Div([dcc.Markdown('''Thank you for using Neurobooth Explorer''', style={'textAlign':'center'})]),
                        html.Hr(),
                    ])


@app.callback(
    [Output(component_id='datatable', component_property='data'),
    Output(component_id='datatable', component_property='columns'),
    #Output(component_id='file_list_dropdown', component_property='options'),
    #Output(component_id='file_list_dropdown', component_property='value'),
    Output(component_id='task_session_dropdown', component_property='options'),
    Output(component_id='task_session_dropdown', component_property='value')],
    [Input("subject_id_dropdown", "value"),
    Input("session_date_dropdown", "value"),
    Input("task_dropdown", "value"),
    Input("clinical_dropdown", "value")])
def update_table(subid_value, date_value, task_value, clinical_value):

    # getting most recent context
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]

    # setting dropdown_value based on most recent context
    dropdown_value=None
    
    if 'subject_id_dropdown' in changed_id:
        dropdown_value = subid_value
    elif 'session_date_dropdown' in changed_id:
        dropdown_value = date_value
    elif 'task_dropdown' in changed_id:
        dropdown_value = task_value
    elif 'clinical_dropdown' in changed_id:
        dropdown_value = clinical_value
    
    print(dropdown_value)

    #subfiles=[]
    session_files=[]
    data_df = pd.DataFrame()
    
    if dropdown_value in sub_id_list:
        #subfiles = get_file_list(nb_data_df[nb_data_df['subject_id']==dropdown_value])
        session_files = get_task_session_files(nb_data_df[nb_data_df['subject_id']==dropdown_value])
        scols = ['subject_id', 'gender', 'age', 'session_date', 'tasks', 'task_log']
        data_df = nb_data_df[nb_data_df['subject_id']==dropdown_value].astype('str').groupby(['session_date']).agg(lambda x: ' '.join(x.unique()))
        data_df.reset_index(inplace=True)
        data_df = data_df[scols]
    elif dropdown_value in session_date_list:
        dropdown_value_datetime = datetime.strptime(dropdown_value, '%Y-%m-%d')
        #subfiles = get_file_list(nb_data_df[nb_data_df['session_date']==dropdown_value_datetime.date()])
        session_files = get_task_session_files(nb_data_df[nb_data_df['session_date']==dropdown_value_datetime.date()])
        dcols = ['subject_id', 'gender', 'age', 'tasks', 'task_log']
        data_df = nb_data_df[nb_data_df['session_date']==dropdown_value_datetime.date()].astype('str').groupby(['subject_id']).agg(lambda x: ' '.join(x.unique()))
        data_df.reset_index(inplace=True)
        data_df = data_df[dcols]
    elif dropdown_value in task_list:
        #subfiles = get_file_list(nb_data_df[nb_data_df['tasks']==dropdown_value])
        session_files = get_task_session_files(nb_data_df[nb_data_df['tasks']==dropdown_value])
        tcols = ['subject_id', 'gender', 'age', 'session_date', 'task_log']
        data_df = nb_data_df[nb_data_df['tasks']==dropdown_value].astype('str').groupby(['subject_id']).agg(lambda x: ' '.join(x.unique()))
        data_df.reset_index(inplace=True)
        data_df = data_df[tcols]
    elif dropdown_value in clinical_list:
        data_df = nb_data_df.sample(n=20)
        #subfiles = get_file_list(data_df)
        session_files = get_task_session_files(data_df)
        data_df = data_df[['subject_id', 'gender', 'age', 'session_date', 'tasks', 'task_log']]
    
    # Creating primary data table
    data = data_df.to_dict('records')
    columns = [{'name': col, 'id': col} for col in data_df.columns]

    # Generating file list
    #subfile_list = subfiles
    #opts = [ {'label': x, 'value': x} for x in subfile_list]
    #val = subfile_list[0]

    # Generating task session file dropdown list
    task_session_opts = [ {'label': x, 'value': x} for x in np.sort(session_files)]
    task_session_val = np.sort(session_files)[-1]

    #return data, columns, opts, val, task_session_opts, task_session_val
    return data, columns, task_session_opts, task_session_val


@app.callback(
    [Output(component_id='task_session_file_datatable', component_property='data'),
    Output(component_id='task_session_file_datatable', component_property='columns'),
    Output('timeseries_graph', 'figure'),
    Output('specgram_graph', 'figure')],
    [Input("task_session_dropdown", "value")])
def update_table(task_session_value):
    task_files=[]
    try:
        for i in all_file_list:
            if task_session_value in i:
                task_files.append(i)
        if len(task_files) > 0:
            task_files = np.sort(list(set(task_files)))
        else:
            task_files.append('No file found')
    except:
        task_files.append('No file found')
    task_file_df = pd.DataFrame(task_files, columns=['task_files'])
    
    data = task_file_df.to_dict('records')
    columns = [{'name': col, 'id': col} for col in task_file_df.columns]

    timeseries_data, specgram_data = parse_files(task_files)

    timeseries_layout = go.Layout(
                margin={'l': 50, 'b': 30, 't': 30, 'r': 50},
                height=500,
                xaxis={
                    'title':'Time',
                    'showgrid':False,
                    'showticklabels':True,
                    'tickmode':'array',
                },
                yaxis={'showgrid':False},
                title={
                    'text':('Acceleration, Gyroscope, Eye Tracker and Mouse Time Series'),
                    'x':0.45,
                    'y':1,
                    'xanchor':'center',
                    'yanchor':'top',
                    'pad':{'t':10}
                },
                titlefont={'size':18},
            )
    
    timeseries_fig=go.Figure(data=timeseries_data, layout=timeseries_layout)
    timeseries_fig.update_layout(
        yaxis2=dict(
            anchor="x",
            overlaying="y",
            side="right"
        )
    )
    timeseries_fig.update_xaxes(tickangle=45, tickfont={'size':14}, showline=True, linewidth=1, linecolor='black', mirror=True, title_font={'size':18})
    timeseries_fig.update_yaxes(tickfont={'size':12})
    
    specgram_layout = go.Layout(
        title = 'Audio Trace',
        yaxis = dict(title = 'Amplitude'), # y-axis label
        xaxis = dict(title = 'Time (seconds)'), # x-axis label
        )
    
    specgram_fig = go.Figure(data=specgram_data, layout=specgram_layout)
    specgram_fig.update_layout(legend_x=1, legend_y=1)

    return data, columns, timeseries_fig, specgram_fig


@app.callback(
    [Output('face_landmarks_with_slider', 'figure'),
    Output('specgram_graph', 'figure')],
    Input('frame_slider', 'value'),
    State('specgram_graph', 'figure'))
def update_face_landmark_frame(selected_frame, specgram_fig):
    #print(specgram_fig)

    face_frame_fig = go.Figure()
    face_frame_fig.add_trace(
                    go.Scatter(
                            x=face_landmark_x[int(selected_frame)],
                            y=face_landmark_y[int(selected_frame)],
                            mode='markers',
                            marker=dict(size=10)
                    )
    )
    face_frame_fig['layout']['yaxis']['autorange'] = "reversed"
    face_frame_fig.update_layout(
                    margin={'l': 10, 'b': 30, 't': 30, 'r': 10},
                    width=450,
                    height=450,
                    title = 'Facial Landmarks',
                    yaxis = dict(title = 'x coordinate'),
                    xaxis = dict(title = 'y coordinate'),
    )

    if specgram_fig==None:
        specgram_fig=go.Figure()
    else:
        plot_min = np.min(specgram_fig['data'][0]['x'])
        plot_max = np.max(specgram_fig['data'][0]['x'])

        x_pos_list = np.linspace(plot_min, plot_max, len(face_landmark_x))
        
        specgram_fig['data'][1]['x'] = [x_pos_list[int(selected_frame)]]*len(specgram_fig['data'][0]['y'])
        specgram_fig['data'][1]['y'] = specgram_fig['data'][0]['y']

    return [face_frame_fig, specgram_fig]


@app.callback(
    [Output('button_container', 'children'),
    Output(component_id='subject_id_dropdown', component_property='options'),
    Output(component_id='session_date_dropdown', component_property='options'),
    Output(component_id='task_dropdown', component_property='options'),
    Output(component_id='clinical_dropdown', component_property='options')],
    Input('db_button', 'n_clicks_timestamp'))
def on_button_click(n_clicks_timestamp):
    
    # defining global variables
    global nb_data_df
    global sub_id_list
    global session_date_list
    global task_list
    global clinical_list
    global all_file_list

    # Retrieving new data from database
    nb_data_df, sub_id_list, session_date_list, task_list, clinical_list = rebuild_master_data_table(sql_query_cmd)
    
    # Generate new all_file_list from new nb_data_df
    all_file_list = get_file_list(nb_data_df)

    updated_sub_id_list_options = [ {'label': x, 'value': x} for x in sub_id_list]
    updated_session_date_list_options = [ {'label': x, 'value': x} for x in session_date_list]
    updated_task_list_options = [ {'label': x, 'value': x} for x in task_list]
    updated_clinical_list_options = [ {'label': x, 'value': x} for x in clinical_list]

    #dt_str = datetime.fromtimestamp(int(n_clicks_timestamp/1000)).strftime('%Y-%m-%d, %H:%M:%S')
    dt_str = datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
    return 'New data was last retrieved at ' + dt_str, updated_sub_id_list_options, updated_session_date_list_options, updated_task_list_options, updated_clinical_list_options


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port='8050', debug=True)