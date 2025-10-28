import neurobooth_terra

import psycopg2
from sshtunnel import SSHTunnelForwarder

import dash_auth
from flask import request


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
import random

from h5io import read_hdf5

from datetime import datetime

from scipy import signal

import credential_reader
from data_table import rebuild_master_data_table, sql_query_cmd


# ==== Read Credentials ==== #
dataflow_args = credential_reader.read_dataflow_configs()
file_locs = dataflow_args['suitable_volumes']

db_args, ssh_args = credential_reader.read_db_secrets()

USERNAME_PASSWORD_PAIRS = credential_reader.get_user_pass_pairs()

face_landmark_filename = '/space/drwho/3/neurobooth/applications/neurobooth-explorer/facial_landmark_file/100001_2022-02-28_08h-55m-00s_passage_obs_1_R001-FLIR_blackfly_1-FLIR_rgb_1_face_landmarks.hdf5'
# ========================== #


# Fetching data at app launch
nb_data_df, bars_df, sub_id_list, session_date_list, task_list, clinical_list = rebuild_master_data_table(sql_query_cmd)


# Building Control subjects list and subset dataframe at app launch
control_indices = []
for ix, diagnosis_list in enumerate(nb_data_df.primary_diagnosis):
    if 'Control' in diagnosis_list:
        control_indices.append(ix)

ctrl_df = nb_data_df.iloc[control_indices]
ctrl_list = ctrl_df.subject_id.unique()
# enter code here to filter out improper controls

# --- Function to get random Control subject EyeTracker hdf5 file --- #
def get_rnd_ctrl_et_hdf5(task):
    rnd_ctrl = random.choice(ctrl_list)
    return ctrl_df[(ctrl_df['subject_id']==rnd_ctrl) & (ctrl_df['tasks']==task) & (ctrl_df['device_id']=='Eyelink_1')].sample(n=1).hdf5_files.iloc[0]


# --- Function to check for other sessions for a subject --- #
def check_for_other_sessions(subj_id, current_session_date, task):
    
    other_sessions_exist = False
    other_session_file_list = []
    
    if subj_id == '100001': # Test id with many many sessions
        return other_sessions_exist, other_session_file_list
    
    if len(nb_data_df[nb_data_df['subject_id']==subj_id].session_date.unique()) > 1:
        other_sessions_exist = True
    
    if other_sessions_exist:
        all_sessions_file_list = nb_data_df[(nb_data_df['subject_id']==subj_id) & (nb_data_df['tasks']==task) & (nb_data_df['device_id']=='Eyelink_1')].hdf5_files.unique()
        for session_file in all_sessions_file_list:
            if current_session_date not in session_file:
                other_session_file_list.append(session_file+'OTHERSESSION')
    
    return other_sessions_exist, other_session_file_list


# --- Function to get fresh copy of nex_annotations table from db --- #
def refresh_nex_annotations():
    with SSHTunnelForwarder(**ssh_args) as tunnel:
        with psycopg2.connect(**db_args) as conn:
            nex_annotations_df = neurobooth_terra.Table('nex_annotations', conn).query()
            nex_anno_cols = ['subject_id', 'session_datetime', 'task_id', 'annotation', 'annotator_name', 'annotation_source', 'annotation_submit_time']
            nex_annotations_df = nex_annotations_df[nex_anno_cols]
    return nex_annotations_df

# Fetching nex_annotations at app launch
nex_annotations_df = refresh_nex_annotations()


# --- Function to insert annotation into nex_annotation table in db --- #
def insert_into_annotation_table(cols, vals):
    with SSHTunnelForwarder(**ssh_args) as tunnel:
        with psycopg2.connect(**db_args) as conn:
            nex_anno_table = neurobooth_terra.Table('nex_annotations', conn)
            nex_anno_table.insert_rows(cols=cols, vals=vals, on_conflict='update')
    return None


# --- Function to extract all file list from dataframe --- #
# def get_file_list(fdf):
#     file_list=[]
#     for file_array in fdf.file_names:
#         for file in file_array:
#             if file[-5:] == '.hdf5':
#                 file_list.append(file)
#     return file_list


# --- Function to generate empty file length dataframe --- #
def generate_empty_file_len_df():
    len_df = pd.DataFrame()
    for col in ['Eyelink', 'Mouse', 'IMU', 'Audio']:
        len_df[col] = np.array([np.nan])
    return len_df


# --- Function to generate empty annotation dataframe --- #
def generate_empty_anno_df():
    anno_df = pd.DataFrame()
    for col in ['subject_id', 'session_datetime', 'task_id', 'annotation', 'annotator_name', 'annotation_source', 'annotation_submit_time']:
        anno_df[col] = np.array([''])
    return anno_df


# --- Function to get file location based on odd/even subject_id --- #
def get_file_loc(filename):
    odd_even_session, _ = op.split(filename)
    for loc in file_locs:
        if op.exists(op.join(loc, odd_even_session)):
            return loc
    return None
    # odd_even_subject_id = int(odd_even_session.split('_')[0])
    # if odd_even_subject_id % 2: #odd
    #     file_loc = file_loc_odd
    # else: #even
    #     file_loc = file_loc_even
    # return file_loc

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


# --- Function for getting start and end task times for movement tasks --- #
def get_movement_task_start_end_times(mv_filename, fig):
    try:
        if mv_filename.endswith('CONTROL') or mv_filename.endswith('OTHERSESSION'): # don't plot task start and end time for CONTROL trace
            return fig
        file_loc = get_file_loc(mv_filename)
        fname = glob.glob(op.join(file_loc, mv_filename))[0]
    except:
        print('file not found at location:', file_loc, 'filename :', mv_filename)
        return fig
    marker = read_hdf5(fname)['marker']

    start_local_ts = []
    end_local_ts = []
    for txt in marker['time_series']:
        if txt[0][:10] == 'Task_start':
            start_local_ts.append(float(txt[0].split('_')[-1]))
        elif txt[0][:8] == 'Task_end':
            end_local_ts.append(float(txt[0].split('_')[-1]))

    if start_local_ts and end_local_ts:
        for start,end in zip(start_local_ts, end_local_ts):
            xstart=datetime.fromtimestamp(start)
            xend=datetime.fromtimestamp(end)
            fig.add_shape(type="rect",
                        xref="x",
                        yref="paper",
                        x0=xstart, y0=0, x1=xend, y1=1,
                        line=dict(color="rgba(0,0,0,0)",width=3,),
                        fillcolor='rgba(255,0,255,0.1)',
                        layer="below"
            )
    
    return fig


# --- Function to get button presses for DSC task --- #
def get_DSC_button_presses(dsc_filename, fig):
    try:
        file_loc = get_file_loc(dsc_filename)
        fname = glob.glob(op.join(file_loc, dsc_filename))[0]
    except:
        print('file not found at location:', file_loc, 'filename :', dsc_filename)
        return fig
    marker = read_hdf5(fname)['marker']

    new_symbol_local_ts = []
    button_press_local_ts = []
    button_release_local_ts = []
    for txt in marker['time_series']:
        if txt[0][:11] == 'Trial_start':
            new_symbol_local_ts.append(float(txt[0].split('_')[-1]))
        elif txt[0][:14] == 'Response_start':
            button_press_local_ts.append(float(txt[0].split('_')[-1]))
        elif txt[0][:12] == 'Response_end':
            button_release_local_ts.append(float(txt[0].split('_')[-1]))

    if new_symbol_local_ts and button_press_local_ts and button_release_local_ts:
        for i in new_symbol_local_ts:
            x=datetime.fromtimestamp(i)
            fig.add_shape(type="line",
                x0=x, y0=0, x1=x, y1=1400,
                line=dict(
                    color="Green",
                    width=1,
                    dash="solid",
                )
            )

        for i in button_press_local_ts:
            x=datetime.fromtimestamp(i)
            fig.add_shape(type="line",
                x0=x, y0=0, x1=x, y1=1400,
                line=dict(
                    color="Red",
                    width=1,
                    dash="dash",
                )
            )

        for i in button_release_local_ts:
            x=datetime.fromtimestamp(i)
            fig.add_shape(type="line",
                x0=x, y0=0, x1=x, y1=1400,
                line=dict(
                    color="Blue",
                    width=1,
                    dash="dot",
                )
            )
        fig.update_shapes(dict(xref='x', yref='y'))

        button_releases = []
        for new_symbol in new_symbol_local_ts:
            for button_release in button_release_local_ts:
                if button_release>new_symbol:
                    button_releases.append(button_release)
                    break

        for start,end in zip(new_symbol_local_ts, button_releases):
            xstart=datetime.fromtimestamp(start)
            xend=datetime.fromtimestamp(end)
            fig.add_shape(type="rect",
                        xref="x",
                        yref="y",
                        x0=xstart, y0=0, x1=xend, y1=1400,
                        line=dict(color="rgba(0,0,0,0)",width=3,),
                        fillcolor='rgba(255,0,255,0.1)',
                        layer="below"
            )

        fig.add_trace(go.Scatter(
            x=[datetime.fromtimestamp(new_symbol_local_ts[0]), datetime.fromtimestamp(button_press_local_ts[0]), datetime.fromtimestamp(button_release_local_ts[0])],
            y=[-10,-30,-60],
            text=['New Symbol', 'Button Press', 'Button Release'],
            mode="text",
            name='Annotation Text'
            )
        )
    
    return fig


# --- Function to parse task session files and return traces --- #
def parse_files(task_files, mbient_sensors):

    timeseries_data=[]
    specgram_data=[]
    len_df = generate_empty_file_len_df()

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

        return timeseries_data, specgram_data, len_df
    
    else:

        for file in task_files:
            file_loc = get_file_loc(file)

            # Computing offsets for adding Control subject time series #########
            if 'Eyelink' in file and file.endswith('.hdf5'):
                et_trg_start=0
                ctrl_trg_start=0
                otr_sess_trg_start=0

                et_data = read_hdf5(op.join(file_loc, file))
                ctrl_dt_corr = int(float(et_data['marker']['time_series'][0][0].split('_')[-1]) - et_data['marker']['time_stamps'][0])
                for ix, txt in enumerate(et_data['marker']['time_series']):
                    if '!V TARGET_POS' in txt[0]:
                        et_trg_start = et_data['marker']['time_stamps'][ix]
                        break

                for ctrl_file in task_files:
                    if ctrl_file.endswith('CONTROL'):
                        ctrl_file_loc = get_file_loc(ctrl_file.replace('CONTROL', ''))
                        try:
                            ctrl_data = read_hdf5(op.join(ctrl_file_loc, ctrl_file.replace('CONTROL', '')))
                        except:
                            break
                        for ix, txt in enumerate(ctrl_data['marker']['time_series']):
                            if '!V TARGET_POS' in txt[0]:
                                ctrl_trg_start = ctrl_data['marker']['time_stamps'][ix]
                                break

                otr_sess_trg_start_dict = {}
                for otr_sess_file in task_files:
                    sess_ky, _ = op.split(otr_sess_file)
                    if otr_sess_file.endswith('OTHERSESSION'):
                        otr_sess_file_loc = get_file_loc(otr_sess_file.replace('OTHERSESSION', ''))
                        try:
                            otr_sess_data = read_hdf5(op.join(otr_sess_file_loc, otr_sess_file.replace('OTHERSESSION', '')))
                        except:
                            break
                        for ix, txt in enumerate(otr_sess_data['marker']['time_series']):
                            if '!V TARGET_POS' in txt[0]:
                                otr_sess_trg_start_dict[sess_ky] = otr_sess_data['marker']['time_stamps'][ix]
                                break

                ctrl_trg_corr = et_trg_start - ctrl_trg_start
                otr_sess_trg_corr_dict = {}
                for ky in otr_sess_trg_start_dict.keys():
                    otr_sess_trg_corr_dict[ky] = et_trg_start - otr_sess_trg_start_dict[ky]
            ####################################################################
            
            if 'Eyelink' in file:
                try:
                    if file.endswith('CONTROL'):
                        fname = glob.glob(op.join(file_loc, file.replace('CONTROL', '')))[0]
                    elif file.endswith('OTHERSESSION'):
                        fname = glob.glob(op.join(file_loc, file.replace('OTHERSESSION', '')))[0]
                    else:
                        fname = glob.glob(op.join(file_loc, file))[0]
                    complete_file = read_hdf5(fname)
                    fdata = complete_file['device_data']
                    mdata = complete_file['marker']

                    # correcting raw timestamps from CTR machine for Control subject or Other Sessions
                    if file.endswith('CONTROL'):
                        fdata['time_stamps'] = fdata['time_stamps'] + ctrl_trg_corr
                        mdata['time_stamps'] = mdata['time_stamps'] + ctrl_trg_corr
                    if file.endswith('OTHERSESSION'):
                        sess_ky, _ = op.split(file)
                        fdata['time_stamps'] = fdata['time_stamps'] + otr_sess_trg_corr_dict[sess_ky]
                        mdata['time_stamps'] = mdata['time_stamps'] + otr_sess_trg_corr_dict[sess_ky]
                    ################################################################

                    len_df.at[0, 'Eyelink'] = len(fdata['time_series'])

                    et_df = pd.DataFrame(fdata['time_series'][:, [0,1,3,4]], columns=['R_gaze_x','R_gaze_y','L_gaze_x','L_gaze_y'])
                    et_df['timestamps'] = fdata['time_stamps']
                    fs = int(1/np.median(np.diff(et_df.timestamps)))
                    #print('Eyelink Sampling Rate =', fs)

                    # # Adding eyelink timestamp #################################
                    # et_df['el_timestamps'] = fdata['time_series'][:,-1]
                    # # Computing correction between eyelink time and control time
                    # et_sampling_corr = et_df['el_timestamps'].apply(lambda x: x-et_df.el_timestamps[0]) - et_df['timestamps'].apply(lambda x: x-et_df.timestamps[0])
                    # ############################################################

                    dt_corr = int(float(mdata['time_series'][0][0].split('_')[-1]) - mdata['time_stamps'][0]) # datetime-correction factor : time correction offset for correcting LSL time to local time
                    # correcting Control subject corrected CTR timestamps to Patient subject's local time
                    if file.endswith('CONTROL') or file.endswith('OTHERSESSION'):
                        dt_corr = ctrl_dt_corr
                    #####################################################################################
                    et_datetime = et_df.timestamps.apply(lambda x: datetime.fromtimestamp(dt_corr+x))

                    # # correcting control timestamps with eyelink timestamp correction factor and coverting to present day datetime
                    # corr_timestamps = et_df['timestamps'] + et_sampling_corr
                    # trg_corr = np.mean(et_sampling_corr)
                    # el_datetime = np.array([datetime.fromtimestamp(dt_corr+i-trg_corr) for i in corr_timestamps])
                    # # target correction is subtracted because undersampling eyetracker leads to expanded time
                    # ##############################################################################################################

                    r_gaze_x_trace_str = 'Right Eye Gaze X : fs = '
                    r_gaze_y_trace_str = 'Right Eye Gaze Y'
                    l_gaze_x_trace_str = 'Left Eye Gaze X'
                    l_gaze_y_trace_str = 'Left Eye Gaze Y'

                    if file.endswith('CONTROL'):
                        subj_id_str = file.split('_')[0]
                        r_gaze_x_trace_str = 'CONTROL: ' + subj_id_str + '<br>' + r_gaze_x_trace_str
                        r_gaze_y_trace_str = 'CONTROL ' + r_gaze_y_trace_str
                        l_gaze_x_trace_str = 'CONTROL ' + l_gaze_x_trace_str
                        l_gaze_y_trace_str = 'CONTROL ' + l_gaze_y_trace_str

                    if file.endswith('OTHERSESSION'):
                        subj_id_str, _ = op.split(file)
                        r_gaze_x_trace_str = subj_id_str +'<br>'+ r_gaze_x_trace_str
                        r_gaze_y_trace_str = subj_id_str +'<br>'+ r_gaze_y_trace_str
                        l_gaze_x_trace_str = subj_id_str +'<br>'+ l_gaze_x_trace_str
                        l_gaze_y_trace_str = subj_id_str +'<br>'+ l_gaze_y_trace_str

                    trace1 = go.Scatter(
                                x=et_datetime[::4],
                                y=et_df['R_gaze_x'][::4],
                                name = r_gaze_x_trace_str + str(fs),
                                mode='lines'
                            )
                    timeseries_data.append(trace1)
                    #print(trace1)

                    trace2 = go.Scatter(
                                x=et_datetime[::4],
                                y=et_df['R_gaze_y'][::4],
                                name = r_gaze_y_trace_str,
                                mode='lines'
                            )
                    timeseries_data.append(trace2)
                    #print(trace2)

                    trace3 = go.Scatter(
                                x=et_datetime[::4],
                                y=et_df['L_gaze_x'][::4],
                                name = l_gaze_x_trace_str,
                                mode='lines',
                                visible='legendonly',
                            )
                    timeseries_data.append(trace3)
                    #print(trace3)

                    trace4 = go.Scatter(
                                x=et_datetime[::4],
                                y=et_df['L_gaze_y'][::4],
                                name = l_gaze_y_trace_str,
                                mode='lines',
                                visible='legendonly',
                            )
                    timeseries_data.append(trace4)
                    #print(trace4)

                    # ### Adding traces w.r.t eyelink times ###
                    # el_trace1 = go.Scatter(
                    #             x=el_datetime,
                    #             y=et_df['R_gaze_x'],
                    #             name='R_gaze_x on Eyelink Time : Marker offset = '+str(np.abs(trg_corr*1000))[:3]+' ms',
                    #             mode='lines',
                    #             visible='legendonly',
                    #         )
                    # timeseries_data.append(el_trace1)

                    # el_trace2 = go.Scatter(
                    #             x=el_datetime,
                    #             y=et_df['R_gaze_y'],
                    #             name='Right Eye Gaze Y on Eyelink Time',
                    #             mode='lines',
                    #             visible='legendonly',
                    #         )
                    # timeseries_data.append(el_trace2)

                    # el_trace3 = go.Scatter(
                    #             x=el_datetime,
                    #             y=et_df['L_gaze_x'],
                    #             name='Left Eye Gaze X on Eyelink Time',
                    #             mode='lines',
                    #             visible='legendonly',
                    #         )
                    # timeseries_data.append(el_trace3)

                    # el_trace4 = go.Scatter(
                    #             x=el_datetime,
                    #             y=et_df['L_gaze_y'],
                    #             name='Left Eye Gaze Y on Eyelink Time',
                    #             mode='lines',
                    #             visible='legendonly',
                    #         )
                    # timeseries_data.append(el_trace4)
                    # #########################################

                    if 'MOT' not in file:
                        # Adding target trace - target trace is always extracted from eyelink marker data
                        ts_ix = []
                        x_coord = []
                        y_coord = []
                        for ix, txt in enumerate(mdata['time_series']):
                            if '!V TARGET_POS' in txt[0]:
                                ts_ix.append(ix)
                                l = txt[0].split(' ')
                                x_coord.append(int(l[3][:-1]))
                                y_coord.append(int(l[4]))

                        ctrl_ts = mdata['time_stamps'][ts_ix]

                        target_pos_df = pd.DataFrame()
                        target_pos_df['ctrl_ts'] = ctrl_ts
                        target_pos_df['x_pos'] = x_coord
                        target_pos_df['y_pos'] = y_coord
                        #print(target_pos_df.head(n=2))
                        target_datetime = target_pos_df.ctrl_ts.apply(lambda x: datetime.fromtimestamp(dt_corr+x))

                        # # eyelink target datetime
                        # trg_corr = np.mean(et_sampling_corr)
                        # el_target_datetime = target_pos_df.ctrl_ts.apply(lambda x: datetime.fromtimestamp(dt_corr+x+trg_corr))
                        # #########################

                        target_x_trace = go.Scatter(
                                    x=target_datetime,
                                    y=(target_pos_df['x_pos']),
                                    name='Target X' if not file.endswith('CONTROL') else 'CONTROL Target X',
                                    line={'shape': 'hv'},
                                    mode='lines',
                                    visible='legendonly',
                                )
                        timeseries_data.append(target_x_trace)

                        target_y_trace = go.Scatter(
                                    x=target_datetime,
                                    y=(target_pos_df['y_pos']),
                                    name='Target Y' if not file.endswith('CONTROL') else 'CONTROL Target y',
                                    line={'shape': 'hv'},
                                    mode='lines',
                                    visible='legendonly',
                                )
                        timeseries_data.append(target_y_trace)

                        # ### Adding target traces on eyelink times ###
                        # el_target_x_trace = go.Scatter(
                        #             x=el_target_datetime,
                        #             y=(target_pos_df['x_pos']),
                        #             name='Target X on Eyelink Time',
                        #             line={'shape': 'hv'},
                        #             mode='lines',
                        #             visible='legendonly',
                        #         )
                        # timeseries_data.append(el_target_x_trace)

                        # el_target_y_trace = go.Scatter(
                        #             x=el_target_datetime,
                        #             y=(target_pos_df['y_pos']),
                        #             name='Target Y on Eyelink Time',
                        #             line={'shape': 'hv'},
                        #             mode='lines',
                        #             visible='legendonly',
                        #         )
                        # timeseries_data.append(el_target_y_trace)
                        # #############################################

                    elif 'MOT' in  file:
                        target_dict = dict()
                        for i in range(10):
                            target_str = 'target_'+str(i)
                            target_dict[target_str] = []
                        
                        target_dt = np.array([datetime.fromtimestamp(dt_corr + timestamp) for timestamp in mdata['time_stamps']])

                        # this ugly piece of code runs faster than beautiful code since the loop runs only once
                        for ix, txt in enumerate(mdata['time_series']):
                            if '!V TARGET_POS' in txt[0]:
                                txt_split = txt[0].split(' ') 
                                if txt_split[2]=='target_0':
                                    target_dict['target_0'].append([target_dt[ix], int(txt_split[3][:-1]), int(txt_split[4])])
                                elif txt_split[2]=='target_1':
                                    target_dict['target_1'].append([target_dt[ix], int(txt_split[3][:-1]), int(txt_split[4])])
                                elif txt_split[2]=='target_2':
                                    target_dict['target_2'].append([target_dt[ix], int(txt_split[3][:-1]), int(txt_split[4])])
                                elif txt_split[2]=='target_3':
                                    target_dict['target_3'].append([target_dt[ix], int(txt_split[3][:-1]), int(txt_split[4])])
                                elif txt_split[2]=='target_4':
                                    target_dict['target_4'].append([target_dt[ix], int(txt_split[3][:-1]), int(txt_split[4])])
                                elif txt_split[2]=='target_5':
                                    target_dict['target_5'].append([target_dt[ix], int(txt_split[3][:-1]), int(txt_split[4])])
                                elif txt_split[2]=='target_6':
                                    target_dict['target_6'].append([target_dt[ix], int(txt_split[3][:-1]), int(txt_split[4])])
                                elif txt_split[2]=='target_7':
                                    target_dict['target_7'].append([target_dt[ix], int(txt_split[3][:-1]), int(txt_split[4])])
                                elif txt_split[2]=='target_8':
                                    target_dict['target_8'].append([target_dt[ix], int(txt_split[3][:-1]), int(txt_split[4])])
                                elif txt_split[2]=='target_9':
                                    target_dict['target_9'].append([target_dt[ix], int(txt_split[3][:-1]), int(txt_split[4])])

                        for ky in target_dict.keys():
                            timeseries_data.append(go.Scatter(
                                x=np.array(target_dict[ky])[:,0],
                                y=np.array(target_dict[ky])[:,1],
                                name=ky+' x',
                                line={"shape": 'hv'},
                                mode='lines',
                                opacity=0.3,
                                visible='legendonly',
                                )
                            )
                            timeseries_data.append(go.Scatter(
                                x=np.array(target_dict[ky])[:,0],
                                y=np.array(target_dict[ky])[:,2],
                                name=ky+' y',
                                line={"shape": 'hv'},
                                mode='lines',
                                opacity=0.3,
                                visible='legendonly',
                                )
                            )

                    #print(timeseries_data)
                except:
                    eye_rand_trace = go.Scatter(
                                        x=np.array([datetime.fromtimestamp(i) for i in np.arange(10)]),
                                        y=[0]*10,
                                        name='Eye Trace file not found or parsed',
                                        mode='lines'
                                        )
                    timeseries_data.append(eye_rand_trace)

            if 'Mouse' in  file:
                try:
                    fname = glob.glob(op.join(file_loc, file))[0]
                    complete_file = read_hdf5(fname)
                    fdata = complete_file['device_data']
                    mdata = complete_file['marker']

                    len_df.at[0, 'Mouse'] = len(fdata['time_series'])

                    mouse_df = pd.DataFrame(fdata['time_series'][:,:], columns=['mouse_x','mouse_y','clicks'])
                    mouse_df['timestamps'] = fdata['time_stamps']
                    fs = int(1/np.median(np.diff(mouse_df.timestamps)))
                    #print('Mouse Sampling Rate =', fs)

                    dt_corr = int(float(mdata['time_series'][0][0].split('_')[-1]) - mdata['time_stamps'][0]) # datetime-correction factor : time correction offset for correcting LSL time to local time
                    mouse_datetime = np.array(mouse_df.timestamps.apply(lambda x: datetime.fromtimestamp(dt_corr+x)))
                    # mouse_datetime = np.array([datetime.fromtimestamp(dt_corr + timestamp) for timestamp in mouse_df.timestamps])

                    trace5 = go.Scatter(
                                x=mouse_datetime[::20],
                                y=mouse_df['mouse_x'][::20],
                                name='Mouse X : fs = '+str(fs),
                                mode='lines',
                                visible='legendonly'
                            )
                    timeseries_data.append(trace5)

                    trace6 = go.Scatter(
                                x=mouse_datetime[::10],
                                y=mouse_df['mouse_y'][::10],
                                name='Mouse Y',
                                mode='lines',
                                visible='legendonly'
                            )
                    timeseries_data.append(trace6)


                    marker_timestamp_for_valid_click = [mdata['time_stamps'][ix] for ix, txt in enumerate(mdata['time_series']) if 'mouse_valid_click' in txt[0]]
                    valid_click_index_in_mouse = []
                    for marker_timestamp in marker_timestamp_for_valid_click:
                        # find mousedata timestamp closest to valid click marker timestamp
                        valid_click_index_in_mouse.append(np.argmin(abs(fdata['time_stamps'] - marker_timestamp)))


                    marker_timestamp_for_cursor_in_target = [mdata['time_stamps'][ix] for ix, txt in enumerate(mdata['time_series']) if 'mouse_in_target' in txt[0] or "'in': 1" in txt[0]]
                    cursor_in_target_index_in_mouse = []
                    for marker_timestamp in marker_timestamp_for_cursor_in_target:
                        # find mousedata timestamp closest to cursor in target marker timestamp
                        cursor_in_target_index_in_mouse.append(np.argmin(abs(fdata['time_stamps'] - marker_timestamp)))
                    #print(cursor_in_target_index_in_mouse[0:2])

                    cursor_in_target_x_trace = go.Scatter(
                                x=mouse_datetime[cursor_in_target_index_in_mouse],
                                y=mouse_df.mouse_x[cursor_in_target_index_in_mouse],
                                name='Cursor in Target X',
                                mode='markers'
                            )
                    timeseries_data.append(cursor_in_target_x_trace)

                    cursor_in_target_y_trace = go.Scatter(
                                x=mouse_datetime[cursor_in_target_index_in_mouse],
                                y=mouse_df.mouse_y[cursor_in_target_index_in_mouse],
                                name='Cursor in Target Y',
                                mode='markers'
                            )
                    timeseries_data.append(cursor_in_target_y_trace)


                    valid_click_x_trace = go.Scatter(
                                x=mouse_datetime[valid_click_index_in_mouse],
                                y=mouse_df.mouse_x[valid_click_index_in_mouse],
                                name='Valid Clicks X',
                                mode='markers'
                            )
                    timeseries_data.append(valid_click_x_trace)

                    valid_click_y_trace = go.Scatter(
                                x=mouse_datetime[valid_click_index_in_mouse],
                                y=mouse_df.mouse_y[valid_click_index_in_mouse],
                                name='Valid Clicks Y',
                                mode='markers'
                            )
                    timeseries_data.append(valid_click_y_trace)
                except:
                    mouse_rand_trace = go.Scatter(
                                        x=np.array([datetime.fromtimestamp(i) for i in np.arange(10)]),
                                        y=[0.5]*10,
                                        name='Mouse Trace file not found or parsed',
                                        mode='lines',
                                        visible='legendonly'
                                        )
                    timeseries_data.append(mouse_rand_trace)

            if 'Mic' in file:
                try:
                    fname = glob.glob(op.join(file_loc, file))[0]
                    complete_file = read_hdf5(fname)
                    fdata = complete_file['device_data']
                    mdata = complete_file['marker']

                    len_df.at[0, 'Audio'] = len(fdata['time_series'])

                    dt_corr = int(float(mdata['time_series'][0][0].split('_')[-1]) - mdata['time_stamps'][0]) # datetime-correction factor : time correction offset for correcting LSL time to local time

                    audio_tstmp = fdata['time_stamps']
                    audio_ts = fdata['time_series']

                    chunk_len = audio_ts.shape[1]
                    # accounting for later addition of time in timeseries data - audio chunks are of length 1025 instead of 1024 
                    if chunk_len %2:
                        chunk_len -= 1
                        audio_ts_full = np.hstack(audio_ts[:,1:])
                    else:
                        audio_ts_full = np.hstack(audio_ts)

                    # chunk timestamps from end of chunck, add beginning
                    audio_tstmp = np.insert(audio_tstmp, 0, audio_tstmp[0] - np.diff(audio_tstmp).mean())
                    tstmps = []
                    for i in range(audio_ts.shape[0]):
                        tstmps.append(np.linspace(audio_tstmp[i], audio_tstmp[i+1], chunk_len))
                    audio_tstmp_full = np.hstack(tstmps)

                    audio_df = pd.DataFrame(audio_ts_full, columns=['amplitude'])
                    audio_df['timestamps'] = audio_tstmp_full
                    fs = int(1/np.median(np.diff(audio_df.timestamps)))
                    #print('Audio Sampling Rate =', fs)

                    audio_df = audio_df.iloc[::100,:]

                    audio_datetime = audio_df.timestamps.apply(lambda x: datetime.fromtimestamp(dt_corr+x))

                    trace7 = go.Scatter(
                                x=audio_datetime,
                                y=audio_df['amplitude'],
                                name='Audio Trace : fs = '+str(fs),
                                mode='lines',
                                opacity=0.5,
                                visible='legendonly',
                                showlegend=True,
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
                                x= audio_df.timestamps.apply(lambda x: x-audio_df.timestamps[0]), #audio_datetime,
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
                                        x=np.array([datetime.fromtimestamp(i) for i in np.arange(10)]),
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

            if any(sensor in file for sensor in mbient_sensors):
                try:
                    sen_suffix = file.split('Mbient_')[-1][:2]
                    
                    fname = glob.glob(op.join(file_loc, file))[0]
                    complete_file = read_hdf5(fname)
                    fdata = complete_file['device_data']
                    mdata = complete_file['marker']

                    len_df.at[0, 'IMU'] = len(fdata['time_series'])

                    imu_df = pd.DataFrame(fdata['time_series'][:,[1,2,3,4,5,6]], columns=['acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z'])
                    imu_df['timestamps'] = fdata['time_stamps']
                    fs = int(1/np.median(np.diff(imu_df.timestamps)))
                    #print('IMU Sampling Rate =', fs)

                    dt_corr = int(float(mdata['time_series'][0][0].split('_')[-1]) - mdata['time_stamps'][0]) # datetime-correction factor : time correction offset for correcting LSL time to local time
                    imu_datetime = imu_df.timestamps.apply(lambda x: datetime.fromtimestamp(dt_corr+x))

                    trace8 = go.Scatter(
                                x=imu_datetime,
                                y=imu_df['acc_x'],
                                name=f'{sen_suffix}_Acceleration X : fs = {str(fs)}',
                                mode='lines',
                                visible='legendonly',
                                yaxis="y2"
                            )
                    timeseries_data.append(trace8)

                    trace9 = go.Scatter(
                                x=imu_datetime,
                                y=imu_df['acc_y'],
                                name=f'{sen_suffix}_Acceleration Y',
                                mode='lines',
                                visible='legendonly',
                                yaxis="y2"
                            )
                    timeseries_data.append(trace9)

                    trace10 = go.Scatter(
                                x=imu_datetime,
                                y=imu_df['acc_z'],
                                name=f'{sen_suffix}_Acceleration Z',
                                mode='lines',
                                visible='legendonly',
                                yaxis="y2"
                            )
                    timeseries_data.append(trace10)

                    trace11 = go.Scatter(
                                x=imu_datetime,
                                y=imu_df['gyr_x'],
                                name=f'{sen_suffix}_Gyroscope X',
                                mode='lines',
                                visible='legendonly',
                                yaxis="y2"
                            )
                    timeseries_data.append(trace11)

                    trace12 = go.Scatter(
                                x=imu_datetime,
                                y=imu_df['gyr_y'],
                                name=f'{sen_suffix}_Gyroscope Y',
                                mode='lines',
                                visible='legendonly',
                                yaxis="y2"
                            )
                    timeseries_data.append(trace12)

                    trace13 = go.Scatter(
                                x=imu_datetime,
                                y=imu_df['gyr_z'],
                                name=f'{sen_suffix}_Gyroscope Z',
                                mode='lines',
                                visible='legendonly',
                                yaxis="y2"
                            )
                    timeseries_data.append(trace13)
                except:
                    imu_rand_trace = go.Scatter(
                                        x=np.array([datetime.fromtimestamp(i) for i in np.arange(10)]),
                                        y=[1.5]*10,
                                        name='IMU Trace file not found or parsed',
                                        mode='lines'
                                        )
                    timeseries_data.append(imu_rand_trace)
        
        print(len(timeseries_data),len(specgram_data))
        return timeseries_data, specgram_data, len_df


# --- Function to read RC Notes --- #
init_str = '''### Clinical Research Coordinator Notes\n \t'''
def read_rc_notes(task_session_value_split):
    text_markdown=init_str
    try:
        tsv = task_session_value_split.split('_')
        subj_id = tsv[0]
        session_date = tsv[1]
        task = '_'.join(tsv[3:])
        rc_notes_fname = subj_id+'_'+session_date+'/'+subj_id+'_'+session_date+'-'+task+'_task_1-notes.txt'
    except:
        text_markdown += 'Error parsing RC notes file name !! Check file naming convention\n \t'
        return text_markdown
    
    try:
        file_loc = get_file_loc(rc_notes_fname)
        fname = glob.glob(op.join(file_loc, rc_notes_fname))[0]
        text_markdown = init_str
        with open(fname) as rc_notes_file:
            for line in rc_notes_file.read():
                if "\n" in line:
                    text_markdown += "\n \t"
                else:
                    text_markdown += line
    except:
        text_markdown += 'Could not read RC notes file\n \t'

    return text_markdown.replace('[','\[').replace(']','\]')


# --- Creating all_files list --- #
#all_file_list = get_file_list(nb_data_df)

# --- Reading face landmark file --- #
face_landmark_data = read_hdf5(face_landmark_filename)['device_data']
face_landmark_timestamps = face_landmark_data['time_stamps']
face_landmark_points = face_landmark_data['time_series']
face_landmark_x = face_landmark_points[::100,:,0]
face_landmark_y = face_landmark_points[::100,:,1]

# --- Generating empty len_df --- #
len_df = generate_empty_file_len_df()

# --- Generating empty annotation_df --- #
anno_df = generate_empty_anno_df()


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
                                dcc.Markdown('''Select Primary Diagnosis''', style={'textAlign':'center'}),
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
                        html.H3('BARS Scores', style={'textAlign':'center'}),
                        html.Div(
                                dcc.Markdown('''
                                * BARS Scores (shown if available) for subject_ids in the table above
                                '''),style={'padding-left':'8%'},
                                ),
                        dash_table.DataTable(
                                id='bars_datatable',
                                style_table={'maxHeight': '400px','overflowY': 'scroll', 'overflowX': 'auto'},
                                style_data={'whiteSpace': 'normal', 'height': 'auto',}
                                ),
                        html.Hr(),
                        html.H3('Task Information', style={'textAlign':'center'}),
                        html.Div(
                                dcc.Markdown('''
                                * Select a task from dropdown to view data
                                * The table shows data files associated with the selected task
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
                                                style_cell={'textAlign': 'left'},
                                                ),
                                        ], className="nine columns", style={'width':'55%', 'display':'inline-block', 'padding-left':'3%', 'verticalAlign':'top'}),
                                ]),
                        html.Hr(),
                        html.H3('Explore Time Series', style={'textAlign':'center'}),
                        html.Div([
                                dcc.Markdown('''
                                * Click legend to toggle trace visibility
                                * Double click any legend to toggle all traces
                                * Selecting any of the Edit Plot radio buttons will re-render the plot automatically
                                ''',style={'padding-left':'8%'}),
                                dcc.Loading(id="loading-indicator", children=None, type="default", fullscreen=False),
                                html.Div('---'),
                                dcc.RadioItems(id='edit-plot-radio',
                                                options=[{'label': 'Edit Plot', 'value': 'True'},
                                                        {'label': 'Non-editable Plot', 'value': 'False'},],
                                                value='False'),
                                html.Div('---'),
                                dcc.Checklist(id='select-mbient-sensor-checkbox',
                                                options=[{'label': 'RH', 'value': 'RH'},
                                                        {'label': 'LH', 'value': 'LH'},
                                                        {'label': 'BK', 'value': 'BK'},
                                                        {'label': 'RF', 'value': 'RF'},
                                                        {'label': 'LF', 'value': 'LF'},],
                                                value=['RH']),
                                html.Div('---'),
                                dcc.Graph(id="timeseries_graph", config={'toImageButtonOptions': {'width': None,'height': None, 'format':'svg'}}),
                                ]),
                        html.Hr(),
                        html.H3('Meta Data', style={'textAlign':'center'}),
                        html.Div([
                                html.Div(dcc.Markdown(id="rc_notes_markdown", children=init_str), 
                                        style={'whiteSpace': 'pre-wrap',
                                                'outline':'1px black solid',
                                                'outline-offset': '-2px',
                                                'width':'46%',
                                                'display':'inline-block',
                                                'horizontalAlign':'left',
                                                'padding-left':'2%',
                                                'padding-right':'2%',
                                                }
                                ),
                                html.Div(
                                        dash_table.DataTable(
                                            id='file_length_datatable',
                                            data=len_df.to_dict('records'),
                                            columns=[{'name': col, 'id': col} for col in len_df.columns],
                                        ), style={'width':'46%',
                                                    'display':'inline-block',
                                                    'verticalAlign':'top',
                                                    'horizontalAlign':'right',
                                                    'padding-left':'2%',
                                                    'padding-right':'2%'}
                                )
                        ]),
                        html.Hr(),
                        html.H3('Annotations', style={'textAlign':'center'}),
                        html.Div(
                                dcc.Markdown('''
                                * Enter annotations in the text box and hit submit
                                * The name of the person making the annotations is required for submitting to the database
                                '''), style={'padding-left':'8%'},
                                ),
                        dcc.RadioItems(id='annotate-control-radio',
                                        options=[{'label': 'Annotate Patient', 'value': 'patient'},
                                                {'label': 'Annotate Control', 'value': 'control', 'disabled':True},],
                                                value='patient'),
                        html.Div([
                                html.Div([
                                        html.Div(dcc.Markdown('Enter annotations below:'), style={'horizontalAlign':'left'}),
                                        dcc.Textarea(
                                            id='annotation-text-box',
                                            value='',
                                            style={'width': '100%', 'height': 100, 'horizontalAlign':'left', 'resize': 'none'},
                                        ),
                                        html.Div([
                                                dcc.Markdown('Enter name of person annotating:',
                                                             style={'horizontalAlign':'left', 'padding-right':'1%', 'display': 'inline-block'}),
                                                dcc.Textarea(
                                                    id='annotator-name-box',
                                                    value='',
                                                    style={'width': '50%', 'height': 20, 'resize': 'none', 'verticalAlign': 'middle', 'display': 'inline-block'},
                                                    )
                                                ]),
                                        html.Div(html.Button('Submit to Database', id='annotation-submit-button', n_clicks=0), style={'padding-bottom':'2%'}),
                                        html.Div(id='successful-submission-text-area', style={'whiteSpace': 'pre-line'})
                                        ], style={'whiteSpace': 'pre-wrap',
                                                'outline':'1px black solid',
                                                'outline-offset': '-2px',
                                                'width':'46%',
                                                'display':'inline-block',
                                                'horizontalAlign':'left',
                                                'padding-left':'2%',
                                                'padding-right':'2%',
                                            }
                                ),
                                html.Div(
                                        dash_table.DataTable(
                                            id='annotation_datatable',
                                            data=anno_df.to_dict('records'),
                                            columns=[{'name': col, 'id': col} for col in anno_df.columns],
                                            # style_cell={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0},
                                            # tooltip_data=[
                                            #     {
                                            #         column: {'value': str(value), 'type': 'markdown'}
                                            #         for column, value in row.items()
                                            #     }
                                            #     for row in anno_df.to_dict('records')
                                            # ],
                                            # tooltip_duration=None,
                                            style_table={'maxHeight': '300px', 'overflowY': 'scroll', 'overflowX': 'scroll'},
                                            style_data={'whiteSpace': 'normal', 'height': 'auto'},
                                        ), style={'width':'46%',
                                                    'display':'inline-block',
                                                    'verticalAlign':'top',
                                                    'horizontalAlign':'right',
                                                    'padding-left':'2%',
                                                    'padding-right':'2%'}
                                )
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
                                                    updatemode='mouseup')
                                            ], style={'width':'79%', 'padding-left':'8%', 'padding-right':'5%', 'padding-bottom':'1%'})
                                        ], className="nine columns", style={'width':'70%', 'display':'inline-block'}),#, 'padding-left':'3%', 'padding-right':'2%'}),
                                html.Div([
                                    dcc.Graph(id="face_landmarks_with_slider", style={'horizontalAlign':'left'}),
                                        ], className="three columns", style={'width':'30%', 'display':'inline-block', 'verticalAlign':'center', 'horizontalAlign':'center'})#, 'padding-left':'2%', 'padding-right':'3%'
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
                                                * Refreshing this page also retrieves new data from the Neurobooth database  
                                                  
                                                Email [spatel@phmi.partners.org](mailto:spatel@phmi.partners.org) for bug reports and feedback''',
                                                style={'padding-left':'8%'}
                                                )
                                ]),
                        html.Hr(),
                        html.Div([dcc.Markdown('''Thank you for using Neurobooth Explorer''', style={'textAlign':'center'})]),
                        html.Hr(),
                    ])


@app.callback(
    [Output(component_id='bars_datatable', component_property='data'),
    Output(component_id='bars_datatable', component_property='columns'),
    Output(component_id='datatable', component_property='data'),
    Output(component_id='datatable', component_property='columns'),
    #Output(component_id='file_list_dropdown', component_property='options'),
    #Output(component_id='file_list_dropdown', component_property='value'),
    Output(component_id='task_session_dropdown', component_property='options'),
    Output(component_id='task_session_dropdown', component_property='value'),
    Output(component_id='select-mbient-sensor-checkbox', component_property='value')],
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
        scols = ['subject_id', 'session_date', 'gender', 'age', 'neurologist', 'primary_diagnosis', 'other_primary_diagnosis', 'secondary_diagnosis', 'diagnosis_notes', 'tasks']
        data_df = nb_data_df[nb_data_df['subject_id']==dropdown_value].astype('str').groupby(['session_date']).agg(lambda x: ' '.join(x.unique()))
        data_df.reset_index(inplace=True)
        data_df = data_df[scols]
    elif dropdown_value in session_date_list:
        dropdown_value_datetime = datetime.strptime(dropdown_value, '%Y-%m-%d')
        #subfiles = get_file_list(nb_data_df[nb_data_df['session_date']==dropdown_value_datetime.date()])
        session_files = get_task_session_files(nb_data_df[nb_data_df['session_date']==dropdown_value_datetime.date()])
        dcols = ['subject_id', 'gender', 'age', 'neurologist', 'primary_diagnosis', 'other_primary_diagnosis', 'secondary_diagnosis', 'diagnosis_notes', 'tasks']
        data_df = nb_data_df[nb_data_df['session_date']==dropdown_value_datetime.date()].astype('str').groupby(['subject_id']).agg(lambda x: ' '.join(x.unique()))
        data_df.reset_index(inplace=True)
        data_df = data_df[dcols]
    elif dropdown_value in task_list:
        #subfiles = get_file_list(nb_data_df[nb_data_df['tasks']==dropdown_value])
        session_files = get_task_session_files(nb_data_df[nb_data_df['tasks']==dropdown_value])
        tcols = ['subject_id', 'session_date', 'gender', 'age', 'neurologist', 'primary_diagnosis', 'other_primary_diagnosis', 'secondary_diagnosis', 'diagnosis_notes']
        data_df = nb_data_df[nb_data_df['tasks']==dropdown_value].astype('str').groupby(['subject_id']).agg(lambda x: ' '.join(x.unique()))
        data_df.reset_index(inplace=True)
        data_df = data_df[tcols]
    elif dropdown_value in clinical_list:
        filtered_indices = []
        for ix, diagnosis_list in enumerate(nb_data_df.primary_diagnosis):
            if dropdown_value in diagnosis_list:
                filtered_indices.append(ix)
        session_files = get_task_session_files(nb_data_df.iloc[filtered_indices])
        ccols = ['subject_id', 'session_date', 'gender', 'age', 'neurologist', 'primary_diagnosis', 'other_primary_diagnosis', 'secondary_diagnosis', 'diagnosis_notes', 'tasks']
        data_df = nb_data_df.iloc[filtered_indices].astype('str').groupby(['subject_id']).agg(lambda x: ' '.join(x.unique()))
        data_df.reset_index(inplace=True)
        data_df = data_df[ccols]
    
    # setting up BARS Scores datatable
    subj_id_list = data_df.subject_id.tolist()
    if len(subj_id_list):
        bars_data_df = bars_df[bars_df['subject_id'].isin(subj_id_list)]
    if len(bars_data_df)==0:
        bars_data_df.loc[0, 'subject_id']='BARS scores not available'
    bars_data = bars_data_df.to_dict('records')
    bars_columns = [{'name': col, 'id': col} for col in bars_df.columns]
    # --- #
    
    # Creating primary data table
    data = data_df.to_dict('records')
    columns = [{'name': col, 'id': col} for col in data_df.columns]

    # Generating file list
    #subfile_list = subfiles
    #opts = [ {'label': x, 'value': x} for x in subfile_list]
    #val = subfile_list[0]

    # Generating task session file dropdown list
    session_files = list(np.sort(session_files))
    session_files.insert(0, 'Select task to view data')
    task_session_opts = [ {'label': x, 'value': x} for x in session_files]
    task_session_val = session_files[0]

    #return data, columns, opts, val, task_session_opts, task_session_val
    return bars_data, bars_columns, data, columns, task_session_opts, task_session_val, ['RH']


@app.callback(
    [Output(component_id='task_session_file_datatable', component_property='data'),
    Output(component_id='task_session_file_datatable', component_property='columns'),
    Output('timeseries_graph', 'figure'),
    Output('timeseries_graph', 'config'),
    Output('specgram_graph', 'figure'),
    Output("loading-indicator","children"),
    Output("file_length_datatable", "data"),
    Output("rc_notes_markdown", "children")],
    [Input("task_session_dropdown", "value"),
    Input('edit-plot-radio', 'value'),
    Input('select-mbient-sensor-checkbox', 'value')])
def update_table(task_session_value, edit_plot_str, mbient_sensor_checklist):
    if edit_plot_str=='False':
        edit_plot = False
    elif edit_plot_str=='True':
        edit_plot = True
    
    if task_session_value=='Select task to view data':
        task_session_value=None
    
    mbient_sensors_to_plot = ['Mbient_'+sen_loc for sen_loc in mbient_sensor_checklist]

    task_files=[]
    try:
        tsv = task_session_value.split('_') # task session value split
        task_str = ''
        if tsv[-1]=='1':
            task_str = '_'.join(tsv[3:-2])
        else:
            task_str = '_'.join(tsv[3:-1])
        
        svg_prim_diag = ','.join(nb_data_df[nb_data_df['subject_id']==tsv[0]].primary_diagnosis.iloc[0])
        svg_prim_diag = svg_prim_diag.replace(' ','_')
        svg_filename = tsv[0]+'_'+tsv[1]+'_'+tsv[2]+'_'+task_str+'_'+svg_prim_diag

        plot_title = tsv[0]+'_'+tsv[1]+' : '+task_str+' : '+svg_prim_diag
        
        # get hdf5 files which match subject_id, session_date, and task name
        task_files = nb_data_df[(nb_data_df['subject_id']==tsv[0]) & (nb_data_df['session_date']==datetime.strptime(tsv[1], "%Y-%m-%d").date()) & (nb_data_df['tasks']=='_'.join(tsv[3:]))].hdf5_files.tolist()
        # then filter for task time because same task can be performed multiple times 
        task_files = [fil for fil in task_files if tsv[2] in fil]
        if len(task_files) > 0:
            task_files = np.sort(list(set(task_files)))
            for trg_task in ['pursuit', 'saccades_horizontal', 'saccades_vertical', 'gaze_holding']:
                if trg_task in task_str:
                    task_files = np.append(task_files, get_rnd_ctrl_et_hdf5('_'.join(tsv[3:]))+'CONTROL')
                    # Looking for other sessions
                    other_sessions_exist, other_session_file_list = check_for_other_sessions(tsv[0], tsv[1], '_'.join(tsv[3:]))
                    if other_sessions_exist:
                        task_files = np.append(task_files, other_session_file_list)
        else:
            task_files.append('No file found')
    except:
        plot_title = 'Accelerometer, Gyroscope, Eye Tracker and Mouse Time Series'
        svg_filename = plot_title
        task_files.append('No file found')
    task_file_df = pd.DataFrame(task_files, columns=['task_files'])
    
    ts_config = {'editable':edit_plot, 'toImageButtonOptions': {'width': None,'height': None, 'format':'svg', 'filename':svg_filename}}
    
    data = task_file_df.to_dict('records')
    columns = [{'name': col, 'id': col} for col in task_file_df.columns]

    rc_notes_markdown = init_str
    if task_session_value:
        rc_notes_markdown = read_rc_notes(task_session_value.split('_obs')[0])

    timeseries_data, specgram_data, len_df = parse_files(task_files, mbient_sensors_to_plot)
    length_data = len_df.to_dict('records')

    timeseries_layout = go.Layout(
                margin={'l': 50, 'b': 30, 't': 30, 'r': 50},
                height=520,
                xaxis={
                    'title':'Time',
                    'showgrid':False,
                    'showticklabels':True,
                    'tickmode':'array',
                },
                yaxis={'showgrid':False},
                title={
                    'text':plot_title,
                    'x':0.45,
                    'y':1,
                    'xanchor':'center',
                    'yanchor':'top',
                    'pad':{'t':10}
                },
                titlefont={'size':18},
                ## default template is "plotly" - comment out template to revert to default
                template="plotly_dark", ## 10 colors
                # template="ggplot2", ## 5 colors
                # template="seaborn", ## 10 colors
            )
    
    timeseries_fig=go.Figure(data=timeseries_data, layout=timeseries_layout)
    timeseries_fig.update_layout(
        yaxis=dict(autorange="reversed"),
        yaxis2=dict(
            anchor="x",
            overlaying="y",
            side="right"
        )
    )

    for filename in task_files:
        if ('_DSC_' in filename) and ('Eye' in filename):
            timeseries_fig = get_DSC_button_presses(filename, timeseries_fig)
        if ('MOT' in filename) and ('Eye' in filename):
            timeseries_fig = get_movement_task_start_end_times(filename, timeseries_fig)

    for movement_task in ['finger_nose', 'foot_tapping', 'sit_to_stand', 'altern_hand_mov']:
        for filename in task_files:
            if (movement_task in filename) & ('Mbient_RH' in filename):
                timeseries_fig = get_movement_task_start_end_times(filename, timeseries_fig)

    for stance_task in ['sitting',
                        'feet_apart_eyes_open', 'feet_apart_eyes_closed', 'feet_together_eyes_open', 'feet_together_eyes_closed',
                        'tandem_stance', 'stance_dominant_foot',
                        'tandem_walk', 'walking']:
        for filename in task_files:
            if (stance_task in filename) & ('Mbient_RH' in filename):
                timeseries_fig = get_movement_task_start_end_times(filename, timeseries_fig)

    for ocular_task in ['pursuit', 'fixation_no_target', 'gaze_holding', 'saccades_horizontal', 'saccades_vertical', 'DSC', 'hevelius', 'passage', 'picture_description']:
        for filename in task_files:
            if (ocular_task in filename) & ('Eye' in filename):
                timeseries_fig = get_movement_task_start_end_times(filename, timeseries_fig)

    for vocal_task in ['ahh', 'gogogo', 'lalala', 'mememe', 'pataka']:
        for filename in task_files:
            if (vocal_task in filename) & ('Mic' in filename):
                timeseries_fig = get_movement_task_start_end_times(filename, timeseries_fig)

    timeseries_fig.update_xaxes(tickangle=45, tickfont={'size':14}, showline=True, linewidth=1, linecolor='black', mirror=True, title_font={'size':18})
    timeseries_fig.update_yaxes(tickfont={'size':12})

    specgram_layout = go.Layout(
        title = 'Audio Trace',
        yaxis = dict(title = 'Amplitude'), # y-axis label
        xaxis = dict(title = 'Time (seconds)'), # x-axis label
        )

    specgram_fig = go.Figure(data=specgram_data, layout=specgram_layout)
    specgram_fig.update_layout(legend_x=1, legend_y=1)

    return data, columns, timeseries_fig, ts_config, specgram_fig, None, length_data, rc_notes_markdown


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
    global bars_df
    global sub_id_list
    global session_date_list
    global task_list
    global clinical_list
    #global all_file_list

    # Retrieving new data from database
    nb_data_df, bars_df, sub_id_list, session_date_list, task_list, clinical_list = rebuild_master_data_table(sql_query_cmd)
    
    # Generate new all_file_list from new nb_data_df
    #all_file_list = get_file_list(nb_data_df)

    updated_sub_id_list_options = [ {'label': x, 'value': x} for x in sub_id_list]
    updated_session_date_list_options = [ {'label': x, 'value': x} for x in session_date_list]
    updated_task_list_options = [ {'label': x, 'value': x} for x in task_list]
    updated_clinical_list_options = [ {'label': x, 'value': x} for x in clinical_list]

    #dt_str = datetime.fromtimestamp(int(n_clicks_timestamp/1000)).strftime('%Y-%m-%d, %H:%M:%S')
    dt_str = datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
    return 'New data was last retrieved at ' + dt_str, updated_sub_id_list_options, updated_session_date_list_options, updated_task_list_options, updated_clinical_list_options


@app.callback(
    Output('annotation_datatable', 'data'),
    Input("task_session_dropdown", "value")
)
def update_annotation_table(task_session_value):
    if (task_session_value is not None) and (task_session_value != 'Select task to view data'):
        tsv = task_session_value.split('_')
        subj_id = tsv[0]
        anno_data = nex_annotations_df[nex_annotations_df['subject_id']==subj_id].to_dict('records')
    else:
        anno_df = generate_empty_anno_df()
        anno_data = anno_df.to_dict('records')
    return anno_data


@app.callback(
    [Output('annotate-control-radio', 'options'),
    Output('annotate-control-radio', 'value')],
    Input('task_session_file_datatable', 'data'),
)
def manage_annotate_control_radio_buttons(tsv_data):
    disable_custom_anno = True
    if tsv_data is not None:
        for dicts in tsv_data:
            if dicts['task_files'].endswith('CONTROL'):
                disable_custom_anno = False

    anno_radio_options=[{'label': 'Annotate Patient', 'value': 'patient'},
                        {'label': 'Annotate Control', 'value': 'control', 'disabled':disable_custom_anno}]
    anno_radio_value = 'patient'
    return [anno_radio_options, anno_radio_value]


@app.callback(
    [Output('successful-submission-text-area', 'children'),
    Output('annotation-text-box', 'value'),
    Output('annotator-name-box', 'value'),
    Output('annotation_datatable', 'data')],
    Input('annotation-submit-button', 'n_clicks'),
    [State('annotate-control-radio', 'value'),
    State("task_session_dropdown", "value"),
    State('annotation-text-box', 'value'),
    State('annotator-name-box', 'value'),
    State('annotation_datatable', 'data'),
    State('task_session_file_datatable', 'data')]
)
def submit_to_database(n_clicks, annotate_control_str, task_session_value, annotation_text, annotator_name, anno_data, tsv_data):
    placeholder_text = ''
    placeholder_name = ''
    
    annotation_text = annotation_text.strip()
    annotator_name = annotator_name.strip().lower()

    if annotate_control_str=='patient':
        annotate_control = False
    elif annotate_control_str=='control':
        annotate_control = True

    if n_clicks > 0 and task_session_value=='Select task to view data':
        return [f'Invalid task session value received: "{task_session_value}"\nSelect a task before making annotation!', placeholder_text, placeholder_name, anno_data]

    if n_clicks > 0 and (len(annotator_name)==0 or len(annotation_text)==0):
        return [f'Annotation text and annotator name cannot be empty fields!', placeholder_text, placeholder_name, anno_data]
    
    if n_clicks > 0:
        try:
            tsv = task_session_value.split('_')
            if annotate_control:
                for dicts in tsv_data:
                    if dicts['task_files'].endswith('CONTROL'):
                        head, tail = op.split(dicts['task_files'])
                        ctrl_task_session_value = tail.split('_R001')[0]
                        tsv = ctrl_task_session_value.split('_')
        except:
            return [f'could not split task session value: "{task_session_value}"', placeholder_text, placeholder_name, anno_data]
        subj_id = tsv[0]
        session_date = tsv[1]
        session_time = tsv[2]
        session_datetime = datetime.strptime(session_date+' '+session_time, '%Y-%m-%d %Hh-%Mm-%Ss') #'2022-04-28 15h-19m-59s'
        task_id = '_'.join(tsv[3:])
        scols = ['subject_id', 'gender', 'age', 'primary_diagnosis']
        data_df = nb_data_df[nb_data_df['subject_id']==subj_id].astype('str').groupby(['session_date']).agg(lambda x: ' '.join(x.unique()))
        data_df.reset_index(inplace=True)
        data_df = data_df[scols]
        gender = data_df['gender'].iloc[0]
        age = int(data_df['age'].iloc[0])
        prim_diagnosis = data_df['primary_diagnosis'].iloc[0].replace('[','{').replace(']','}')
        source = 'neurobooth_explorer'
        username = request.authorization['username']
        submit_time_str = datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
        submit_time = datetime.strptime(submit_time_str, '%Y-%m-%d, %H:%M:%S')
        
        cols = ['subject_id', 'session_datetime', 'gender', 'age', 'primary_diagnosis', 'task_id',
                'annotation', 'annotator_name', 'annotation_source', 'source_user_id', 'annotation_submit_time']
        vals = [(subj_id, session_datetime, gender, age, prim_diagnosis, task_id,
                annotation_text, annotator_name, source, username, submit_time)]
        _ = insert_into_annotation_table(cols, vals)

        global nex_annotations_df
        nex_annotations_df = refresh_nex_annotations()
        anno_data = nex_annotations_df[nex_annotations_df['subject_id']==subj_id].to_dict('records')

        return [f'Successfully added annotation to database table!', placeholder_text, placeholder_name, anno_data]

    # default return
    return [None, placeholder_text, placeholder_name, anno_data]


if __name__ == '__main__':
    # context = ('/home/sid/.db_secrets/nb_cert.pem', '/home/sid/.db_secrets/nb_key.pem')
    context = ('/usr/etc/certs/server.crt', '/usr/etc/certs/server.key')
    app.run_server(host='0.0.0.0', port='8050', debug=True, ssl_context=context)
    # app.run_server(host='127.0.0.1', port='8050', debug=True)
    # app.run_server(host='0.0.0.0', port='8050', debug=True)
