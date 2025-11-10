
import os
import pandas as pd
import base64

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

import dash_auth

import credential_reader

# ==== Read Credentials ==== #
dataflow_args = credential_reader.read_dataflow_configs()
file_locs = dataflow_args['suitable_volumes']

db_args, ssh_args = credential_reader.read_db_secrets()

USERNAME_PASSWORD_PAIRS = credential_reader.get_user_pass_pairs()

# ========================== #


def get_file_loc(session_id):
    for vol in file_locs:
        if os.path.exists(os.path.join(vol, session_id)):
            return vol


def get_session_files(session_id, task):
    vol = get_file_loc(session_id)
    all_files = []
    for _, _, files in os.walk(os.path.join(vol, session_id)):
        all_files.extend(files)
        break
    task_files = [file for file in all_files if task in file]
    
    session_files = []
    for file in task_files:
        if file.endswith('wav') \
            or 'gaze.mp4' in file \
            or 'word_segmentation' in file \
            or 'landmarks.mp4' in file \
            or 'eye_crop.mp4' in file:
            session_files.append(file)

    return session_files


sub_id_list = ['Select Session', '101031_2025-10-08', '101033_2025-10-21', '101034_2025-10-21']
task_list = ['Select Task', 'DSC', 'passage', 'picture_description']

video_ids = ["right_gaze_video", "left_gaze_video", "face_landmark_video", "right_eye_crop_video", "left_eye_crop_video"]
audio_id = "main-speech_audio"

app = dash.Dash(__name__)
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)

app.layout = html.Div([
                        html.H1('Neuroflix', style={'textAlign':'center'}),
                        html.H4('Multi Modal Integration of neurobooth data for insight and discovery', style={'textAlign':'center'}),
                        html.Hr(),
                        html.Div([
                                dcc.Markdown('''Select Subject ID''', style={'textAlign':'center'}),
                                dcc.Dropdown(id="subject_id_dropdown",
                                            options=[ {'label': x, 'value': x} for x in sub_id_list],
                                            value=sub_id_list[0],
                                            clearable=False),
                                ], className="three columns", style={'width':'20%', 'display':'inline-block', 'padding-left':'7%', 'padding-right':'2%'}),
                        html.Div([
                                dcc.Markdown('''Select Task''', style={'textAlign':'center'}),
                                dcc.Dropdown(id="task_id_dropdown",
                                            clearable=False),
                                ], className="three columns", style={'width':'20%', 'display':'inline-block', 'padding-left':'7%', 'padding-right':'2%'}),
                        html.Hr(),
                        html.Div([
                                html.Div([
                                        dash_table.DataTable(
                                                id='task_session_file_datatable',
                                                style_table={'maxHeight': '100px','overflowY': 'scroll', 'overflowX': 'auto'},
                                                style_cell={'textAlign': 'left'},
                                                ),
                                        ], className="nine columns", style={'width':'70%', 'display':'inline-block', 'padding-left':'7%', 'verticalAlign':'top'}),
                                ]),
                        html.Hr(),
                        html.Div([
                                html.Button("STOP", id="stop-btn", n_clicks=0, style={'margin': '10px'}),
                                html.Button("PLAY", id="play-btn", n_clicks=0, style={'margin': '10px'}),
                                html.Button("PAUSE", id="pause-btn", n_clicks=0, style={'margin': '10px'}),
                                ], style={'textAlign':'center', 'width': '50%', 'margin':'auto', 'verticalAlign': 'middle'}),
                        html.Div([
                                html.Div([
                                        html.Video(
                                            id="right_gaze_video",
                                            controls=True,
                                            autoPlay=True,
                                            style={"width": "95%", "border": "1px solid #ccc", "borderRadius": "8px"},
                                            ),
                                        ], className="nine columns", style={'width':'37%', 'display':'inline-block', 'padding-left':'3%', 'verticalAlign':'top'}),
                                html.Div([
                                        html.Video(
                                            id="left_gaze_video",
                                            controls=True,
                                            autoPlay=True,
                                            style={"width": "95%", "border": "1px solid #ccc", "borderRadius": "8px"},
                                            ),
                                        ], className="nine columns", style={'width':'37%', 'display':'inline-block', 'padding-left':'3%', 'verticalAlign':'top'}),
                                html.Div([
                                        html.Video(
                                            id="face_landmark_video",
                                            controls=True,
                                            autoPlay=True,
                                            style={"width": "95%", "border": "1px solid #ccc", "borderRadius": "8px"},
                                            ),
                                        ], className="nine columns", style={'width':'19%', 'display':'inline-block', 'padding-left':'1%', 'verticalAlign':'center'}),
                                ]),
                        html.Hr(),
                        html.Div([
                                html.Div([
                                        html.Video(
                                            id="right_eye_crop_video",
                                            controls=True,
                                            autoPlay=True,
                                            style={"width": "95%", "border": "1px solid #ccc", "borderRadius": "8px"},
                                            ),
                                        ], className="nine columns", style={'width':'45%', 'display':'inline-block', 'padding-left':'5%', 'verticalAlign':'top'}),
                                html.Div([
                                        html.Video(
                                            id="left_eye_crop_video",
                                            controls=True,
                                            autoPlay=True,
                                            style={"width": "95%", "border": "1px solid #ccc", "borderRadius": "8px"},
                                            ),
                                        ], className="nine columns", style={'width':'45%', 'display':'inline-block', 'padding-left':'5%', 'verticalAlign':'top'}),
                                ]),
                        html.Hr(),
                        html.Div([
                                html.Div([
                                    html.Audio(
                                        id="speech_audio",
                                        controls=True,
                                        autoPlay=True,
                                        style={"width": "100%", "marginTop": "10px"}
                                        ),
                                ], className="nine columns", style={'width':'80%', 'display':'inline-block', 'padding-left':'10%', 'verticalAlign':'top'}),
                                ]),
                    ])


@app.callback(
    [Output(component_id='task_id_dropdown', component_property='options'),
    Output(component_id='task_id_dropdown', component_property='value'),],
    [Input("subject_id_dropdown", "value"),])
def update_table(subid_value):

    dropdown_value = subid_value
    
    if dropdown_value == 'Select Session':
        return [], None

    task_opts = [ {'label': x, 'value': x} for x in task_list]
    task_val = task_list[0]

    return task_opts, task_val


@app.callback(
    [Output(component_id='task_session_file_datatable', component_property='data'),
    Output(component_id='task_session_file_datatable', component_property='columns'),],
    [Input("task_id_dropdown", "value"),Input("subject_id_dropdown", "value")])
def update_table(taskid_value, subjid_value):

    dropdown_value=taskid_value

    if dropdown_value == None or dropdown_value == 'Select Task':
        task_file_df = pd.DataFrame(['Please select a task'], columns=['task_files'])
        data = task_file_df.to_dict('records')
        columns = [{'name': col, 'id': col} for col in task_file_df.columns] 
    else:
        session_files = get_session_files(subjid_value, dropdown_value)
        task_file_df = pd.DataFrame(session_files, columns=['task_files'])
        data = task_file_df.to_dict('records')
        columns = [{'name': col, 'id': col} for col in task_file_df.columns]
    return data, columns


@app.callback(
    [Output("speech_audio", "src"),
     Output("right_gaze_video", "src"),
     Output("left_gaze_video", "src"),
     Output("face_landmark_video", "src"),
     Output("right_eye_crop_video", "src"),
     Output("left_eye_crop_video", "src"),],
    [Input("task_session_file_datatable", "data"),
     Input("subject_id_dropdown", "value")]
)
def update_audio_video_src(session_files, session_id):
    
    expected_returns = None, None, None, None, None, None

    def _extract_fname(end_str: str) -> str:
        '''Matched end string of a file and returns that file'''
        fname = [file_dict['task_files'] for file_dict in session_files if file_dict['task_files'].endswith(end_str)]
        if fname:
            return fname[0]
        else:
            return None

    def _convert_to_b64_str(video_fname, path) -> str:
        '''This same function converts wav files to b64 strings as well'''
        if not video_fname:
            return None
        video_path = os.path.join(path, video_fname)
        with open(video_path, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        return video_base64
    
    if not session_files or session_id == 'Select Session':
        return expected_returns
    else:
        # print(session_files, session_id)
        if len(session_files)==1:
            return expected_returns
        
        vol = get_file_loc(session_id)
        path = os.path.join(vol, session_id)

        # --- extracting filenames --- #
        wav_file = _extract_fname('.wav')

        right_gaze = _extract_fname('right_gaze.mp4')
        left_gaze  = _extract_fname('left_gaze.mp4')

        face_landmarks = _extract_fname('landmarks.mp4')

        right_eye_crop = _extract_fname('right_eye_crop.mp4')
        left_eye_crop  = _extract_fname('left_eye_crop.mp4')
        # ---------------------------- #
        
        # --- Encoding Videos --- #
        wav_file_b64 = _convert_to_b64_str(wav_file, path)

        right_gaze_b64 = _convert_to_b64_str(right_gaze, path)
        left_gaze_b64  = _convert_to_b64_str(left_gaze, path)

        face_landmarks_b64 = _convert_to_b64_str(face_landmarks, path)

        right_eye_crop_b64 = _convert_to_b64_str(right_eye_crop, path)
        left_eye_crop_b64  = _convert_to_b64_str(left_eye_crop, path)
        # ----------------------- #
        
        
        return f"data:audio/wav;base64,{wav_file_b64}", \
            f"data:video/mp4;base64,{right_gaze_b64}", \
            f"data:video/mp4;base64,{left_gaze_b64}", \
            f"data:video/mp4;base64,{face_landmarks_b64}", \
            f"data:video/mp4;base64,{right_eye_crop_b64}", \
            f"data:video/mp4;base64,{left_eye_crop_b64}", \


# # PLAY — resumes any paused media (does not reset currentTime)
# app.clientside_callback(
#     """
#     function(n) {
#         if (n === 0) return;
#         const vids = %s;
#         const aud = '%s';
#         vids.forEach(id => {
#             const v = document.getElementById(id);
#             if (v && v.paused) {
#                 v.play();
#             }
#         });
#         const a = document.getElementById(aud);
#         if (a && a.paused) {
#             a.play();
#         }
#         return '';
#     }
#     """ % (video_ids[0], audio_id),
#     Output("play-btn", "n_clicks"),
#     Input("play-btn", "n_clicks"),
# )

# # PAUSE — pauses media if playing, does nothing otherwise
# app.clientside_callback(
#     """
#     function(n) {
#         if (n === 0) return;
#         const vids = %s;
#         const aud = '%s';
#         vids.forEach(id => {
#             const v = document.getElementById(id);
#             if (v && !v.paused) {
#                 v.pause();
#             }
#         });
#         const a = document.getElementById(aud);
#         if (a && !a.paused) {
#             a.pause();
#         }
#         return '';
#     }
#     """ % (video_ids, audio_id),
#     Output("pause-btn", "n_clicks"),
#     Input("pause-btn", "n_clicks"),
# )


# # STOP — pauses all media and resets to start (currentTime = 0)
# app.clientside_callback(
#     """
#     function(n) {
#         if (n === 0) return;
#         const vids = %s;
#         const aud = '%s';
#         vids.forEach(id => {
#             const v = document.getElementById(id);
#             if (v) {
#                 v.pause();
#                 v.currentTime = 0;
#             }
#         });
#         const a = document.getElementById(aud);
#         if (a) {
#             a.pause();
#             a.currentTime = 0;
#         }
#         return '';
#     }
#     """ % (video_ids, audio_id),
#     Output("stop-btn", "n_clicks"),
#     Input("stop-btn", "n_clicks"),
# )


if __name__ == '__main__':
    context = ('/usr/etc/certs/server.crt', '/usr/etc/certs/server.key')
    app.run_server(host='0.0.0.0', port='8501', debug=True, ssl_context=context)

