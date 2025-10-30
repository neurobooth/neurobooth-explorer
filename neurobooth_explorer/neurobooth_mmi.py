
import os
import pandas as pd

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
            or 'gaze_density' in file \
            or 'word_segmentation' in file \
            or 'landmarks' in file \
            or 'gaze_pixels' in file:
            session_files.append(file)

    return session_files


sub_id_list = ['Select Session', '101031_2025-10-08', '101033_2025-10-21', '101034_2025-10-21']
task_list = ['Select Task', 'DSC', 'passage', 'picture_description']


app = dash.Dash(__name__)
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)

app.layout = html.Div([
                        html.H1('Neurobooth MMI', style={'textAlign':'center'}),
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
                                                style_table={'maxHeight': '150px','overflowY': 'scroll', 'overflowX': 'auto'},
                                                style_cell={'textAlign': 'left'},
                                                ),
                                        ], className="nine columns", style={'width':'55%', 'display':'inline-block', 'padding-left':'3%', 'verticalAlign':'top'}),
                                ]),
                        html.Hr(),
                        html.Div([
                                html.Div([
                                        html.Video(
                                            id="right_gaze_video",
                                            controls=True,
                                            autoPlay=False,
                                            style={"width": "95%", "border": "1px solid #ccc", "borderRadius": "8px"},
                                            ),
                                        ], className="nine columns", style={'width':'37%', 'display':'inline-block', 'padding-left':'3%', 'verticalAlign':'top'}),
                                html.Div([
                                        html.Video(
                                            id="left_gaze_video",
                                            controls=True,
                                            autoPlay=False,
                                            style={"width": "95%", "border": "1px solid #ccc", "borderRadius": "8px"},
                                            ),
                                        ], className="nine columns", style={'width':'37%', 'display':'inline-block', 'padding-left':'3%', 'verticalAlign':'top'}),
                                html.Div([
                                        html.Video(
                                            id="face_landmark_video",
                                            controls=True,
                                            autoPlay=False,
                                            style={"width": "95%", "border": "1px solid #ccc", "borderRadius": "8px"},
                                            ),
                                        ], className="nine columns", style={'width':'19%', 'display':'inline-block', 'padding-left':'1%', 'verticalAlign':'top'}),
                                ]),
                        html.Hr(),
                        html.Div([
                                html.Div([
                                    html.Audio(
                                        id="speech_audio",
                                        src='',
                                        controls=True,
                                        style={"width": "100%", "marginTop": "10px"}
                                        ),
                                ], className="nine columns", style={'width':'80%', 'display':'inline-block', 'padding-left':'10%', 'verticalAlign':'top'}),
                                ]),
                        html.Hr(),
                        html.Div([
                                html.Div([
                                        html.Video(
                                            id="right_eye_crop_video",
                                            controls=True,
                                            autoPlay=False,
                                            style={"width": "95%", "border": "1px solid #ccc", "borderRadius": "8px"},
                                            ),
                                        ], className="nine columns", style={'width':'45%', 'display':'inline-block', 'padding-left':'5%', 'verticalAlign':'top'}),
                                html.Div([
                                        html.Video(
                                            id="left_eye_crop_video",
                                            controls=True,
                                            autoPlay=False,
                                            style={"width": "95%", "border": "1px solid #ccc", "borderRadius": "8px"},
                                            ),
                                        ], className="nine columns", style={'width':'45%', 'display':'inline-block', 'padding-left':'5%', 'verticalAlign':'top'}),
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
    Output("speech_audio", "src"),
    [Input("task_session_file_datatable", "data"),
     Input("subject_id_dropdown", "value")]
)
def update_audio_src(session_files, session_id):
    if not session_files or session_id == 'Select Session':
        return None
    else:
        # print(session_files, session_id)
        if len(session_files)==1:
            return None
        vol = get_file_loc(session_id)
        audio_file = [file_dict['task_files'] for file_dict in session_files if file_dict['task_files'].endswith('.wav')]
        audio_file = audio_file[0]
        if audio_file:
            path = os.path.join(vol, session_id)
            path = os.path.join(path, audio_file)
            print(path)
            return app.get_asset_url(path)

# # Serve audio files from local directory
# @app.server.route(f"/{AUDIO_FOLDER}/<path:filename>")
# def serve_audio(filename):
#     return send_from_directory(AUDIO_FOLDER, filename)



if __name__ == '__main__':
    context = ('/usr/etc/certs/server.crt', '/usr/etc/certs/server.key')
    app.run_server(host='0.0.0.0', port='8050', debug=True, ssl_context=context)

