from neurobooth_terra import list_tables, Table

import psycopg2
from sshtunnel import SSHTunnelForwarder

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash import callback_context
import plotly.graph_objects as go
import dash_table

import numpy as np
import pandas as pd

from os import walk
from os.path import join
import h5py

# --- Setting db access args --- #
ssh_args = dict(
        ssh_address_or_host='neurodoor.nmr.mgh.harvard.edu',
        ssh_username='sp1022',
        host_pkey_directories='C:\\Users\\siddh\\.ssh',
        remote_bind_address=('192.168.100.1', 5432),
        local_bind_address=('localhost', 6543),
        allow_agent=False
)

db_args = dict(
    database='neurobooth', user='neurovisualizer', password='edwardtufte',
    # host='localhost'
)

# --- Accessing subject_ids --- #
with SSHTunnelForwarder(**ssh_args) as tunnel:
    with psycopg2.connect(port=tunnel.local_bind_port,
                          host=tunnel.local_bind_host,
                          **db_args) as conn:
        subject_df = Table('subject', conn).query()

sub_id = subject_df.index.tolist()
sub_id = np.sort(sub_id)

# --- Accessing data collection dates --- #
with SSHTunnelForwarder(**ssh_args) as tunnel:
    with psycopg2.connect(port=tunnel.local_bind_port,
                          host=tunnel.local_bind_host,
                          **db_args) as conn:
        sensor_file_log_df = Table('sensor_file_log', conn).query()

sensor_file_log_df['date'] = [str(i.date()) for i in sensor_file_log_df.file_start_time]
dates = list(dict.fromkeys([str(i) for i in pd.Series([i.date() for i in sensor_file_log_df.file_start_time]).sort_values()]))

# --- Accessing clinical data --- #
with SSHTunnelForwarder(**ssh_args) as tunnel:
    with psycopg2.connect(port=tunnel.local_bind_port,
                          host=tunnel.local_bind_host,
                          **db_args) as conn:
        clinical_df = Table('clinical', conn).query()

clinical = ['Ataxia-Telangiectasia','Spino Cerebellar Ataxia', 'Parkinsonism']

app = dash.Dash(__name__)

app.layout = html.Div([
                        html.H1('Neurobooth Explorer', style={'textAlign':'center'}),
                        html.H4('A web-app to browse Neurobooth data', style={'textAlign':'center'}),
                        html.Hr(),
                        html.Div(
                                dcc.Markdown('''
                                * Use the dropdown menus to select subject_id, date or clinical indication
                                * The table displays context specific information depending on dropdown selection
                                '''),style={'padding-left':'8%'},
                                ),
                        html.Div([
                                dcc.Markdown('''Select Subject ID''', style={'textAlign':'center'}),
                                dcc.Dropdown(id="subject_id_dropdown",
                                            options=[ {'label': x, 'value': x} for x in sub_id],
                                            value=sub_id[0],
                                            clearable=False),
                                ], className="four columns", style={'width':'30%', 'display':'inline-block', 'padding-left':'4%', 'padding-right':'1%'}),
                        html.Div([
                                dcc.Markdown('''Select Date''', style={'textAlign':'center'}),
                                dcc.Dropdown(id="date_dropdown",
                                            options=[ {'label': x, 'value': x} for x in dates],
                                            value=dates[0],
                                            clearable=False),
                                ], className="four columns", style={'width':'30%', 'display':'inline-block', 'padding-right':'1%'}),
                        html.Div([
                                dcc.Markdown('''Select Clinical Indication''', style={'textAlign':'center'}),
                                dcc.Dropdown(id="clinical_dropdown",
                                            options=[ {'label': x, 'value': x} for x in clinical],
                                            value=clinical[0],
                                            clearable=False),
                                ], className="four columns", style={'width':'30%', 'display':'inline-block'}),
                        dash_table.DataTable(
                                id='datatable',
                                style_table={'maxHeight': '400px','overflowY': 'scroll'}
                                ),
                        html.Hr(),
                        html.Div(
                                dcc.Markdown('''
                                * Select file from the dropdown menu below
                                * The menu changes depending on selection made in the panel above
                                '''),style={'padding-left':'8%'},
                                ),
                        html.Hr(),
                        html.Div([dcc.Markdown('''
                                                Hints:
                                                * Refresh page to retrieve latest data from the Neurobooth database
                                                * Double click anywhere in the plot area to reset view''', style={'padding-left':'8%'})]),
                        html.Hr(),
                        html.Div([dcc.Markdown('''Thank you for using Neurobooth Explorer''', style={'textAlign':'center'})]),
                        html.Hr(),
                    ])


@app.callback(
    [Output(component_id='datatable', component_property='data'),
    Output(component_id='datatable', component_property='columns')],
    [Input("subject_id_dropdown", "value"),
    Input("date_dropdown", "value"),
    Input("clinical_dropdown", "value")])
def update_table(subid_value, date_value, clinical_value):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'subject_id_dropdown' in changed_id:
        dropdown_value = subid_value
    elif 'date_dropdown' in changed_id:
        dropdown_value = date_value
    elif 'clinical_dropdown' in changed_id:
        dropdown_value = clinical_value
    
    #print(dropdown_value)
    
    data_df = pd.DataFrame()
    
    if dropdown_value in sub_id:
        data_df = subject_df[subject_df.index == dropdown_value]
    elif dropdown_value in dates:
        data_df = sensor_file_log_df[sensor_file_log_df['date'] == dropdown_value]
        data_df = data_df[data_df.columns[:-1]]
    elif dropdown_value in clinical:
        data_df = clinical_df
    
    #print(data_df)
    data = data_df.to_dict('records')
    columns = [{'name': col, 'id': col} for col in data_df.columns]
    
    return data, columns

app.run_server(debug=True)