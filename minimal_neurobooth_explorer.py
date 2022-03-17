from neurobooth_terra import Table

import psycopg2
from sshtunnel import SSHTunnelForwarder

import configparser

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash import callback_context
import dash_table

import numpy as np
import pandas as pd

from os import walk
from os.path import join

from datetime import datetime

###########################################
### Comment out section depending on os ###
###########################################

### WINDOWS Legion ###

# --- ssh, db cred, variable assignment for Windows 11 on Legion --- #
config_file_loc = 'C:\\Users\\siddh\\.db_secrets\\db_secrets.txt'
config = configparser.ConfigParser()
config.read(config_file_loc)

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


# ### LINUX P620 ###

# # --- ssh, db cred, variable assignment for Ubuntu on P620 Workstation --- #
# config_file_loc = '~/.db_secrets/db_secrets.txt'
# config = configparser.ConfigParser()
# config.read(config_file_loc)

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
# host = secrets['database']['host']
# port = 5432
# conn = psycopg2.connect(database=database, 
#                         user=secrets['database']['user'],
#                         password=secrets['database']['pass'],
#                         host=host,
#                         port=port)


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
                        html.Div([
                                dcc.Markdown('''Click button to retrieve new data from Neurobooth database'''),
                                html.Button('Get New Data', id='db_button', n_clicks_timestamp=0),
                                html.Div(id='button_container', children='Click button to retreive new data', style={'padding-top':'2%'})
                                ], style={'textAlign':'center', 'width': '50%', 'margin':'auto', 'verticalAlign': 'middle'}),
                        html.Hr(),
                        html.Div([dcc.Markdown('''
                                                Hints:
                                                * Refresh page to retrieve latest data from the Neurobooth database
                                                * Double click anywhere in the plot area to reset view''', style={'padding-left':'8%'})]),
                        html.Hr(),
                        html.Div([dcc.Markdown('''
                                                Thank you for using Neurobooth Explorer''', style={'textAlign':'center'})]),
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

@app.callback(
    Output('button_container', 'children'),
    Input('db_button', 'n_clicks_timestamp'))
def on_button_click(n_clicks_timestamp):
    dt_str = datetime.fromtimestamp(int(n_clicks_timestamp/1000)).strftime('%Y-%m-%d, %H:%M:%S')
    return 'The connection to database was last refreshed at ' + dt_str

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port='8050', debug=True)