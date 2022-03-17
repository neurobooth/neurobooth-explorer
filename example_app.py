import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import dash_auth

import configparser
auth_config = configparser.ConfigParser()
auth_config_file_loc = 'C:\\Users\\siddh\\.db_secrets\\users.txt'
auth_config.read(auth_config_file_loc)
USERNAME_PASSWORD_PAIRS = dict()
for ky in auth_config[auth_config.sections()[0]]:
    USERNAME_PASSWORD_PAIRS[ky] = auth_config[auth_config.sections()[0]][ky]

color_list = ['Gold', 'MediumTurquoise', 'LightGreen']

app = dash.Dash(__name__)
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)

app.layout = html.Div([
                        html.Div([dcc.Markdown('''Select Color''', style={'textAlign':'left'})]),
                        dcc.Dropdown(
                                id="dropdown",
                                options=[ {'label': x, 'value': x} for x in color_list],
                                value=color_list[0],
                                clearable=False,
                                ),
                        dcc.Graph(id="graph"),
                        html.Hr(),
                    ])

@app.callback(
    Output("graph", "figure"), 
    [Input("dropdown", "value")])
def display_color(color):
    fig1 = go.Figure(
        data=go.Bar(y=[2, 3, 1], marker_color=color))
    return fig1

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port='8050', debug=True)
