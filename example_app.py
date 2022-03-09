import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

color_list = ['Gold', 'MediumTurquoise', 'LightGreen']

app = dash.Dash(__name__)

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
