import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Sample data for years
year_list = [i for i in range(1980, 2024, 1)]

# Create Dash app
data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv', parse_dates=['Date'])
data = data.rename(columns={'Automobile_Sales': 'Sales', 'Vehicle_Type': 'Car_Type'})
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

# Layout of the app
app.layout = html.Div([
    html.H1("Yearly vs. Recession Statistics Dashboard"),

    html.Label("Select Statistics Type:"),
    dcc.Dropdown(
        id='select-statistics',
        options=[
            {'label': 'Yearly Statistics', 'value': 'yearly'},
            {'label': 'Recession Statistics', 'value': 'recession'},
        ],
        value='yearly',
        clearable=False
    ),

    html.Label("Select Year:"),
    dcc.Dropdown(
        id='select-year',
        options=[{'label': year, 'value': year} for year in year_list],
        value='2022',
        disabled=False  # Enabled by default
    ),

    html.Div([
        dcc.Graph(id='line-chart'),
        dcc.Graph(id='bar-chart'),
        dcc.Graph(id='pie-chart'),
    ]),
    html.Div(id='output-container'),
])

# Callback to update year dropdown based on selected statistics type
@app.callback(
    Output('select-year', 'disabled'),
    Output('select-year', 'value'),
    Input('select-statistics', 'value')
)
def update_year_dropdown(selected_statistics):
    if selected_statistics == 'yearly':
        # Enable the year dropdown and set a default value
        return False, '2022'
    elif selected_statistics == 'recession':
        # Disable the year dropdown and clear its value
        return True, None

# Callback to display selected options and update charts
@app.callback(
    [Output('output-container', 'children'),
     Output('line-chart', 'figure'),
     Output('bar-chart', 'figure'),
     Output('pie-chart', 'figure')],
    [Input('select-statistics', 'value'),
     Input('select-year', 'value')]
)
def update_charts(selected_statistics, selected_year):
    if selected_statistics == 'yearly':
        # Filter data for the selected year
        yearly_data = data[data['Date'].dt.year == int(selected_year)]

        # Line chart for monthly sales across the selected year
        line_chart = px.line(
            yearly_data,
            x='Date',
            y='Sales',
            title=f'Monthly Sales Across {selected_year}'
        )

        # Bar chart for monthly sales grouped by car type
        bar_chart = px.bar(
            yearly_data,
            x='Date',
            y='Sales',
            color='Car_Type',
            title=f'Monthly Sales Grouped by Car Type in {selected_year}'
        )

        # Pie chart for advertisement expenditure grouped by car type
        pie_chart = px.pie(
            yearly_data,
            names='Car_Type',
            values='Advertising_Expenditure',
            title=f'Advertising Expenditure Grouped by Car Type in {selected_year}'
        )

        # String for output-container.children
        output_text = 'Yearly Statistics'

        return [output_text, line_chart, bar_chart, pie_chart]
    elif selected_statistics == 'recession':
        # Filter data for recession periods
        recession_data = data[data['Recession'] == 1]

        # Line chart for monthly sales across recession periods
        line_chart2 = px.line(
            recession_data,
            x='Date',
            y='Sales',
            title=f'Monthly Sales Across Recession Periods'
        )

        # Bar chart for monthly sales grouped by car type in recession periods
        bar_chart2 = px.bar(
            recession_data,
            x='Date',
            y='Sales',
            color='Car_Type',
            title=f'Monthly Sales Grouped by Car Type in Recession Periods'
        )

        # Pie chart for advertising expenditure grouped by car type in recession periods
        pie_chart2 = px.pie(
            recession_data,
            names='Car_Type',
            values='Advertising_Expenditure',
            title=f'Advertising Expenditure Grouped by Car Type in Recession Periods'
        )

        # String for output-container.children
        output_text = 'Recession Statistics'

        return [output_text, line_chart2, bar_chart2, pie_chart2]


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)