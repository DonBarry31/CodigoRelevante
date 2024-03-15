import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Sample data (replace this with your actual data)
data = pd.read_excel('PowerBI\DatosEjercicios.xlsx', engine='openpyxl')
df = pd.DataFrame(data)
app = dash.Dash(__name__)

# Extract unique muscles from the 'Musculos Empleados' column
all_muscles_unique = [muscle.title() for muscle in df['Musculos empleados']]
all_muscles = set(','.join(all_muscles_unique).replace(' ', '').split(','))
sorted_all_muscles = sorted(all_muscles)
replace_dict={'DeltoideAnterior':'Deltoide Anterior', 'DeltoidePosterior': 'Deltoide Posterior', 'DeltoideLateral': 'Deltoide Lateral', 'ManguitoRotador':'Manguito Rotador', 'PechoClavicular' :'Pecho Clavicular','PechoEnGeneral':'Pecho En General', 'TeresMajor' : 'Teres Major'}
updated_list = [replace_dict.get(item, item) for item in sorted_all_muscles]

# Ensure the 'Repeticiones Recomendadas' column is of type string
df['Repeticiones Recomendadas'] = df['Repeticiones Recomendadas'].astype(str)

# Split the 'repeticiones' column into 'min reps' and 'max reps'
split_reps = df['Repeticiones Recomendadas'].str.split('|', expand=True)

# Convert the 'min reps' column to numeric
df['min reps'] = pd.to_numeric(split_reps[0], errors='coerce')

# Keep 'max reps' as string to preserve leading zeros
df['max reps'] = split_reps[1]

# Convert 'max reps' to numeric, preserving leading zeros
df['max reps'] = df['max reps'].apply(lambda x: int(x) if x.isdigit() else x)

print(df[['min reps', 'max reps']])
app.layout = html.Div([
    html.H1("Gym Exercise Dashboard"),
    
    # Dropdown for selecting muscles
    dcc.Dropdown(
        id='muscle-dropdown',
        options=[{'label': muscle, 'value': muscle} for muscle in updated_list],
        value=list(all_muscles)[0],
        multi=False
    ),

    # Graph for displaying ratio fatiga estimulo colored by Extendido/Contraido
    dcc.Graph(id='ratio-extendido-chart'),

    # Graphs for displaying ratio fatiga estimulo, difficulty level, min reps, and max reps
    dcc.Graph(id='ratio-chart'),
    dcc.Graph(id='difficulty-chart'),
    dcc.Graph(id='reps-chart')
])

# Define callback to update charts based on dropdown selection
@app.callback(
    [Output('ratio-extendido-chart', 'figure'),
     Output('ratio-chart', 'figure'),
     Output('difficulty-chart', 'figure'),
     Output('reps-chart', 'figure')],
    [Input('muscle-dropdown', 'value')]
)
def update_content(selected_muscle):
    # Convert selected_muscle to string
    selected_muscle = str(selected_muscle)
    
    # Use lower() to make the comparison case-insensitive
    filtered_df = df[df['Musculos empleados'].str.lower().str.contains(selected_muscle.lower(), na=False)]
    
    # Create a bar chart for ratio fatiga estimulo colored by Extendido/Contraido
    ratio_extendido_chart = px.bar(
        filtered_df,
        x='Ejercicios',
        y='Ratio Fatiga-Estimulo',
        title=f'Ratio Fatiga-Estimulo Dividido por la Posicion del Musculo, Independiente del Musculo Elegido: {selected_muscle}',
        color='Extendido/Contraido',  # Color based on Extendido/Contraido
        labels={'Ratio Fatiga-Estimulo': 'Ratio Fatiga-Estimulo'},
    )

    # Create a bar chart for ratio fatiga estimulo
    ratio_chart = px.bar(
        filtered_df,
        x='Ejercicios',
        y='Ratio Fatiga-Estimulo',
        title=f'Ratio Fatiga-Estimulo Dividido por Exigencia al Cuerpo en General, Independiente del Musculo Elegido: {selected_muscle}',
        color='Aislacion/Compuesto',  # Color based on Aislacion/Compuesto
        labels={'Ratio Fatiga-Estimulo': 'Ratio Fatiga-Estimulo'},
    )

    # Create a sorted bar chart for difficulty level
    difficulty_chart = px.bar(
        filtered_df,
        x='Ejercicios',
        y='Dificultad',
        title=f'Nivel de Dificultad del Ejercicio, Independiente del musculo elegido y Coloreado por Lesividad',
        labels={'Dificultad': 'Difficulty Level'},
        color='Lesividad',  # Use 'Lesividad' column for color mapping
        color_continuous_scale='RdYlGn_r',  # Red-Yellow-Green color scale
    )

    # Update color bar title
    difficulty_chart.update_layout(coloraxis_colorbar=dict(title='Lesividad'))

    # Sort the bars by difficulty level
    difficulty_chart.update_layout(
        xaxis=dict(categoryorder='total ascending')
    )

    # Create a stacked bar chart for min and max reps
    reps_chart = px.bar(
        filtered_df,
        x='Ejercicios',
        y=['min reps', 'max reps'],
        title=f'Repeticiones Minimas Y Maximas',
        labels={'value': 'Reps'},
        barmode='stack',
    )

    return ratio_extendido_chart, ratio_chart, difficulty_chart, reps_chart

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

