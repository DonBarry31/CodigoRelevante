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

# Define the layout of the app

# Define the layout of the app
app.layout = html.Div([
    html.H1("Gym Exercise Dashboard"),
    
    # Dropdown for selecting muscles
    dcc.Dropdown(
        id='muscle-dropdown',
        options=[{'label': muscle, 'value': muscle} for muscle in updated_list],
        value=list(all_muscles)[0],
        multi=False
    ),
    
    # Text for displaying exercise information
    html.Div(id='exercise-list-output'),

    # Graphs for displaying difficulty level, min reps, and max reps
    dcc.Graph(id='difficulty-chart'),
    dcc.Graph(id='reps-chart'),
    dcc.Graph(id='ratio-chart')
])

# Define callback to update exercise list, difficulty chart, and reps chart based on dropdown selection
@app.callback(
    [Output('exercise-list-output', 'children'),
     Output('difficulty-chart', 'figure'),
     Output('reps-chart', 'figure'),
     Output('ratio-chart', 'figure')],
    [Input('muscle-dropdown', 'value')]
)
def update_content(selected_muscle):
    # Convert selected_muscle to string
    selected_muscle = str(selected_muscle)
    
    # Use lower() to make the comparison case-insensitive
    filtered_df = df[df['Musculos empleados'].str.lower().str.contains(selected_muscle.lower(), na=False)]
    
    # Get the list of exercises
    exercise_list = filtered_df['Ejercicios'].tolist()
    
    # Display the list as a comma-separated string
    exercise_str = ', '.join(exercise_list)

    # Create a sorted bar chart for difficulty level
    difficulty_chart = px.bar(
        filtered_df,
        x='Ejercicios',
        y='Dificultad',
        title=f'Difficulty Level for {selected_muscle}',
        labels={'Dificultad': 'Difficulty Level'},
    )

    # Create a stacked bar chart for min and max reps
    reps_chart = px.bar(
        filtered_df,
        x='Ejercicios',
        y=['min reps', 'max reps'],
        title=f'Min and Max Reps for {selected_muscle}',
        labels={'value': 'Reps'},
        barmode='stack',
    )

    ratio_chart = px.bar(
        filtered_df,
        x='Ejercicios',
        y='Ratio Fatiga-Estimulo',
        title=f'Ratio Fatiga-Estimulo for {selected_muscle}',
        color='Aislacion/Compuesto',  # Color based on Aislacion/Compuesto
        labels={'Ratio Fatiga-Estimulo': 'Ratio Fatiga-Estimulo'},
    )

    return f"Exercises involving {selected_muscle}: {exercise_str}", difficulty_chart, reps_chart, ratio_chart

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

