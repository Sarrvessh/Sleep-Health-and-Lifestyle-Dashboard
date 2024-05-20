import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import base64

# Load the data
data = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Splitting data into predictors (X) and target variable (y)
X = data.drop('Sleep Duration', axis=1)
y = data['Sleep Duration']

# Custom transformer to extract numeric values from 'Blood Pressure' column
class ExtractBloodPressure(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['Blood Pressure'] = X_copy['Blood Pressure'].apply(lambda x: re.findall(r'\d+', str(x)))
        X_copy['Systolic'] = X_copy['Blood Pressure'].apply(lambda x: int(x[0]) if len(x) > 0 else None)
        X_copy['Diastolic'] = X_copy['Blood Pressure'].apply(lambda x: int(x[1]) if len(x) > 1 else None)
        X_copy = X_copy.drop('Blood Pressure', axis=1)
        return X_copy

# Preprocessing pipeline
numeric_features = ['Age', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']
categorical_features = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('blood_pressure', ExtractBloodPressure(), ['Blood Pressure'])
    ])

# Apply preprocessing pipeline
X_preprocessed = preprocessor.fit_transform(X)

# Elbow method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_preprocessed)
    wcss.append(kmeans.inertia_)

# Identify the optimal number of clusters using the elbow method
optimal_k = 4  # Adjust this value based on the elbow method plot
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_preprocessed)
data['Cluster'] = clusters

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_preprocessed)

# Splitting the dataset into training and testing sets for regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a pipeline with preprocessing and linear regression model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save the Elbow plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
elbow_plot_path = 'elbow_plot.png'
plt.savefig(elbow_plot_path)
plt.close()

# Save the PCA cluster plot
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
for cluster in np.unique(clusters):
    plt.scatter(X_pca[clusters == cluster, 0], X_pca[clusters == cluster, 1], label=f'Cluster {cluster}', c=colors[cluster % len(colors)])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids', edgecolor='black')
plt.title('Clusters Visualization (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
pca_plot_path = 'pca_plot.png'
plt.savefig(pca_plot_path)
plt.close()

# Z-test for Sleep Duration based on Gender
male_sleep_duration = data[data['Gender'] == 'Male']['Sleep Duration']
female_sleep_duration = data[data['Gender'] == 'Female']['Sleep Duration']
z_stat, p_value_z = stats.ttest_ind(male_sleep_duration, female_sleep_duration)

# Chi-Square test for Sleep Disorder vs. Occupation
contingency_table = pd.crosstab(data['Sleep Disorder'], data['Occupation'])
chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)

# ANOVA for Sleep Duration across different Occupations
model = ols('y ~ C(X["Occupation"])', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Encode images to base64
def encode_image(image_path):
    with open(image_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')
    return encoded_image

elbow_plot_encoded = encode_image(elbow_plot_path)
pca_plot_encoded = encode_image(pca_plot_path)

# Summary statistics
summary_stats = data.describe().transpose()

# Plot for visualizing summary statistics
fig_hist_age = px.histogram(data, x='Age', nbins=20, title='Distribution of Age')
fig_hist_sleep_duration = px.histogram(data, x='Sleep Duration', nbins=20, title='Distribution of Sleep Duration')
fig_pie_gender = px.pie(data, names='Gender', title='Gender Distribution')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Sleep Health and Lifestyle Dashboard", className='title'),

    html.Div([
        html.Button("Show Regression Results", id='regression-results-btn', n_clicks=0, className='nav-button'),
        html.Button("Show Regression Graph", id='regression-graph-btn', n_clicks=0, className='nav-button'),
        html.Button("Show Cluster Results", id='cluster-results-btn', n_clicks=0, className='nav-button'),
        html.Button("Show Cluster Graphs", id='cluster-graphs-btn', n_clicks=0, className='nav-button'),
        html.Button("Show Statistical Tests", id='tests-btn', n_clicks=0, className='nav-button'),
    ], className='button-container'),

    dcc.Location(id='url', refresh=False),

    html.Div(id='page-content', className='content')
], className='main-container')

# Add external CSS
app.css.append_css({
    'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'
})

# Add custom CSS
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

# Define the custom CSS styles
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

            body {
                background: url('https://i.cbc.ca/1.6534070.1658953881!/fileImage/httpImage/image.jpg_gen/derivatives/original_1180/sleep-app-reviews.jpg') no-repeat center center fixed;
                background-size: cover;
                color: #343a40;
                font-family: 'Roboto', sans-serif;
                margin: 0;
                padding: 0;
                transition: background-color 0.5s ease;
            }
            .main-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: rgba(255, 255, 255, 0.8);
                border-radius: 15px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                animation: fadeIn 1s ease-in-out;
            }
            .title {
                text-align: center;
                margin-bottom: 30px;
                font-size: 32px;
                font-weight: 700;
                color: #007bff;
                animation: fadeInDown 1s ease-in-out;
            }
            .button-container {
                text-align: center;
                margin-bottom: 30px;
            }
            .nav-button {
                background-color: #007bff;
                border: none;
                color: white;
                padding: 14px 28px;
                font-size: 16px;
                margin: 10px;
                transition: all 0.3s ease;
                cursor: pointer;
                border-radius: 5px;
                box-shadow: 0 4px 8px rgba(0, 123, 255, 0.2);
            }
            .nav-button:hover {
                background-color: #0056b3;
                color: white;
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0, 123, 255, 0.3);
            }
            .nav-button:focus {
                outline: none;
            }
            .content {
                background: rgba(255, 255, 255, 0.9);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                animation: fadeIn 1s ease-in-out;
            }
            .content h2 {
                color: #343a40;
                font-weight: 500;
                margin-bottom: 20px;
            }
            .content p, .content ul, .content pre {
                color: #495057;
                font-weight: 400;
                font-size: 16px;
                line-height: 1.6;
            }
            .content ul {
                list-style-type: none;
                padding: 0;
            }
            .content ul li {
                background: #f8f9fa;
                margin: 5px 0;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .content img {
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                display: block;
                margin: 20px auto;
            }
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            @keyframes fadeInDown {
                from { opacity: 0; transform: translateY(-20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


# Define the callbacks for navigation
@app.callback(
    Output('url', 'pathname'),
    [Input('regression-results-btn', 'n_clicks'),
     Input('regression-graph-btn', 'n_clicks'),
     Input('cluster-results-btn', 'n_clicks'),
     Input('cluster-graphs-btn', 'n_clicks'),
     Input('tests-btn', 'n_clicks')]
)
def navigate_page(regression_results_clicks, regression_graph_clicks, cluster_results_clicks, cluster_graphs_clicks, tests_clicks):
    ctx = dash.callback_context

    if not ctx.triggered:
        return '/'
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'regression-results-btn':
        return '/regression-results'
    elif button_id == 'regression-graph-btn':
        return '/regression-graph'
    elif button_id == 'cluster-results-btn':
        return '/cluster-results'
    elif button_id == 'cluster-graphs-btn':
        return '/cluster-graphs'
    elif button_id == 'tests-btn':
        return '/tests'
    return '/'

# Define the content for each page
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/regression-results':
        return html.Div([
            html.H2("Regression Results"),
            html.P(f"Mean Squared Error: {mse}"),
            html.P(f"R-squared Score: {r2}")
        ])
    elif pathname == '/regression-graph':
        return html.Div([
            html.H2("Regression Graph"),
            dcc.Graph(
                figure=px.scatter(
                    x=y_test, y=y_pred, labels={'x': 'Actual Sleep Duration', 'y': 'Predicted Sleep Duration'},
                    title='Actual vs. Predicted Sleep Durations (Linear Regression)'
                ).add_scatter(x=y_test, y=y_test, mode='lines', name='Actual')
            )
        ])
    elif pathname == '/cluster-results':
        return html.Div([
            html.H2("Cluster Results"),
            html.P(f"Optimal Number of Clusters: {optimal_k}"),
            html.P("Number of Data Points in Each Cluster:"),
            html.Ul([html.Li(f"Cluster {i}: {count}") for i, count in enumerate(data['Cluster'].value_counts().sort_index())])
        ])
    elif pathname == '/cluster-graphs':
        return html.Div([
            html.H2("Cluster Graphs"),
            html.Img(src='data:image/png;base64,{}'.format(elbow_plot_encoded)),
            html.Img(src='data:image/png;base64,{}'.format(pca_plot_encoded))
        ])
    elif pathname == '/tests':
        return html.Div([
            html.H2("Statistical Tests"),
            html.H3("Z-test for Sleep Duration based on Gender"),
            html.P(f"Z-statistic: {z_stat}"),
            html.P(f"P-value: {p_value_z}"),
            html.H3("Chi-Square test for Sleep Disorder vs. Occupation"),
            html.P(f"Chi2: {chi2}"),
            html.P(f"P-value: {p_value_chi2}"),
            html.P(f"Degrees of Freedom: {dof}"),
            html.H3("ANOVA for Sleep Duration across different Occupations"),
            html.Pre(f"{anova_table}")
        ])
    else:
        return html.Div([
            html.H2("Overview of the Dataset"),
            html.P("This dashboard presents an analysis of sleep health and lifestyle data."),
            html.H3("Summary Statistics"),
            html.Table([
                html.Thead(
                    html.Tr([html.Th(col) for col in summary_stats.columns])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(summary_stats.index[i]),
                        html.Td(summary_stats.iloc[i]['count']),
                        html.Td(summary_stats.iloc[i]['mean']),
                        html.Td(summary_stats.iloc[i]['std']),
                        html.Td(summary_stats.iloc[i]['min']),
                        html.Td(summary_stats.iloc[i]['25%']),
                        html.Td(summary_stats.iloc[i]['50%']),
                        html.Td(summary_stats.iloc[i]['75%']),
                        html.Td(summary_stats.iloc[i]['max'])
                    ]) for i in range(len(summary_stats))
                ])
            ]),
            html.H3("Visualizations"),
            dcc.Graph(figure=fig_hist_age),
            dcc.Graph(figure=fig_hist_sleep_duration),
            dcc.Graph(figure=fig_pie_gender)
        ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
