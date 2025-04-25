import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import names

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Create function to generate random student data
def generate_student_data(num_students=700):
    # Generate random student names
    student_names = [names.get_full_name() for _ in range(num_students)]

    # Generate random student IDs
    student_ids = [f"S{i+1:03d}" for i in range(num_students)]

    # Generate random departments
    departments = ['Computer Science', 'Electrical Engineering', 'Mechanical Engineering',
                  'Civil Engineering', 'Chemical Engineering']
    dept_values = [random.choice(departments) for _ in range(num_students)]

    # Generate random year of study (1-4)
    years = [random.randint(1, 4) for _ in range(num_students)]

    # Generate attendance percentage (50-100%)
    attendance = [round(random.uniform(50, 100), 2) for _ in range(num_students)]

    # Generate total days in college (based on year and attendance)
    max_days_per_year = 220
    total_days = [int(max_days_per_year * years[i] * attendance[i]/100) for i in range(num_students)]

    # Generate CGPA (2.0-10.0)
    cgpa = [round(random.uniform(2.0, 10.0), 2) for _ in range(num_students)]

    # Generate SGPA (slightly variant from CGPA)
    sgpa = [round(min(10.0, max(2.0, cgpa[i] + random.uniform(-1.0, 1.0))), 2) for i in range(num_students)]

    # Generate backlogs (more likely if CGPA is low)
    backlogs = [max(0, int(np.random.poisson(max(0, (6 - cgpa[i]) * 1.5)))) for i in range(num_students)]

    # Generate passed courses (3-8 per semester, based on year and backlogs)
    passed_courses = [random.randint(3, 8) * years[i] - backlogs[i] for i in range(num_students)]
    passed_courses = [max(0, p) for p in passed_courses]

    # Generate current status
    statuses = ['Active', 'Probation', 'Graduated', 'On Leave', 'Dropped']
    # Logic: students with low CGPA or high backlogs more likely to be on probation or dropped
    status_values = []
    for i in range(num_students):
        if years[i] == 4 and cgpa[i] >= 6.0 and backlogs[i] <= 2:
            status = 'Graduated' if random.random() < 0.8 else 'Active'
        elif cgpa[i] < 5.0 or backlogs[i] > 5:
            if random.random() < 0.7:
                status = 'Probation' if random.random() < 0.6 else 'Dropped'
            else:
                status = 'Active'
        elif random.random() < 0.05:
            status = 'On Leave'
        else:
            status = 'Active'
        status_values.append(status)

    # Generate placement offers (more likely if CGPA is high and backlogs are low)
    placement_offers = []
    for i in range(num_students):
        if years[i] < 3:  # First and second-year students don't get offers
            placement_offers.append(0)
        elif status_values[i] == 'Dropped':
            placement_offers.append(0)
        else:
            # Higher CGPA and fewer backlogs mean more placement offers
            placement_probability = min(1.0, max(0.0, (cgpa[i] - 6) / 4 - backlogs[i] * 0.05))
            num_offers = np.random.poisson(placement_probability * 3)
            placement_offers.append(max(0, num_offers))

    # Create DataFrame
    df = pd.DataFrame({
        'StudentID': student_ids,
        'Name': student_names,
        'Department': dept_values,
        'Year': years,
        'Attendance': attendance,
        'DaysInCollege': total_days,
        'CGPA': cgpa,
        'SGPA': sgpa,
        'Backlogs': backlogs,
        'PassedCourses': passed_courses,
        'Status': status_values,
        'PlacementOffers': placement_offers
    })

    return df

# Generate random student data
student_df = generate_student_data(700)

# Display basic information about the dataset
print("Student Dataset Shape:", student_df.shape)
print("\nSample Data:")
print(student_df.head())
print("\nData Summary:")
print(student_df.describe())

# Data preparation for visualization and analysis
print("\nDepartment Distribution:")
print(student_df['Department'].value_counts())

print("\nStatus Distribution:")
print(student_df['Status'].value_counts())

# Check for correlations between numerical variables
numerical_cols = ['Attendance', 'DaysInCollege', 'CGPA', 'SGPA', 'Backlogs', 'PassedCourses', 'PlacementOffers']
correlation_matrix = student_df[numerical_cols].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualizations

# 1. CGPA Distribution
plt.figure(figsize=(10, 6))
sns.histplot(student_df['CGPA'], kde=True, bins=20)
plt.title('Distribution of CGPA Among Students')
plt.xlabel('CGPA')
plt.ylabel('Count')
plt.axvline(student_df['CGPA'].mean(), color='red', linestyle='--', label=f'Mean: {student_df["CGPA"].mean():.2f}')
plt.legend()
plt.savefig('cgpa_distribution.png')

# 2. Attendance vs CGPA Scatter Plot with Department
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Attendance', y='CGPA', hue='Department', data=student_df, alpha=0.7)
plt.title('Relationship Between Attendance and CGPA')
plt.xlabel('Attendance (%)')
plt.ylabel('CGPA')
plt.grid(True, alpha=0.3)
plt.savefig('attendance_vs_cgpa.png')

# 3. Backlogs Distribution by Year
plt.figure(figsize=(10, 6))
sns.boxplot(x='Year', y='Backlogs', data=student_df)
plt.title('Backlogs Distribution by Year of Study')
plt.xlabel('Year of Study')
plt.ylabel('Number of Backlogs')
plt.savefig('backlogs_by_year.png')

# 4. Placement Offers by Department
plt.figure(figsize=(12, 6))
dept_placement = student_df.groupby('Department')['PlacementOffers'].agg(['mean', 'sum', 'count'])
dept_placement['avg_per_student'] = dept_placement['sum'] / dept_placement['count']
dept_placement = dept_placement.sort_values('avg_per_student', ascending=False)

sns.barplot(x=dept_placement.index, y=dept_placement['avg_per_student'])
plt.title('Average Placement Offers by Department')
plt.xlabel('Department')
plt.ylabel('Average Number of Placement Offers')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('placement_by_department.png')

# 5. Status Distribution Pie Chart
plt.figure(figsize=(10, 8))
status_counts = student_df['Status'].value_counts()
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90, shadow=True)
plt.title('Student Status Distribution')
plt.axis('equal')
plt.savefig('status_distribution.png')

# 6. Heatmap of Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Student Metrics')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')

# 7. CGPA and SGPA comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.regplot(x='CGPA', y='SGPA', data=student_df, scatter_kws={'alpha':0.5})
plt.title('CGPA vs SGPA')
plt.xlabel('CGPA')
plt.ylabel('SGPA')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.histplot(student_df['SGPA'] - student_df['CGPA'], kde=True, bins=20)
plt.title('Difference between SGPA and CGPA')
plt.xlabel('SGPA - CGPA')
plt.axvline(0, color='red', linestyle='--')
plt.tight_layout()
plt.savefig('cgpa_sgpa_comparison.png')

# 8. Days in College vs Attendance
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Attendance', y='DaysInCollege', hue='Year', size='CGPA',
                sizes=(20, 200), alpha=0.7, data=student_df)
plt.title('Relationship Between Attendance, Days in College, and Year')
plt.xlabel('Attendance (%)')
plt.ylabel('Days in College')
plt.grid(True, alpha=0.3)
plt.savefig('attendance_days_year.png')

# Deep Learning Component - Prediction Model for Placement Offers

# Prepare features for deep learning
features = ['Year', 'Attendance', 'CGPA', 'SGPA', 'Backlogs', 'PassedCourses']
X = student_df[features].values
y = student_df['PlacementOffers'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build a simple neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPlacement Prediction Model - Mean Absolute Error: {test_mae:.2f}")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.tight_layout()
plt.savefig('dl_training_history.png')

# Make predictions on the test set
y_pred = model.predict(X_test).flatten()

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
plt.title('Actual vs Predicted Placement Offers')
plt.xlabel('Actual Placement Offers')
plt.ylabel('Predicted Placement Offers')
plt.grid(True, alpha=0.3)
plt.savefig('placement_prediction.png')

# Identify key factors for placement success using feature importance
# For simplicity, we'll use a Random Forest model for feature importance
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance for Placement Offers:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance for Placement Offers')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Cluster Analysis using K-means
# Find optimal number of clusters using Elbow Method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'o-')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True, alpha=0.3)
plt.savefig('kmeans_elbow.png')

# Choose k=4 based on elbow method (hypothetical)
kmeans = KMeans(n_clusters=4, random_state=42)
student_df['Cluster'] = kmeans.fit_predict(X_scaled)

# Reduce dimensions for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=student_df['Cluster'], palette='viridis', s=50, alpha=0.7)
plt.title('Student Clusters based on Academic Performance')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, alpha=0.3)
plt.savefig('student_clusters.png')

# Analyze clusters
cluster_profiles = student_df.groupby('Cluster')[features + ['PlacementOffers']].mean()
print("\nCluster Profiles:")
print(cluster_profiles)

# Export the data to CSV
student_df.to_csv('student_data.csv', index=False)

print("\nData analysis and visualization completed. All results saved as PNG files.")
print("Student data exported to 'student_data.csv'")

# ADDITIONAL CODE STARTS HERE

# 9. Advanced Analysis: Student Performance by Department and Year
print("\n\n--- ADVANCED STUDENT ANALYTICS ---\n")

# Create a pivot table for department and year performance analysis
dept_year_performance = pd.pivot_table(
    student_df,
    values=['CGPA', 'Backlogs', 'PlacementOffers'],
    index=['Department'],
    columns=['Year'],
    aggfunc={'CGPA': 'mean', 'Backlogs': 'mean', 'PlacementOffers': 'mean'}
)

print("\nDepartment and Year Performance Analysis:")
print(dept_year_performance)

# Plot department performance across years
plt.figure(figsize=(14, 8))
for i, metric in enumerate(['CGPA', 'Backlogs', 'PlacementOffers']):
    plt.subplot(1, 3, i+1)
    dept_year_data = dept_year_performance[metric].transpose()
    dept_year_data.plot(kind='bar', ax=plt.gca())
    plt.title(f'Average {metric} by Department and Year')
    plt.xlabel('Year of Study')
    plt.ylabel(f'Average {metric}')
    plt.legend(title='Department')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('dept_year_performance.png')

# 10. Student Risk Analysis
# Define risk factors for students
def calculate_risk_score(row):
    risk_score = 0

    # Low CGPA is a risk factor
    if row['CGPA'] < 5.0:
        risk_score += 3
    elif row['CGPA'] < 6.0:
        risk_score += 2
    elif row['CGPA'] < 7.0:
        risk_score += 1

    # High number of backlogs is a risk factor
    if row['Backlogs'] > 5:
        risk_score += 3
    elif row['Backlogs'] > 2:
        risk_score += 2
    elif row['Backlogs'] > 0:
        risk_score += 1

    # Low attendance is a risk factor
    if row['Attendance'] < 65:
        risk_score += 3
    elif row['Attendance'] < 75:
        risk_score += 2
    elif row['Attendance'] < 85:
        risk_score += 1

    # Status is a risk factor
    if row['Status'] == 'Probation':
        risk_score += 2
    elif row['Status'] == 'On Leave':
        risk_score += 1

    return risk_score

# Calculate risk score for each student
student_df['RiskScore'] = student_df.apply(calculate_risk_score, axis=1)

# Define risk categories
student_df['RiskCategory'] = pd.cut(
    student_df['RiskScore'],
    bins=[-1, 2, 5, 9],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

# Display risk distribution
print("\nStudent Risk Distribution:")
print(student_df['RiskCategory'].value_counts())

# Plot risk distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='RiskCategory', hue='Department', data=student_df)
plt.title('Student Risk Distribution by Department')
plt.xlabel('Risk Category')
plt.ylabel('Number of Students')
plt.xticks(rotation=0)
plt.legend(title='Department', loc='upper right')
plt.tight_layout()
plt.savefig('risk_distribution.png')

# 11. Time Series Analysis: Simulated Semester Performance
# Generate synthetic data for semester-wise performance
semesters = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4', 'Sem 5', 'Sem 6', 'Sem 7', 'Sem 8']
semester_data = []

# Function to generate realistic semester GPAs based on overall CGPA
def generate_semester_gpas(cgpa, num_semesters):
    # Add some random fluctuation around the CGPA
    base = max(2.0, min(10.0, cgpa))
    semester_gpas = []
    for i in range(num_semesters):
        # More fluctuation in earlier semesters, more stability later
        if i < 2:
            fluctuation = random.uniform(-1.5, 1.5)
        elif i < 4:
            fluctuation = random.uniform(-1.0, 1.0)
        else:
            fluctuation = random.uniform(-0.5, 0.5)

        # Ensure GPA stays within bounds and has a slight upward trend
        trend = min(0.2, 0.05 * i)
        semester_gpa = max(2.0, min(10.0, base + fluctuation + trend))
        semester_gpas.append(round(semester_gpa, 2))

    return semester_gpas

# Generate semester data for a sample of students (50 students)
student_sample = student_df.sample(50, random_state=42)
for idx, student in student_sample.iterrows():
    # Calculate number of semesters based on year
    num_semesters = min(8, student['Year'] * 2)

    # Generate semester GPAs
    semester_gpas = generate_semester_gpas(student['CGPA'], num_semesters)

    # Add data for each semester
    for i in range(num_semesters):
        semester_data.append({
            'StudentID': student['StudentID'],
            'Name': student['Name'],
            'Department': student['Department'],
            'Semester': semesters[i],
            'SemesterGPA': semester_gpas[i]
        })

# Create semester dataframe
semester_df = pd.DataFrame(semester_data)

# Plot semester-wise performance for a few selected students
plt.figure(figsize=(12, 8))
for student_id in semester_df['StudentID'].unique()[:5]:
    student_semester_data = semester_df[semester_df['StudentID'] == student_id]
    plt.plot(student_semester_data['Semester'], student_semester_data['SemesterGPA'], marker='o', label=student_id)

plt.title('Semester-wise GPA Progression for Selected Students')
plt.xlabel('Semester')
plt.ylabel('GPA')
plt.ylim(0, 10.5)
plt.grid(True, alpha=0.3)
plt.legend(title='Student ID')
plt.tight_layout()
plt.savefig('semester_progression.png')

# Calculate average semester performance by department
dept_semester_performance = semester_df.groupby(['Department', 'Semester'])['SemesterGPA'].mean().unstack()
print("\nDepartment Semester Performance:")
print(dept_semester_performance)

# Plot department semester performance
plt.figure(figsize=(12, 6))
dept_semester_performance.plot(marker='o', ax=plt.gca())
plt.title('Average GPA by Department and Semester')
plt.xlabel('Department')
plt.ylabel('Average GPA')
plt.grid(True, alpha=0.3)
plt.legend(title='Semester')
plt.tight_layout()
plt.savefig('dept_semester_performance.png')

# 12. Student Success Prediction: Classification model
# Define "success" as having at least one placement offer
student_df['Success'] = (student_df['PlacementOffers'] > 0).astype(int)

# Only consider 3rd and 4th year students for success prediction
senior_students = student_df[student_df['Year'] >= 3].copy()

# Prepare features for classification
X_class = senior_students[features].values
y_class = senior_students['Success'].values

# Standardize features
X_class_scaled = scaler.fit_transform(X_class)

# Split data
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
    X_class_scaled, y_class, test_size=0.2, random_state=42
)

# Import classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Train multiple classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Compare classifiers
print("\nStudent Success Classification Models:")
for name, clf in classifiers.items():
    clf.fit(X_class_train, y_class_train)
    y_class_pred = clf.predict(X_class_test)
    accuracy = accuracy_score(y_class_test, y_class_pred)
    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_class_test, y_class_pred))

# Use the best classifier (assumed to be Random Forest based on typical performance)
best_clf = classifiers['Random Forest']
y_class_pred = best_clf.predict(X_class_test)

# Plot confusion matrix for the best classifier
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_class_test, y_class_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Student Success Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0.5, 1.5], ['Not Successful', 'Successful'])
plt.yticks([0.5, 1.5], ['Not Successful', 'Successful'])
plt.tight_layout()
plt.savefig('success_confusion_matrix.png')

# 13. Department Performance Dashboard
# Create a comprehensive performance metric for departments
dept_metrics = student_df.groupby('Department').agg({
    'CGPA': 'mean',
    'Backlogs': 'mean',
    'PlacementOffers': ['mean', 'sum'],
    'Attendance': 'mean',
    'StudentID': 'count'
}).reset_index()

# Flatten multi-level columns
dept_metrics.columns = ['_'.join(col).strip('_') for col in dept_metrics.columns.values]
dept_metrics.rename(columns={'StudentID_count': 'NumStudents'}, inplace=True)

# Calculate success rate (students with at least 1 placement offer)
dept_success = senior_students.groupby('Department')['Success'].mean().reset_index()
dept_success.rename(columns={'Success': 'SuccessRate'}, inplace=True)

# Merge metrics
dept_metrics = pd.merge(dept_metrics, dept_success, on='Department', how='left')
dept_metrics['SuccessRate'] = dept_metrics['SuccessRate'].fillna(0)
dept_metrics['SuccessRate'] = dept_metrics['SuccessRate'] * 100  # Convert to percentage

print("\nDepartment Performance Dashboard:")
print(dept_metrics)

# Create radar chart for department comparison
from math import pi

# Prepare data for radar chart
metrics = ['CGPA_mean', 'Attendance_mean', 'SuccessRate',
           'PlacementOffers_mean']
metrics_names = ['Avg CGPA', 'Avg Attendance', 'Success Rate (%)',
                'Avg Placement Offers']

# Normalize metrics to 0-1 scale for radar chart
radar_data = dept_metrics.copy()
for metric in metrics:
    if metric == 'Backlogs_mean':  # Inverse relationship (lower is better)
        max_val = radar_data[metric].max()
        radar_data[metric] = 1 - (radar_data[metric] / max_val)
    else:
        min_val = radar_data[metric].min()
        max_val = radar_data[metric].max()
        radar_data[metric] = (radar_data[metric] - min_val) / (max_val - min_val)

# Function to create radar chart
def radar_chart(df, categories, title):
    # Number of variables
    N = len(categories)

    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], metrics_names, size=12)

    # Draw y-axis labels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=10)
    plt.ylim(0, 1)

    # Plot each department
    for i, dept in enumerate(df['Department']):
        values = df.loc[i, metrics].values.flatten().tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=dept)
        ax.fill(angles, values, alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, size=15, y=1.1)

    return fig, ax

# Create radar chart
fig, ax = radar_chart(radar_data, metrics, 'Department Performance Comparison')
plt.tight_layout()
plt.savefig('department_radar_chart.png')

# 14. Student Network Analysis using NetworkX
import networkx as nx

# Create a synthetic student network based on departments and years
G = nx.Graph()

# Add nodes (students)
for idx, student in student_df.iterrows():
    G.add_node(student['StudentID'],
               name=student['Name'],
               department=student['Department'],
               year=student['Year'],
               cgpa=student['CGPA'])

# Add edges based on department and year (students in same department and year are more likely to connect)
for i, student1 in student_df.iterrows():
    for j, student2 in student_df.iterrows():
        if i < j:  # Avoid duplicate edges
            id1, id2 = student1['StudentID'], student2['StudentID']

            # Base probability
            p = 0.01

            # Same department increases connection probability
            if student1['Department'] == student2['Department']:
                p += 0.1

            # Same year increases connection probability
            if student1['Year'] == student2['Year']:
                p += 0.1

            # Similar CGPA increases connection probability (academic similarity)
            if abs(student1['CGPA'] - student2['CGPA']) < 1.0:
                p += 0.05

            # Randomize connections
            if random.random() < p:
                G.add_edge(id1, id2)
                # Add weights to edges (strength of connection)
for u, v in G.edges():
    # Get student data
    student1 = student_df[student_df['StudentID'] == u].iloc[0]
    student2 = student_df[student_df['StudentID'] == v].iloc[0]

    # Calculate connection strength based on similarities
    weight = 1.0

    # Department match increases weight
    if student1['Department'] == student2['Department']:
        weight += 0.5

    # Same year increases weight
    if student1['Year'] == student2['Year']:
        weight += 0.3

    # CGPA similarity increases weight
    cgpa_diff = abs(student1['CGPA'] - student2['CGPA'])
    weight += max(0, 0.5 - cgpa_diff/10)

    # Assign edge weight
    G[u][v]['weight'] = weight

# Calculate network metrics
print("\nStudent Network Analysis:")
print(f"Number of nodes (students): {G.number_of_nodes()}")
print(f"Number of edges (connections): {G.number_of_edges()}")

# Calculate degree centrality (number of connections)
degree_centrality = nx.degree_centrality(G)
nx.set_node_attributes(G, degree_centrality, 'degree_centrality')

# Calculate betweenness centrality (bridge nodes)
betweenness_centrality = nx.betweenness_centrality(G)
nx.set_node_attributes(G, betweenness_centrality, 'betweenness_centrality')

# Identify communities using Louvain method
try:
    from community import community_louvain
    partition = community_louvain.best_partition(G)
    nx.set_node_attributes(G, partition, 'community')
    print(f"Number of communities detected: {len(set(partition.values()))}")
except ImportError:
    print("Community detection package not installed, skipping community detection")
    partition = {node: 0 for node in G.nodes()}  # Default community

# Visualize the network (limited to a sample for clarity)
sample_size = min(100, len(student_df))
sample_students = random.sample(list(G.nodes()), sample_size)
G_sample = G.subgraph(sample_students)

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G_sample, seed=42)

# Color nodes by department
dept_colors = {
    'Computer Science': 'red',
    'Electrical Engineering': 'blue',
    'Mechanical Engineering': 'green',
    'Civil Engineering': 'orange',
    'Chemical Engineering': 'purple'
}

# Node colors based on department
node_colors = [dept_colors[student_df[student_df['StudentID'] == node]['Department'].values[0]]
               for node in G_sample.nodes()]

# Node sizes based on CGPA
node_sizes = [50 + 20 * student_df[student_df['StudentID'] == node]['CGPA'].values[0]
              for node in G_sample.nodes()]

# Edge weights
edge_weights = [G_sample[u][v]['weight'] * 0.5 for u, v in G_sample.edges()]

# Draw the network
nx.draw_networkx(
    G_sample,
    pos=pos,
    node_color=node_colors,
    node_size=node_sizes,
    width=edge_weights,
    with_labels=False,
    alpha=0.7
)

plt.title('Student Relationship Network (Sample)')
plt.axis('off')

# Create legend for departments
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
           for color in dept_colors.values()]
labels = list(dept_colors.keys())
plt.legend(handles, labels, title='Department', loc='upper right')

plt.savefig('student_network.png')

# 15. Create a comprehensive student dashboard
# Calculate percentiles for key metrics
student_df['CGPA_Percentile'] = student_df['CGPA'].rank(pct=True) * 100
student_df['Attendance_Percentile'] = student_df['Attendance'].rank(pct=True) * 100
student_df['Backlogs_Percentile'] = (1 - student_df['Backlogs'].rank(pct=True)) * 100  # Inverse (lower is better)

# Create a student performance index (weighted average of percentiles)
student_df['PerformanceIndex'] = (
    0.5 * student_df['CGPA_Percentile'] +
    0.3 * student_df['Attendance_Percentile'] +
    0.2 * student_df['Backlogs_Percentile']
)

# Identify top performers
top_performers = student_df.nlargest(10, 'PerformanceIndex')
print("\nTop 10 Performers:")
print(top_performers[['StudentID', 'Name', 'Department', 'Year', 'CGPA', 'PerformanceIndex']])

# Identify students needing attention (high risk and low performance)
at_risk = student_df[
    (student_df['RiskCategory'] == 'High Risk') &
    (student_df['PerformanceIndex'] < 30)
].sort_values('PerformanceIndex')

print("\nStudents Needing Attention:")
print(at_risk[['StudentID', 'Name', 'Department', 'Year', 'CGPA', 'Backlogs', 'RiskScore', 'PerformanceIndex']])

# 16. Create a prediction model for dropout risk
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
le_dept = LabelEncoder()
le_status = LabelEncoder()

# Create feature set for dropout prediction
dropout_features = student_df.copy()
dropout_features['Department_Encoded'] = le_dept.fit_transform(dropout_features['Department'])
dropout_features['Status_Encoded'] = le_status.fit_transform(dropout_features['Status'])

# Define dropout (proxy using "Dropped" status or high risk with poor performance)
dropout_features['Dropout'] = ((dropout_features['Status'] == 'Dropped') |
                              ((dropout_features['RiskCategory'] == 'High Risk') &
                               (dropout_features['PerformanceIndex'] < 20))).astype(int)

# Define features for dropout prediction
X_dropout = dropout_features[[
    'Year', 'Attendance', 'CGPA', 'SGPA', 'Backlogs',
    'PassedCourses', 'Department_Encoded', 'RiskScore'
]].values

y_dropout = dropout_features['Dropout'].values

# Split data
X_dropout_train, X_dropout_test, y_dropout_train, y_dropout_test = train_test_split(
    X_dropout, y_dropout, test_size=0.2, random_state=42
)

# Train a Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_dropout_train, y_dropout_train)

# Evaluate model
y_dropout_pred = gb_model.predict(X_dropout_test)
print("\nDropout Prediction Model:")
print(f"Accuracy: {accuracy_score(y_dropout_test, y_dropout_pred):.4f}")
print(classification_report(y_dropout_test, y_dropout_pred))

# Get feature importance
dropout_importance = pd.DataFrame({
    'Feature': ['Year', 'Attendance', 'CGPA', 'SGPA', 'Backlogs',
               'PassedCourses', 'Department', 'RiskScore'],
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance for Dropout Prediction:")
print(dropout_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=dropout_importance)
plt.title('Feature Importance for Dropout Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('dropout_feature_importance.png')

# 17. Student Performance Trend Analysis
# Calculate average metrics by year for each department
year_trends = student_df.groupby(['Department', 'Year']).agg({
    'CGPA': 'mean',
    'Attendance': 'mean',
    'Backlogs': 'mean',
    'PlacementOffers': 'mean',
    'RiskScore': 'mean'
}).reset_index()

# Plot trends by department
plt.figure(figsize=(15, 12))
metrics = ['CGPA', 'Attendance', 'Backlogs', 'PlacementOffers', 'RiskScore']
departments = student_df['Department'].unique()

for i, metric in enumerate(metrics):
    plt.subplot(3, 2, i+1)
    for dept in departments:
        dept_data = year_trends[year_trends['Department'] == dept]
        plt.plot(dept_data['Year'], dept_data[metric], marker='o', label=dept)

    plt.title(f'{metric} Trend by Year')
    plt.xlabel('Year of Study')
    plt.ylabel(f'Average {metric}')
    plt.grid(True, alpha=0.3)
    plt.xticks([1, 2, 3, 4])

    if i == 0:  # Only show legend on first subplot
        plt.legend(title='Department')

plt.tight_layout()
plt.savefig('performance_trends.png')

# 18. Export comprehensive analysis to Excel
with pd.ExcelWriter('student_analytics_report.xlsx') as writer:
    # Basic student data
    student_df.to_excel(writer, sheet_name='Student Data', index=False)

    # Department metrics
    dept_metrics.to_excel(writer, sheet_name='Department Metrics', index=False)

    # Risk analysis
    risk_analysis = student_df.groupby(['Department', 'RiskCategory']).size().unstack().fillna(0)
    risk_analysis.to_excel(writer, sheet_name='Risk Analysis')

    # Performance by year
    year_trends.to_excel(writer, sheet_name='Year Trends', index=False)

    # Top performers
    top_performers.to_excel(writer, sheet_name='Top Performers', index=False)

    # At-risk students
    at_risk.to_excel(writer, sheet_name='At Risk Students', index=False)

print("\nComprehensive analysis exported to 'student_analytics_report.xlsx'")

# 19. Interactive Dashboard Components (simulated)
# In a real implementation, this would connect to a web framework like Dash or Streamlit
# Here we'll demonstrate the code structure

def create_student_profile_dashboard(student_id):
    """Generate profile data for a specific student"""
    student = student_df[student_df['StudentID'] == student_id].iloc[0]

    print(f"\nStudent Profile: {student['Name']} ({student_id})")
    print(f"Department: {student['Department']}")
    print(f"Year: {student['Year']}")
    print(f"CGPA: {student['CGPA']}")
    print(f"Attendance: {student['Attendance']}%")
    print(f"Backlogs: {student['Backlogs']}")
    print(f"Risk Category: {student['RiskCategory']}")
    print(f"Performance Index: {student['PerformanceIndex']:.2f}")

    # Get student percentile in department
    dept_students = student_df[student_df['Department'] == student['Department']]
    cgpa_percentile = (dept_students['CGPA'] < student['CGPA']).mean() * 100

    print(f"CGPA Percentile in Department: {cgpa_percentile:.2f}%")

    # Performance recommendations
    if student['RiskCategory'] == 'High Risk':
        print("\nRecommendations:")
        print("- Schedule academic counseling session")
        print("- Enroll in remedial classes for subjects with backlogs")
        print("- Implement strict attendance monitoring")
    elif student['RiskCategory'] == 'Medium Risk':
        print("\nRecommendations:")
        print("- Regular check-ins with academic advisor")
        print("- Focus on improving attendance and reducing backlogs")
        print("- Consider peer tutoring for challenging subjects")
    else:
        print("\nRecommendations:")
        print("- Continue with current academic progress")
        print("- Explore advanced courses or research opportunities")
        print("- Consider mentoring other students")

# Demonstrate with a sample student
sample_student_id = student_df.sample(1)['StudentID'].values[0]
create_student_profile_dashboard(sample_student_id)

# 20. Department Insights and Recommendations
def department_insights(department_name):
    """Generate insights and recommendations for a specific department"""
    dept_data = student_df[student_df['Department'] == department_name]

    print(f"\nDepartment Insights: {department_name}")
    print(f"Number of Students: {len(dept_data)}")
    print(f"Average CGPA: {dept_data['CGPA'].mean():.2f}")
    print(f"Average Attendance: {dept_data['Attendance'].mean():.2f}%")
    print(f"Average Backlogs: {dept_data['Backlogs'].mean():.2f}")
    print(f"Risk Distribution: {dept_data['RiskCategory'].value_counts(normalize=True).multiply(100).round(1)}")

    # Placement success
    dept_seniors = dept_data[dept_data['Year'] >= 3]
    if len(dept_seniors) > 0:
        placement_rate = (dept_seniors['PlacementOffers'] > 0).mean() * 100
        print(f"Placement Success Rate: {placement_rate:.2f}%")

    # Generate insights
    print("\nKey Insights:")

    # CGPA comparison
    overall_cgpa = student_df['CGPA'].mean()
    dept_cgpa = dept_data['CGPA'].mean()

    if dept_cgpa > overall_cgpa * 1.05:
        print(f"- {department_name} CGPA is significantly above college average")
    elif dept_cgpa < overall_cgpa * 0.95:
        print(f"- {department_name} CGPA is below college average")

    # Risk analysis
    high_risk_pct = (dept_data['RiskCategory'] == 'High Risk').mean() * 100
    if high_risk_pct > 15:
        print(f"- High proportion of at-risk students ({high_risk_pct:.1f}%)")

    # Placement analysis
    if len(dept_seniors) > 0:
        college_placement_rate = (student_df[student_df['Year'] >= 3]['PlacementOffers'] > 0).mean() * 100
        if placement_rate < college_placement_rate * 0.9:
            print("- Placement rate below college average")
        elif placement_rate > college_placement_rate * 1.1:
            print("- Placement rate above college average")

    # Recommendations
    print("\nRecommendations:")
    if high_risk_pct > 15:
        print("- Implement intervention program for high-risk students")
        print("- Schedule regular faculty-student mentoring sessions")

    if len(dept_seniors) > 0 and placement_rate < college_placement_rate:
        print("- Enhance industry collaboration and placement preparation")
        print("- Organize additional skill development workshops")

    if dept_data['Attendance'].mean() < 80:
        print("- Improve attendance monitoring and engagement activities")

# Demonstrate with a sample department
sample_dept = random.choice(student_df['Department'].unique())
department_insights(sample_dept)

# Final summary
print("\n===== Student Analytics System Summary =====")
print(f"Total Students Analyzed: {len(student_df)}")
print(f"Departments: {', '.join(student_df['Department'].unique())}")
print(f"Average College CGPA: {student_df['CGPA'].mean():.2f}")
print(f"Students at High Risk: {(student_df['RiskCategory'] == 'High Risk').sum()} ({(student_df['RiskCategory'] == 'High Risk').mean() * 100:.1f}%)")
print(f"Top Performing Department: {dept_metrics.iloc[dept_metrics['CGPA_mean'].argmax()]['Department']}")
print(f"Department with Highest Placement Rate: {dept_metrics.iloc[dept_metrics['SuccessRate'].argmax()]['Department']} ({dept_metrics['SuccessRate'].max():.1f}%)")
print("\nAnalysis Complete! All visualizations and reports saved.")

# prompt: create a code to get pdf report for above code and outputs with images and each image hav esome relavent information

!pip install reportlab
!pip install PyPDF2

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
import os
import pandas as pd

def create_pdf_report(images_folder=".", output_filename="student_analytics_report.pdf"):
    # Create a new PDF file
    c = canvas.Canvas(output_filename, pagesize=letter)

    # Define variables to control the layout of the PDF
    x_offset = 50  # Left margin
    y_offset = 750 # Top margin
    image_width = 400
    image_height = 300

    # Add a title to the PDF
    c.setFont("Helvetica-Bold", 20)
    c.drawString(x_offset, y_offset, "Student Analytics Report")
    y_offset -= 50

    # Loop through all the images in the specified folder
    for filename in sorted(os.listdir(images_folder)):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):  # Update file types
            # Load the image
            try:
                image_path = os.path.join(images_folder, filename)
                img = Image.open(image_path)
                img_reader = ImageReader(img)
            except FileNotFoundError:
                print(f"Warning: Image file not found: {image_path}")
                continue

            # Draw the image on the PDF
            c.drawImage(img_reader, x_offset, y_offset - image_height, width=image_width, height=image_height)

            # Add image description
            c.setFont("Helvetica", 12)
            c.drawString(x_offset, y_offset - image_height - 20, filename[:-4].replace("_", " ").title())

            # Move to the next position for the next image
            y_offset -= image_height + 40  # Adjust spacing as needed

            # Add a new page if we're running out of space
            if y_offset < 100:
                c.showPage()
                y_offset = 750

    # Save the PDF file
    c.save()

# Call the function to create the PDF report.
create_pdf_report()
