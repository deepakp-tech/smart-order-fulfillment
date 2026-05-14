import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Load your historical data
# Ensure your assignments.csv or historical_data.csv has these columns
try:
    data = pd.read_csv("../data/historical_assignments.csv")
except FileNotFoundError:
    print("Error: Historical data file not found. Please ensure you have a CSV with features and labels.")
    exit()

# 2. Preprocessing
# Mapping urgency to numerical values
urgency_map = {"High": 0, "Medium": 1, "Low": 2}
data["urgency_rank"] = data["urgency"].map(urgency_map)

# Features: Customer Lat/Long, Order Quantity, Urgency
X = data[["latitude", "longitude", "quantity", "urgency_rank"]]
y = data["assigned_warehouse"]

# Encode the target warehouse names into integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split for training and evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 3. Define Level 0 Models (Base Learners)
# KNN needs scaling to handle coordinates effectively
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# 4. Define Level 1 Model and the Stack
base_learners = [
    ('spatial_knn', knn_pipeline),
    ('operational_rf', rf_model)
]

#decides which base learner to trust
stack_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(),
    cv=5  # Uses 5-fold cross-validation to train Level 1 on Level 0 predictions
)

# 5. Train and Save
print("Training the Level 0 & 1 Hybrid Model...")
stack_model.fit(X_train, y_train)

# Save the full stack and the label encoder
model_package = {
    'model': stack_model,
    'label_encoder': le,
    'urgency_map': urgency_map
}

joblib.dump(model_package, "../models/smart_fulfillment_stack.pkl")
print("Hybrid model successfully saved to ../models/smart_fulfillment_stack.pkl")

# Evaluate accuracy
accuracy = stack_model.score(X_test, y_test)
print(f"Model Validation Accuracy: {accuracy:.2%}")