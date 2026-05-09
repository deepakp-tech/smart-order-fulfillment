# Smart Order Fulfillment System

An intelligent logistics optimization platform designed to automate warehouse assignment and streamline B2B order fulfillment operations. The system combines rule-based dispatching with machine learning-driven recommendations to assign customer orders to the most suitable warehouse based on inventory availability, delivery urgency, and geographic proximity.

The platform integrates both **K-Nearest Neighbors (KNN)** and **Random Forest** models to improve warehouse prediction accuracy and support scalable fulfillment decision-making.

---

# Features

## Intelligent Order Dispatch
- Automatically assigns orders to the nearest warehouse with sufficient inventory
- Prioritizes orders based on urgency levels
- Uses geodesic distance calculations for optimized warehouse selection
- Dynamically updates warehouse stock after each assignment

## Machine Learning Integration
- Implements both **KNN** and **Random Forest** classifiers
- Predicts the most suitable warehouse for incoming orders
- Retrains models automatically after dispatch execution
- Supports real-time prediction through REST APIs

## Interactive Dashboard
- Displays warehouse stock levels in real time
- Visualizes assignment data using Chart.js
- Allows CSV uploads for orders and warehouse datasets
- Provides downloadable assignment reports

## REST API Backend
- Built using Flask
- Supports upload, dispatch, prediction, and analytics endpoints
- Includes structured validation and error handling

## Data Handling & Persistence
- Uses Pandas for preprocessing and data management
- Stores trained ML models using Joblib serialization
- Supports CSV-based workflows for easy testing and deployment

---

# Tech Stack

## Backend

| Technology | Purpose |
|------------|---------|
| Flask | Backend API framework |
| Pandas | Data processing |
| Scikit-learn | Machine learning models |
| Geopy | Distance calculation |
| Joblib | Model persistence |
| Flask-CORS | Cross-origin support |

---

## Frontend

| Technology | Purpose |
|------------|---------|
| HTML5 | Structure |
| CSS3 | Responsive styling |
| Vanilla JavaScript | Frontend logic |
| Chart.js | Data visualization |

---

# System Workflow

```text
CSV Upload
   ↓
Data Validation
   ↓
Dispatch Algorithm
   ↓
Warehouse Assignment
   ↓
Inventory Update
   ↓
Model Training (KNN + Random Forest)
   ↓
Prediction API
```

---

# Dispatch Logic

The dispatch engine balances urgency, inventory availability, and geographic proximity to optimize warehouse allocation.

## Workflow
1. Orders are sorted based on urgency
2. Warehouses with sufficient stock are filtered
3. Geodesic distances are calculated
4. The nearest eligible warehouse is selected
5. Stock is updated dynamically
6. Assignment data is stored for ML training

---

## Sample Dispatch Function

```python
def dispatch_orders():

    # Sort orders based on urgency
    orders = orders.sort_values("urgency_rank")

    for order in orders:

        # Find warehouses with enough stock
        eligible = warehouses[
            warehouses.stock >= order.quantity
        ]

        # Calculate distances
        distances = [
            geodesic(order.location, wh.location).km
            for wh in eligible
        ]

        # Select nearest warehouse
        assigned_warehouse = eligible.iloc[min_distance_index]

        # Update stock
        assigned_warehouse.stock -= order.quantity

        # Save assignment
        record_assignment(order, assigned_warehouse)

    # Train ML models
    train_knn_model()
    train_random_forest_model()
```

---

# Machine Learning Models

The project uses two classification models for warehouse prediction.

## K-Nearest Neighbors (KNN)
- Similarity-based prediction model
- Effective for localized geographic patterns
- Configured with `k = 3`

## Random Forest Classifier
- Ensemble-based prediction model
- Handles complex decision patterns efficiently
- Improves prediction robustness and accuracy

---

## Feature Set

```text
[latitude, longitude, quantity, urgency_rank]
```

## Prediction Target

```text
warehouse_id
```

---

# API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload order and warehouse CSV files |
| POST | `/dispatch` | Execute dispatch and retrain models |
| GET | `/assignments` | Retrieve assignment results |
| POST | `/predict/knn` | Predict warehouse using KNN |
| POST | `/predict/random-forest` | Predict warehouse using Random Forest |
| GET | `/warehouse-stock` | Fetch warehouse inventory data |
| GET | `/download-assignments` | Download assignment results |

---

# Sample Prediction Request

## Request

```http
POST /predict/random-forest
Content-Type: application/json
```

```json
{
  "latitude": 13.08,
  "longitude": 80.27,
  "quantity": 2,
  "urgency": "High"
}
```

---

## Response

```json
{
  "predicted_warehouse_id": "WH-102",
  "model_used": "Random Forest"
}
```

---

# Installation

## Prerequisites
- Python 3.8+
- Modern web browser

---

## Clone the Repository

```bash
git clone https://github.com/your-username/smart-order-fulfillment-system.git

cd smart-order-fulfillment-system/backend
```

---

## Create Virtual Environment

### Linux / macOS

```bash
python -m venv venv

source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Prepare Sample Data

```bash
mkdir -p data models

cp ../sample_orders.csv data/orders.csv

cp ../sample_warehouses.csv data/warehouses.csv
```

---

## Run the Application

```bash
python app.py
```

The backend server will start at:

```text
http://localhost:5000
```

---

# Security & Validation

- CSV format validation
- File size restrictions
- Input sanitization
- Secure CORS configuration
- API-level validation and exception handling

---

# Configurable Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `KNN_NEIGHBORS` | `3` | Number of neighbors used in KNN |
| `RF_ESTIMATORS` | `100` | Number of trees in Random Forest |
| `UPLOAD_FOLDER` | `./data` | CSV upload directory |
| `MAX_CONTENT_LENGTH` | `16 MB` | Maximum upload size |

---

# Future Improvements

- Real-time route optimization
- Demand forecasting
- Docker containerization
- Cloud deployment
- Authentication and role-based access
- Database integration
- Live order tracking

---

