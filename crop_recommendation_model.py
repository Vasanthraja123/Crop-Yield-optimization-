import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import sys
import numpy as np
import warnings

# Suppress specific XGBoost warnings about use_label_encoder
warnings.filterwarnings('ignore', message='Parameters: { "use_label_encoder" } are not used.')

def main():
    try:
        # 1. Load data
        df = pd.read_csv("Crop_recommendation.csv")  # has columns: Nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall, label
    except FileNotFoundError:
        print("Error: Crop_recommendation.csv file not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Check and correct column names if needed
    expected_columns = ["Nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall", "label"]
    for col in expected_columns:
        if col not in df.columns:
            print(f"Error: Expected column '{col}' not found in dataset columns: {list(df.columns)}")
            sys.exit(1)

    # Use correct column names with exact case
    X = df[["Nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"]]
    y = df["label"]
    
    # 2. Encode labels to numeric values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save the label encoder classes for future prediction
    np.save('label_encoder_classes.npy', label_encoder.classes_)
    print(f"Original labels: {label_encoder.classes_}")
    print(f"Encoded as: {np.unique(y_encoded)}")
    
    # 3. Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )  # preserves class proportions

    # 4. Balance classes with SMOTE
    # Adjust k_neighbors if any class has fewer samples than 6 (k_neighbors+1)
    min_class_count = pd.Series(y_train).value_counts().min()
    k_neighbors = 5
    if min_class_count <= k_neighbors:
        k_neighbors = max(1, min_class_count - 1)
        print(f"Warning: Adjusting SMOTE k_neighbors to {k_neighbors} due to small class size.")

    smote = SMOTE(random_state=42, sampling_strategy="auto", k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)  # synthetic minority oversampling

    # 5. Define XGBoost estimator - REMOVED deprecated parameter
    xgb_clf = xgb.XGBClassifier(
        objective="multi:softprob",      # multiclass with probability outputs
        num_class=len(label_encoder.classes_),  # Use number of classes from label encoder
        eval_metric="mlogloss",
        seed=42
    )

    # 6. Hyperparameter grid
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0]
    }

    # 7. Grid search with 5-fold stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        verbose=2,
        n_jobs=-1
    )
    grid.fit(X_res, y_res)  # will automatically early stop on no improvement

    print("Best parameters:", grid.best_params_)

    # 8. Evaluate on test set
    best_model = grid.best_estimator_
    y_pred_encoded = best_model.predict(X_test)
    
    # Convert back to original labels for the report
    y_pred_original = label_encoder.inverse_transform(y_pred_encoded)
    y_test_original = label_encoder.inverse_transform(y_test)

    print("✅ Test Accuracy:", accuracy_score(y_test, y_pred_encoded))
    print("🔍 Classification Report (using numeric labels):\n", classification_report(y_test, y_pred_encoded))
    print("🔍 Classification Report (using original crop names):\n", 
          classification_report(y_test_original, y_pred_original))

    # 9. Save the trained model and label encoder
    joblib.dump(best_model, "crop_recommender_xgb.pkl")
    joblib.dump(label_encoder, "crop_label_encoder.pkl")
    print("💾 Model saved to crop_recommender_xgb.pkl")
    print("💾 Label encoder saved to crop_label_encoder.pkl")
    
    # Create a simple prediction function to demonstrate using the model
    print("\n--- Example of using the model for prediction ---")
    
    # Create a sample input
    sample = X.iloc[0:1]  # Just taking the first row as an example
    print(f"Sample input (soil conditions): \n{sample.to_dict('records')[0]}")
    
    # Predict using the trained model
    predicted_class_encoded = best_model.predict(sample)[0]
    predicted_crop = label_encoder.inverse_transform([predicted_class_encoded])[0]
    print(f"Predicted crop: {predicted_crop}")
    
    # Show how to use the model for future predictions
    print("\n--- How to use this model for future predictions ---")
    print("""
# Example code for prediction:
import joblib

# Load the saved model and label encoder
model = joblib.load('crop_recommender_xgb.pkl')
label_encoder = joblib.load('crop_label_encoder.pkl')

# Example input data (N, P, K, temperature, humidity, ph, rainfall)
new_data = [[83, 45, 60, 28, 70.3, 7.0, 150.9]]

# Make prediction
predicted_label_encoded = model.predict(new_data)[0]
predicted_crop = label_encoder.inverse_transform([predicted_label_encoded])[0]
print(f"Recommended crop: {predicted_crop}")
    """)

if __name__ == "__main__":
    main()