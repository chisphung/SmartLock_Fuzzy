#!/usr/bin/env python3
"""
CSI People Counting Model Training Script

This script trains a machine learning model to predict the number of people
in a room based on WiFi CSI (Channel State Information) data.

Usage:
    python train_csi_model.py --data training_data.jsonl --output csi_model.pkl
    python train_csi_model.py --data training_data.jsonl --model rf  # Random Forest
    python train_csi_model.py --data training_data.jsonl --model nn  # Neural Network
"""

import argparse
import json
import os
import pickle
from datetime import datetime
from typing import List, Tuple

import numpy as np


def load_training_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and parse CSI training data from JSONL file.
    
    Args:
        data_path: Path to the training_data.jsonl file
        
    Returns:
        X: Feature matrix (N x num_features)
        y: Labels (people count)
    """
    X_list = []
    y_list = []
    
    print(f"Loading data from {data_path}...")
    
    with open(data_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                amplitudes = record.get('amplitudes', [])
                people_count = record.get('people_count', 0)
                rssi = record.get('rssi', 0)
                
                if not amplitudes:
                    continue
                
                # Create feature vector
                features = extract_features(amplitudes, rssi)
                X_list.append(features)
                y_list.append(people_count)
                
            except json.JSONDecodeError:
                continue
    
    if not X_list:
        raise ValueError("No valid samples found in training data!")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Loaded {len(y)} samples")
    print(f"Feature shape: {X.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y


def extract_features(amplitudes: List[int], rssi: int) -> np.ndarray:
    """
    Extract features from raw CSI amplitude data.
    
    Features:
    - Statistical features (mean, std, min, max, median)
    - Frequency domain features (FFT-based)
    - RSSI
    
    Args:
        amplitudes: List of CSI amplitude values
        rssi: RSSI value
        
    Returns:
        Feature vector
    """
    amp = np.array(amplitudes, dtype=np.float32)
    
    # Handle edge cases
    if len(amp) == 0:
        return np.zeros(20)
    
    features = []
    
    # Basic statistics
    features.append(np.mean(amp))
    features.append(np.std(amp))
    features.append(np.min(amp))
    features.append(np.max(amp))
    features.append(np.median(amp))
    features.append(np.max(amp) - np.min(amp))  # Range
    
    # Quartiles
    features.append(np.percentile(amp, 25))
    features.append(np.percentile(amp, 75))
    
    # Higher order statistics
    if len(amp) > 1 and np.std(amp) > 0:
        # Skewness
        skew = np.mean(((amp - np.mean(amp)) / np.std(amp)) ** 3)
        # Kurtosis
        kurt = np.mean(((amp - np.mean(amp)) / np.std(amp)) ** 4) - 3
    else:
        skew = 0
        kurt = 0
    features.append(skew)
    features.append(kurt)
    
    # Energy features
    features.append(np.sum(amp ** 2))  # Total energy
    features.append(np.mean(amp ** 2))  # Mean energy
    
    # Variance-based features
    if len(amp) >= 4:
        # Divide into segments and compute variance
        segments = np.array_split(amp, 4)
        seg_vars = [np.var(s) for s in segments if len(s) > 0]
        features.extend(seg_vars[:4] if len(seg_vars) >= 4 else seg_vars + [0] * (4 - len(seg_vars)))
    else:
        features.extend([0, 0, 0, 0])
    
    # FFT-based features (if enough samples)
    if len(amp) >= 8:
        fft = np.abs(np.fft.fft(amp))[:len(amp)//2]
        features.append(np.mean(fft))
        features.append(np.max(fft))
    else:
        features.extend([0, 0])
    
    # RSSI
    features.append(rssi)
    
    # Number of subcarriers (for normalization)
    features.append(len(amp))
    
    return np.array(features)


def train_random_forest(X: np.ndarray, y: np.ndarray, n_estimators: int = 100):
    """Train a Random Forest classifier."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Train model
    print(f"\nTraining Random Forest with {n_estimators} trees...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Cross-validation
    if len(y) >= 5:
        cv_scores = cross_val_score(model, X, y, cv=min(5, len(np.unique(y))))
        print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Classification report
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    feature_names = [
        'mean', 'std', 'min', 'max', 'median', 'range',
        'q25', 'q75', 'skew', 'kurt',
        'energy_sum', 'energy_mean',
        'var_seg1', 'var_seg2', 'var_seg3', 'var_seg4',
        'fft_mean', 'fft_max',
        'rssi', 'n_subcarriers'
    ]
    importances = sorted(zip(feature_names, model.feature_importances_), 
                         key=lambda x: x[1], reverse=True)
    for name, imp in importances[:10]:
        print(f"  {name}: {imp:.4f}")
    
    return model


def train_neural_network(X: np.ndarray, y: np.ndarray):
    """Train a simple neural network classifier."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, 
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Train model
    print("\nTraining Neural Network...")
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Return both model and scaler
    return {'model': model, 'scaler': scaler}


def train_gradient_boosting(X: np.ndarray, y: np.ndarray):
    """Train a Gradient Boosting classifier."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Train model
    print("\nTraining Gradient Boosting...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model


def save_model(model, output_path: str):
    """Save trained model to file."""
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CSI People Counting Model")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to training_data.jsonl file")
    parser.add_argument("--output", type=str, default="csi_model.pkl",
                        help="Output path for trained model")
    parser.add_argument("--model", type=str, default="rf",
                        choices=["rf", "nn", "gb"],
                        help="Model type: rf (Random Forest), nn (Neural Network), gb (Gradient Boosting)")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Number of trees for Random Forest/Gradient Boosting")
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        print("\nTo collect training data:")
        print("  1. Run the server: cd server_side/backend && python main.py")
        print("  2. Run edge server: cd edge_side/infra && python ws_server.py")
        print("  3. Power on ESP32-CAM")
        print("  4. Data will be saved to edge_side/infra/csi_data/training_data.jsonl")
        return
    
    # Load data
    X, y = load_training_data(args.data)
    
    if len(np.unique(y)) < 2:
        print("\nWarning: Only one class found in training data!")
        print("Collect data with different numbers of people for classification.")
        return
    
    # Train model
    print(f"\n{'='*50}")
    print(f"Training {args.model.upper()} model")
    print(f"{'='*50}")
    
    if args.model == "rf":
        model = train_random_forest(X, y, args.n_estimators)
    elif args.model == "nn":
        model = train_neural_network(X, y)
    elif args.model == "gb":
        model = train_gradient_boosting(X, y)
    
    # Save model
    save_model(model, args.output)
    
    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"{'='*50}")
    print(f"\nTo use the model for inference, load it with:")
    print(f"  import pickle")
    print(f"  with open('{args.output}', 'rb') as f:")
    print(f"      model = pickle.load(f)")
    print(f"  prediction = model.predict(features)")


if __name__ == "__main__":
    main()
