# final_working_solution.py
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image
import glob
import joblib
import matplotlib.pyplot as plt

def extract_certificate_features(img_path):
    """Extract features that distinguish fake vs authentic certificates"""
    img = Image.open(img_path)
    img_array = np.array(img.convert('RGB'))
    
    features = []
    
    # 1. Basic color statistics
    brightness = np.mean(img_array)
    red_mean = np.mean(img_array[:, :, 0])
    green_mean = np.mean(img_array[:, :, 1])
    blue_mean = np.mean(img_array[:, :, 2])
    
    # 2. Color variance (texture)
    brightness_std = np.std(img_array)
    red_std = np.std(img_array[:, :, 0])
    green_std = np.std(img_array[:, :, 1])
    blue_std = np.std(img_array[:, :, 2])
    
    # 3. Color ratios (might indicate different paper/printing)
    rg_ratio = red_mean / (green_mean + 1e-8)
    rb_ratio = red_mean / (blue_mean + 1e-8)
    gb_ratio = green_mean / (blue_mean + 1e-8)
    
    # 4. Image dimensions
    width, height = img.size
    aspect_ratio = width / height
    
    # 5. File characteristics
    file_size = os.path.getsize(img_path)
    
    # 6. Edge density (might detect different printing quality)
    from scipy import ndimage
    gray_img = np.mean(img_array, axis=2)
    edges = ndimage.sobel(gray_img)
    edge_density = np.mean(np.abs(edges))
    
    # 7. Brightness distribution
    brightness_skew = np.mean((img_array - brightness) ** 3) / (brightness_std ** 3 + 1e-8)
    
    # 8. Color saturation
    max_rgb = np.max(img_array, axis=2)
    min_rgb = np.min(img_array, axis=2)
    saturation = np.mean((max_rgb - min_rgb) / (max_rgb + 1e-8))
    
    # Combine all features
    features.extend([
        brightness, red_mean, green_mean, blue_mean,
        brightness_std, red_std, green_std, blue_std,
        rg_ratio, rb_ratio, gb_ratio,
        width, height, aspect_ratio,
        file_size, edge_density, brightness_skew, saturation
    ])
    
    return features

def create_feature_dataset(data_path):
    """Create dataset with manual features"""
    features_list = []
    labels = []
    filenames = []
    
    for split in ['train', 'test', 'valid']:
        split_path = os.path.join(data_path, split)
        if not os.path.exists(split_path):
            continue
            
        for img_file in glob.glob(os.path.join(split_path, "*.jpg")) + \
                       glob.glob(os.path.join(split_path, "*.png")):
            try:
                # Extract features
                feature_vector = extract_certificate_features(img_file)
                features_list.append(feature_vector)
                
                # Get label
                if 'Upto-4th-Sem' in os.path.basename(img_file):
                    labels.append(1)  # authentic
                else:
                    labels.append(0)  # fake
                
                filenames.append(os.path.basename(img_file))
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    return np.array(features_list), np.array(labels), filenames

def train_random_forest_model(X, y):
    """Train Random Forest classifier"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Authentic']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, X_test, y_test, y_pred

def analyze_feature_importance(model, feature_names):
    """Analyze which features are most important"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n📊 FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    for i in range(min(10, len(indices))):
        print(f"{i+1:2d}. {feature_names[indices[i]]:20s} {importances[indices[i]]:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.title("Top 10 Feature Importances")
    plt.bar(range(10), importances[indices[:10]])
    plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    data_path = "C:/Users/omkar/Downloads/Certificate forgery detection.v1i.yolokeras"
    
    print("🎯 FINAL WORKING SOLUTION - Manual Feature Extraction")
    print("="*60)
    
    # Create feature dataset
    print("📊 Extracting features from certificates...")
    X, y, filenames = create_feature_dataset(data_path)
    
    if len(X) == 0:
        print("❌ No features extracted!")
        return
    
    print(f"✅ Extracted {X.shape[0]} samples with {X.shape[1]} features")
    print(f"📈 Class distribution - Fake: {np.sum(y == 0)}, Authentic: {np.sum(y == 1)}")
    
    # Feature names for interpretation
    feature_names = [
        'brightness', 'red_mean', 'green_mean', 'blue_mean',
        'brightness_std', 'red_std', 'green_std', 'blue_std',
        'rg_ratio', 'rb_ratio', 'gb_ratio',
        'width', 'height', 'aspect_ratio',
        'file_size', 'edge_density', 'brightness_skew', 'saturation'
    ]
    
    # Train model
    print("\n🤖 Training Random Forest model...")
    model, X_test, y_test, y_pred = train_random_forest_model(X, y)
    
    # Analyze feature importance
    analyze_feature_importance(model, feature_names)
    
    # Save model
    joblib.dump(model, 'certificate_detector_rf.pkl')
    print(f"\n💾 Model saved as 'certificate_detector_rf.pkl'")
    
    # Save feature names
    feature_info = {
        'feature_names': feature_names,
        'feature_count': len(feature_names)
    }
    joblib.dump(feature_info, 'feature_info.pkl')
    print("💾 Feature information saved as 'feature_info.pkl'")
    
    # Final assessment
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > 0.95:
        print(f"\n🎉 EXCELLENT! Model achieved {accuracy:.1%} accuracy!")
        print("Your certificates have clear statistical differences!")
    elif accuracy > 0.85:
        print(f"\n✅ VERY GOOD! Model achieved {accuracy:.1%} accuracy!")
    else:
        print(f"\n⚠️ Model achieved {accuracy:.1%} accuracy.")

# Function to predict new certificates
def predict_certificate(model_path, feature_info_path, image_path):
    """Predict if a certificate is fake or authentic"""
    # Load model and feature info
    model = joblib.load(model_path)
    feature_info = joblib.load(feature_info_path)
    
    # Extract features from new image
    features = extract_certificate_features(image_path)
    
    # Make prediction
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]
    
    class_name = "Authentic" if prediction == 1 else "Fake"
    confidence = probability[prediction]
    
    print(f"📄 Certificate: {os.path.basename(image_path)}")
    print(f"🔍 Prediction: {class_name}")
    print(f"📊 Confidence: {confidence:.4f}")
    print(f"💯 Fake probability: {probability[0]:.4f}")
    print(f"✅ Authentic probability: {probability[1]:.4f}")
    
    return class_name, confidence

if __name__ == "__main__":
    main()
    
    # Example of how to use the trained model
    print("\n" + "="*60)
    print("🔮 PREDICTION EXAMPLE")
    print("="*60)
    
    # You can use this to predict new certificates:
    # predict_certificate('certificate_detector_rf.pkl', 'feature_info.pkl', 'path/to/your/certificate.jpg')