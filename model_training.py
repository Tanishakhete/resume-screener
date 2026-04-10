"""
Model Training Module for Resume Screening System.
Handles feature extraction, model training, and evaluation.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


class FeatureExtractor:
    """Extract features from text using various vectorization techniques."""
    
    def __init__(self, method='tfidf', max_features=5000, ngram_range=(1, 2)):
        """
        Initialize feature extractor.
        
        Args:
            method: 'tfidf' or 'count'
            max_features: Maximum number of features
            ngram_range: N-gram range for vectorization
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.8,
                stop_words='english'
            )
        elif method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.8,
                stop_words='english'
            )
        else:
            raise ValueError("Method must be 'tfidf' or 'count'")
    
    def fit_transform(self, texts):
        """
        Fit vectorizer and transform texts.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            array: Feature matrix
        """
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """
        Transform texts using fitted vectorizer.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            array: Feature matrix
        """
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names from vectorizer."""
        return self.vectorizer.get_feature_names_out()
    
    def save(self, filepath):
        """Save vectorizer to file."""
        joblib.dump(self.vectorizer, filepath)
    
    def load(self, filepath):
        """Load vectorizer from file."""
        self.vectorizer = joblib.load(filepath)


class ResumeClassifier:
    """Machine learning classifier for resume categorization."""
    
    def __init__(self, algorithm='logistic_regression', **kwargs):
        """
        Initialize classifier.
        
        Args:
            algorithm: 'naive_bayes', 'logistic_regression', 'svm', or 'random_forest'
            **kwargs: Additional parameters for the classifier
        """
        self.algorithm = algorithm
        self.model = None
        self.is_trained = False
        
        if algorithm == 'naive_bayes':
            self.model = MultinomialNB(**kwargs)
        elif algorithm == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                **kwargs
            )
        elif algorithm == 'svm':
            self.model = SVC(
                probability=True,
                random_state=42,
                **kwargs
            )
        elif algorithm == 'random_forest':
            self.model = RandomForestClassifier(
                random_state=42,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def train(self, X_train, y_train):
        """
        Train the classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            array: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            array: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, target_names=None):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            target_names: Optional list of class names
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted'),
            'classification_report': classification_report(
                y_test, predictions, 
                target_names=target_names,
                output_dict=True
            )
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds
            
        Returns:
            dict: Cross-validation scores
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores
        }
    
    def save(self, filepath):
        """Save model to file."""
        joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        """Load model from file."""
        self.model = joblib.load(filepath)
        self.is_trained = True


class ModelTrainer:
    """Complete model training pipeline."""
    
    def __init__(self, algorithm='logistic_regression', vectorizer_method='tfidf'):
        """
        Initialize model trainer.
        
        Args:
            algorithm: ML algorithm to use
            vectorizer_method: 'tfidf' or 'count'
        """
        self.algorithm = algorithm
        self.vectorizer_method = vectorizer_method
        self.feature_extractor = None
        self.classifier = None
        self.label_mapping = None
        self.reverse_label_mapping = None
    
    def prepare_data(self, texts, labels, test_size=0.2, random_state=42):
        """
        Prepare data for training.
        
        Args:
            texts: List of preprocessed texts
            labels: List of labels
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            tuple: Train and test splits
        """
        # Create label mapping
        unique_labels = sorted(set(labels))
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}
        
        # Convert labels to numeric
        numeric_labels = [self.label_mapping[label] for label in labels]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, numeric_labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=numeric_labels
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, texts, labels, test_size=0.2):
        """
        Complete training pipeline.
        
        Args:
            texts: List of preprocessed texts
            labels: List of labels
            test_size: Proportion of test set
            
        Returns:
            dict: Training results including metrics
        """
        # Prepare data
        X_train_texts, X_test_texts, y_train, y_test = self.prepare_data(
            texts, labels, test_size
        )
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(method=self.vectorizer_method)
        
        # Extract features
        X_train = self.feature_extractor.fit_transform(X_train_texts)
        X_test = self.feature_extractor.transform(X_test_texts)
        
        # Initialize and train classifier
        self.classifier = ResumeClassifier(algorithm=self.algorithm)
        self.classifier.train(X_train, y_train)
        
        # Evaluate
        target_names = [self.reverse_label_mapping[i] for i in range(len(self.label_mapping))]
        metrics = self.classifier.evaluate(X_test, y_test, target_names)
        
        # Cross-validation
        X_full = self.feature_extractor.transform(texts)
        y_full = [self.label_mapping[label] for label in labels]
        cv_results = self.classifier.cross_validate(X_full, y_full)
        
        results = {
            'metrics': metrics,
            'cross_validation': cv_results,
            'label_mapping': self.label_mapping,
            'reverse_label_mapping': self.reverse_label_mapping,
            'algorithm': self.algorithm,
            'vectorizer_method': self.vectorizer_method
        }
        
        return results
    
    def predict(self, texts):
        """
        Make predictions on new texts.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            list: Predictions with labels and probabilities
        """
        # Extract features
        X = self.feature_extractor.transform(texts)
        
        # Predict
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        # Convert to labels
        results = []
        for pred, probs in zip(predictions, probabilities):
            label = self.reverse_label_mapping[pred]
            confidence = probs[pred]
            all_probs = {
                self.reverse_label_mapping[i]: prob 
                for i, prob in enumerate(probs)
            }
            results.append({
                'prediction': label,
                'confidence': confidence,
                'all_probabilities': all_probs
            })
        
        return results
    
    def save(self, model_dir='models'):
        """
        Save trained model and vectorizer.
        
        Args:
            model_dir: Directory to save models
        """
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save feature extractor
        self.feature_extractor.save(f'{model_dir}/vectorizer.joblib')
        
        # Save classifier
        self.classifier.save(f'{model_dir}/classifier.joblib')
        
        # Save label mappings and metadata
        metadata = {
            'label_mapping': self.label_mapping,
            'reverse_label_mapping': self.reverse_label_mapping,
            'algorithm': self.algorithm,
            'vectorizer_method': self.vectorizer_method
        }
        joblib.dump(metadata, f'{model_dir}/metadata.joblib')
        
        print(f"Model saved to {model_dir}/")
    
    def load(self, model_dir='models'):
        """
        Load trained model and vectorizer.
        
        Args:
            model_dir: Directory containing saved models
        """
        # Load metadata
        metadata = joblib.load(f'{model_dir}/metadata.joblib')
        self.label_mapping = metadata['label_mapping']
        self.reverse_label_mapping = metadata['reverse_label_mapping']
        self.algorithm = metadata['algorithm']
        self.vectorizer_method = metadata['vectorizer_method']
        
        # Load feature extractor
        self.feature_extractor = FeatureExtractor(method=self.vectorizer_method)
        self.feature_extractor.load(f'{model_dir}/vectorizer.joblib')
        
        # Load classifier
        self.classifier = ResumeClassifier(algorithm=self.algorithm)
        self.classifier.load(f'{model_dir}/classifier.joblib')
        
        print(f"Model loaded from {model_dir}/")


def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_algorithms(texts, labels, algorithms=None):
    """
    Compare different algorithms.
    
    Args:
        texts: List of preprocessed texts
        labels: List of labels
        algorithms: List of algorithms to compare
        
    Returns:
        DataFrame: Comparison results
    """
    if algorithms is None:
        algorithms = ['naive_bayes', 'logistic_regression', 'svm']
    
    results = []
    
    for algo in algorithms:
        print(f"\nTraining {algo}...")
        trainer = ModelTrainer(algorithm=algo)
        train_results = trainer.train(texts, labels)
        
        metrics = train_results['metrics']
        cv = train_results['cross_validation']
        
        results.append({
            'Algorithm': algo,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'CV Mean': cv['mean_accuracy'],
            'CV Std': cv['std_accuracy']
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Model Training Module - Test with sample data")
    print("=" * 50)
    
    # Sample data for testing
    sample_texts = [
        "python machine learning data analysis tensorflow keras neural network deep learning",
        "web development javascript html css react angular nodejs frontend backend",
        "human resources recruitment hiring talent management employee relations hr",
        "data science statistics analytics pandas numpy matplotlib seaborn visualization",
        "software engineering java spring boot microservices api development testing"
    ] * 10  # Replicate for more samples
    
    sample_labels = [
        'Data Scientist', 'Web Developer', 'HR', 'Data Scientist', 'Software Engineer'
    ] * 10
    
    # Train model
    trainer = ModelTrainer(algorithm='logistic_regression')
    results = trainer.train(sample_texts, sample_labels, test_size=0.2)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Precision: {results['metrics']['precision']:.4f}")
    print(f"Recall: {results['metrics']['recall']:.4f}")
    print(f"F1-Score: {results['metrics']['f1_score']:.4f}")
    
    print(f"\nCV Accuracy: {results['cross_validation']['mean_accuracy']:.4f} (+/- {results['cross_validation']['std_accuracy']:.4f})")
