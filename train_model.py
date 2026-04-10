"""
Training Script for Resume Screening System.
Downloads dataset and trains the classification model.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# Import custom modules
from data_preprocessing import TextPreprocessor
from model_training import ModelTrainer, compare_algorithms


def download_sample_dataset():
    """
    Create a sample resume dataset if no dataset is available.
    In production, replace this with actual dataset loading.
    """
    print("Creating sample resume dataset...")
    
    # Sample data for demonstration
    # In production, replace with actual resume dataset from Kaggle
    data = {
        'resume_text': [
            # Data Scientist examples
            "Python data scientist machine learning tensorflow keras deep learning neural networks pandas numpy scikit-learn data analysis visualization statistics SQL data mining predictive modeling",
            "Data science expert with 5 years experience in Python R SQL machine learning algorithms random forest gradient boosting regression classification clustering",
            "Machine learning engineer specializing in NLP computer vision deep learning frameworks pytorch tensorflow keras model deployment AWS cloud",
            "Senior data scientist statistical analysis hypothesis testing A/B testing experimental design Bayesian statistics time series forecasting",
            "Data analyst business intelligence tableau powerbi SQL data warehousing ETL data pipelines reporting dashboards metrics KPIs",
            
            # Web Developer examples
            "Full stack web developer JavaScript React Angular Vue Node.js Express MongoDB HTML CSS responsive design REST API GraphQL",
            "Frontend developer React Redux TypeScript webpack babel sass css-in-js component testing jest enzyme HTML5 CSS3",
            "Backend developer Python Django Flask Node.js Express PostgreSQL MySQL Redis caching microservices architecture API design",
            "Web developer PHP Laravel WordPress Drupal CMS e-commerce payment integration shopping cart inventory management",
            "JavaScript developer ES6+ async await promises functional programming DOM manipulation event handling AJAX fetch API",
            
            # HR examples
            "Human resources specialist recruitment talent acquisition interviewing onboarding employee relations performance management",
            "HR manager compensation benefits payroll administration labor law compliance diversity inclusion training development",
            "Recruiter sourcing candidates LinkedIn Indeed job boards applicant tracking systems ATS interviewing skills assessment",
            "Talent acquisition specialist employer branding campus recruiting executive search headhunting negotiation offer management",
            "HR business partner organizational development change management succession planning workforce planning talent management",
            
            # Software Engineer examples
            "Software engineer Java Spring Boot microservices REST API CI/CD Jenkins Git version control agile scrum testing",
            "Senior software developer C++ system design algorithms data structures performance optimization memory management debugging",
            "Software architect cloud native Kubernetes Docker containerization AWS Azure GCP infrastructure as code terraform",
            "Mobile app developer Android Kotlin iOS Swift React Native Flutter cross-platform development mobile UI UX",
            "DevOps engineer automation scripting bash python ruby ansible puppet chef monitoring logging ELK stack prometheus",
            
            # Data Analyst examples
            "Data analyst SQL Excel pivot tables vlookup macros VBA reporting business analysis requirements gathering process improvement",
            "Business analyst Tableau PowerBI QlikView data visualization storytelling stakeholder management project management agile",
            "Financial analyst budgeting forecasting variance analysis financial modeling excel advanced formulas pivot charts",
            "Marketing analyst Google Analytics SEO SEM digital marketing campaign analysis customer segmentation CRM Salesforce",
            "Operations analyst supply chain logistics inventory optimization Six Sigma lean manufacturing process mapping",
            
            # Network Engineer examples
            "Network engineer Cisco CCNA CCNP routing switching VLAN BGP OSPF firewall security VPN WAN LAN",
            "Systems administrator Linux Windows server administration Active Directory DNS DHCP VMware virtualization",
            "Cloud engineer AWS solutions architect EC2 S3 RDS Lambda serverless cloudformation terraform networking VPC",
            "Security engineer cybersecurity penetration testing vulnerability assessment SIEM incident response compliance",
            "Database administrator Oracle MySQL PostgreSQL MongoDB replication backup recovery performance tuning indexing",
        ] * 5,  # Replicate for more samples
        'category': [
            'Data Scientist', 'Data Scientist', 'Data Scientist', 'Data Scientist', 'Data Scientist',
            'Web Developer', 'Web Developer', 'Web Developer', 'Web Developer', 'Web Developer',
            'HR', 'HR', 'HR', 'HR', 'HR',
            'Software Engineer', 'Software Engineer', 'Software Engineer', 'Software Engineer', 'Software Engineer',
            'Data Analyst', 'Data Analyst', 'Data Analyst', 'Data Analyst', 'Data Analyst',
            'Network Engineer', 'Network Engineer', 'Network Engineer', 'Network Engineer', 'Network Engineer',
        ] * 5
    }
    
    df = pd.DataFrame(data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/resume_dataset.csv', index=False)
    
    print(f"✅ Sample dataset created with {len(df)} samples")
    return 'data/resume_dataset.csv'


def train_model(dataset_path, algorithm='logistic_regression', 
                vectorizer_method='tfidf', compare=False):
    """
    Train the resume classification model.
    
    Args:
        dataset_path: Path to the dataset CSV
        algorithm: ML algorithm to use
        vectorizer_method: 'tfidf' or 'count'
        compare: Whether to compare multiple algorithms
    """
    print("=" * 60)
    print("Resume Screening System - Model Training")
    print("=" * 60)
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"✅ Loaded {len(df)} samples")
    print(f"📊 Categories: {df['category'].nunique()}")
    print(f"📋 Category distribution:")
    print(df['category'].value_counts())
    
    # Initialize preprocessor
    print("\n🔧 Initializing text preprocessor...")
    preprocessor = TextPreprocessor()
    
    # Preprocess texts
    print("📝 Preprocessing resume texts...")
    processed_texts = []
    for idx, text in enumerate(df['resume_text']):
        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(df)} resumes...")
        processed_texts.append(preprocessor.preprocess(str(text)))
    
    print(f"✅ Preprocessing complete")
    
    labels = df['category'].values
    
    # Compare algorithms if requested
    if compare:
        print("\n🔍 Comparing algorithms...")
        comparison_df = compare_algorithms(processed_texts, labels)
        print("\n📊 Algorithm Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv('data/algorithm_comparison.csv', index=False)
        print("\n💾 Comparison saved to data/algorithm_comparison.csv")
        
        # Select best algorithm
        best_algo = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Algorithm']
        print(f"\n🏆 Best algorithm: {best_algo}")
        algorithm = best_algo
    
    # Train model
    print(f"\n🤖 Training {algorithm} model...")
    trainer = ModelTrainer(
        algorithm=algorithm,
        vectorizer_method=vectorizer_method
    )
    
    results = trainer.train(processed_texts, labels, test_size=0.2)
    
    # Print results
    print("\n" + "=" * 60)
    print("📈 Training Results")
    print("=" * 60)
    
    metrics = results['metrics']
    print(f"\n✅ Accuracy:  {metrics['accuracy']:.4f}")
    print(f"✅ Precision: {metrics['precision']:.4f}")
    print(f"✅ Recall:    {metrics['recall']:.4f}")
    print(f"✅ F1-Score:  {metrics['f1_score']:.4f}")
    
    cv = results['cross_validation']
    print(f"\n🔁 Cross-Validation Accuracy: {cv['mean_accuracy']:.4f} (+/- {cv['std_accuracy']:.4f})")
    
    # Print classification report
    print("\n📋 Classification Report:")
    report = metrics['classification_report']
    for category in report.keys():
        if category not in ['accuracy', 'macro avg', 'weighted avg']:
            precision = report[category]['precision']
            recall = report[category]['recall']
            f1 = report[category]['f1-score']
            support = report[category]['support']
            print(f"  {category:20s} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    # Save model
    print("\n💾 Saving model...")
    trainer.save('models')
    
    # Save evaluation results
    joblib.dump(results, 'evaluation_results.joblib')
    print("💾 Evaluation results saved to evaluation_results.joblib")
    
    print("\n" + "=" * 60)
    print("✅ Training complete! Model saved to models/")
    print("=" * 60)
    
    return trainer, results


def main():
    """Main function to run training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Resume Screening Model')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to dataset CSV (default: create sample dataset)')
    parser.add_argument('--algorithm', type=str, default='logistic_regression',
                       choices=['naive_bayes', 'logistic_regression', 'svm', 'random_forest'],
                       help='ML algorithm to use')
    parser.add_argument('--vectorizer', type=str, default='tfidf',
                       choices=['tfidf', 'count'],
                       help='Vectorization method')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple algorithms')
    
    args = parser.parse_args()
    
    # Download or use provided dataset
    if args.dataset is None:
        dataset_path = download_sample_dataset()
    else:
        dataset_path = args.dataset
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found: {dataset_path}")
            print("Creating sample dataset instead...")
            dataset_path = download_sample_dataset()
    
    # Train model
    train_model(
        dataset_path=dataset_path,
        algorithm=args.algorithm,
        vectorizer_method=args.vectorizer,
        compare=args.compare
    )


if __name__ == "__main__":
    main()
