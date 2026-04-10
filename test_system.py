"""
Test Script for Resume Screening System.
Quick verification that all components work correctly.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing module imports...")
    try:
        from data_preprocessing import TextPreprocessor, TextExtractor
        from model_training import ModelTrainer, FeatureExtractor, ResumeClassifier
        from prediction import ResumePredictor
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def test_preprocessing():
    """Test text preprocessing."""
    print("\nTesting text preprocessing...")
    try:
        from data_preprocessing import TextPreprocessor
        
        sample_text = """
        John Doe
        Email: john@example.com | Phone: 123-456-7890
        
        Software Engineer with 5 years experience in Python, JavaScript, and React.
        Expert in machine learning, data structures, and algorithms.
        """
        
        preprocessor = TextPreprocessor()
        processed = preprocessor.preprocess(sample_text)
        
        print(f"Original length: {len(sample_text)}")
        print(f"Processed length: {len(processed)}")
        print(f"Processed text: {processed[:100]}...")
        print("✅ Preprocessing works!")
        return True
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return False


def test_model_loading():
    """Test model loading."""
    print("\nTesting model loading...")
    try:
        from prediction import ResumePredictor
        
        predictor = ResumePredictor('models')
        success = predictor.load_model()
        
        if success:
            print("✅ Model loaded successfully!")
            # Print model info
            print(f"   Algorithm: {predictor.model_trainer.algorithm}")
            print(f"   Vectorizer: {predictor.model_trainer.vectorizer_method}")
            print(f"   Categories: {list(predictor.model_trainer.label_mapping.keys())}")
            return True
        else:
            print("❌ Failed to load model")
            return False
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False


def test_prediction():
    """Test prediction on sample text."""
    print("\nTesting prediction...")
    try:
        from prediction import ResumePredictor
        
        predictor = ResumePredictor('models')
        if not predictor.load_model():
            return False
        
        # Sample resume texts
        test_resumes = {
            "Data Scientist": """
                Data scientist with expertise in Python, machine learning, and deep learning.
                Experience with TensorFlow, PyTorch, and statistical analysis.
                Published research papers on neural networks and computer vision.
            """,
            "Web Developer": """
                Full stack web developer specializing in React, Node.js, and JavaScript.
                Built responsive web applications with HTML5, CSS3, and modern frameworks.
                Experience with REST APIs and database design.
            """,
            "HR": """
                Human resources professional with 5 years experience in recruitment.
                Expert in talent acquisition, employee relations, and performance management.
                Managed full-cycle recruiting and onboarding processes.
            """
        }
        
        print("\nPredictions:")
        for expected_category, text in test_resumes.items():
            result = predictor.predict_category(resume_text=text)
            predicted = result['predicted_category']
            confidence = result['confidence'] * 100
            status = "✅" if predicted == expected_category else "⚠️"
            print(f"  {status} Expected: {expected_category:15s} | Predicted: {predicted:15s} ({confidence:.1f}%)")
        
        print("✅ Prediction works!")
        return True
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_job_matching():
    """Test job description matching."""
    print("\nTesting job description matching...")
    try:
        from prediction import ResumePredictor
        
        predictor = ResumePredictor('models')
        if not predictor.load_model():
            return False
        
        resume_text = """
            Python developer with machine learning experience.
            Skilled in TensorFlow, scikit-learn, and data analysis.
        """
        
        job_desc = """
            We are looking for a Data Scientist with Python and ML experience.
            Must know TensorFlow, machine learning algorithms, and statistics.
        """
        
        processed = predictor.preprocessor.preprocess(resume_text)
        match = predictor.calculate_match_score(processed, job_desc)
        
        print(f"  Match Score: {match['match_score']*100:.1f}%")
        print(f"  Similarity: {match['similarity_score']*100:.1f}%")
        print(f"  Interpretation: {match['interpretation']}")
        
        print("✅ Job matching works!")
        return True
    except Exception as e:
        print(f"❌ Job matching error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Resume Screening System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Text Preprocessing", test_preprocessing),
        ("Model Loading", test_model_loading),
        ("Prediction", test_prediction),
        ("Job Matching", test_job_matching),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run the web app: streamlit run app.py")
        print("  2. Open http://localhost:8501 in your browser")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
