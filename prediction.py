"""
Prediction Module for Resume Screening System.
Handles resume classification and job description matching.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_preprocessing import TextExtractor, TextPreprocessor
from model_training import ModelTrainer


class ResumePredictor:
    """Predict job category from resume and match with job descriptions."""
    
    def __init__(self, model_dir='models'):
        """
        Initialize predictor with trained model.
        
        Args:
            model_dir: Directory containing trained model files
        """
        self.preprocessor = TextPreprocessor()
        self.model_trainer = ModelTrainer()
        self.is_model_loaded = False
        self.model_dir = model_dir
    
    def load_model(self):
        """Load trained model if not already loaded."""
        if not self.is_model_loaded:
            try:
                self.model_trainer.load(self.model_dir)
                self.is_model_loaded = True
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return True
    
    def predict_category(self, resume_path=None, resume_text=None, file_type=None):
        """
        Predict job category from resume.
        
        Args:
            resume_path: Path to resume file (optional if resume_text provided)
            resume_text: Raw resume text (optional if resume_path provided)
            file_type: File type if resume_path is provided
            
        Returns:
            dict: Prediction results
        """
        # Ensure model is loaded
        if not self.load_model():
            return {'error': 'Model not loaded'}
        
        # Extract text if file path provided
        if resume_text is None:
            if resume_path is None:
                return {'error': 'Either resume_path or resume_text must be provided'}
            
            try:
                resume_text = TextExtractor.extract_text(resume_path, file_type)
            except Exception as e:
                return {'error': f'Failed to extract text: {e}'}
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess(resume_text)
        
        if not processed_text.strip():
            return {'error': 'No valid text found in resume'}
        
        # Make prediction
        predictions = self.model_trainer.predict([processed_text])
        
        return {
            'raw_text': resume_text,
            'processed_text': processed_text,
            'predicted_category': predictions[0]['prediction'],
            'confidence': predictions[0]['confidence'],
            'all_probabilities': predictions[0]['all_probabilities']
        }
    
    def calculate_similarity(self, resume_text, job_description):
        """
        Calculate similarity between resume and job description.
        
        Args:
            resume_text: Preprocessed resume text
            job_description: Job description text
            
        Returns:
            float: Similarity score (0-1)
        """
        # Ensure model is loaded
        if not self.load_model():
            return 0.0
        
        # Preprocess job description
        processed_jd = self.preprocessor.preprocess(job_description)
        
        # Get feature vectors
        resume_vector = self.model_trainer.feature_extractor.transform([resume_text])
        jd_vector = self.model_trainer.feature_extractor.transform([processed_jd])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(resume_vector, jd_vector)[0][0]
        
        return similarity
    
    def calculate_match_score(self, resume_text, job_description, 
                              category_weight=0.3, similarity_weight=0.7):
        """
        Calculate comprehensive match score.
        
        Args:
            resume_text: Preprocessed resume text
            job_description: Job description text
            category_weight: Weight for category match
            similarity_weight: Weight for text similarity
            
        Returns:
            dict: Match score details
        """
        # Calculate text similarity
        similarity = self.calculate_similarity(resume_text, job_description)
        
        # Get prediction for resume
        predictions = self.model_trainer.predict([resume_text])
        predicted_category = predictions[0]['prediction']
        category_confidence = predictions[0]['confidence']
        
        # Try to extract job category from description
        jd_lower = job_description.lower()
        job_categories = self.model_trainer.label_mapping.keys()
        
        category_match = 0.0
        for category in job_categories:
            # Check if category keywords are in job description
            category_keywords = category.lower().replace('_', ' ').split()
            matches = sum(1 for keyword in category_keywords if keyword in jd_lower)
            if matches > 0:
                category_match = max(category_match, matches / len(category_keywords))
        
        # Calculate weighted score
        final_score = (category_weight * category_match + 
                      similarity_weight * similarity)
        
        return {
            'match_score': final_score,
            'similarity_score': similarity,
            'category_match': category_match,
            'predicted_category': predicted_category,
            'category_confidence': category_confidence,
            'interpretation': self._interpret_score(final_score)
        }
    
    def _interpret_score(self, score):
        """Interpret match score."""
        if score >= 0.8:
            return "Excellent Match - Highly Recommended"
        elif score >= 0.6:
            return "Good Match - Recommended for Interview"
        elif score >= 0.4:
            return "Average Match - Consider for Interview"
        elif score >= 0.2:
            return "Below Average - May Not Be Suitable"
        else:
            return "Poor Match - Not Recommended"
    
    def rank_resumes(self, resumes_data, job_description):
        """
        Rank multiple resumes based on job description match.
        
        Args:
            resumes_data: List of dicts with 'name', 'path', 'text' keys
            job_description: Job description text
            
        Returns:
            list: Ranked resumes with scores
        """
        results = []
        
        for resume in resumes_data:
            try:
                # Get resume text
                if 'text' in resume and resume['text']:
                    resume_text = resume['text']
                elif 'path' in resume and resume['path']:
                    file_type = resume.get('file_type')
                    resume_text = TextExtractor.extract_text(resume['path'], file_type)
                else:
                    continue
                
                # Preprocess
                processed_text = self.preprocessor.preprocess(resume_text)
                
                if not processed_text.strip():
                    continue
                
                # Calculate match score
                match_result = self.calculate_match_score(processed_text, job_description)
                
                results.append({
                    'name': resume.get('name', 'Unknown'),
                    'path': resume.get('path', ''),
                    'match_score': match_result['match_score'],
                    'similarity_score': match_result['similarity_score'],
                    'predicted_category': match_result['predicted_category'],
                    'interpretation': match_result['interpretation'],
                    'details': match_result
                })
                
            except Exception as e:
                print(f"Error processing {resume.get('name', 'Unknown')}: {e}")
                continue
        
        # Sort by match score (descending)
        results.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Add rank
        for idx, result in enumerate(results, 1):
            result['rank'] = idx
        
        return results
    
    def extract_keywords(self, text, top_n=10):
        """
        Extract important keywords from text using TF-IDF.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            list: Top keywords with scores
        """
        if not self.load_model():
            return []
        
        # Preprocess
        processed_text = self.preprocessor.preprocess(text)
        
        # Transform to features
        feature_vector = self.model_trainer.feature_extractor.transform([processed_text])
        
        # Get feature names
        feature_names = self.model_trainer.feature_extractor.get_feature_names()
        
        # Get scores
        scores = feature_vector.toarray()[0]
        
        # Sort by score
        top_indices = np.argsort(scores)[-top_n:][::-1]
        
        keywords = [
            {'keyword': feature_names[i], 'score': scores[i]}
            for i in top_indices if scores[i] > 0
        ]
        
        return keywords


def quick_predict(resume_path, model_dir='models'):
    """
    Quick prediction function for single resume.
    
    Args:
        resume_path: Path to resume file
        model_dir: Model directory
        
    Returns:
        dict: Prediction results
    """
    predictor = ResumePredictor(model_dir)
    return predictor.predict_category(resume_path=resume_path)


def match_resume_to_job(resume_path, job_description, model_dir='models'):
    """
    Match resume to job description.
    
    Args:
        resume_path: Path to resume file
        job_description: Job description text
        model_dir: Model directory
        
    Returns:
        dict: Match results
    """
    predictor = ResumePredictor(model_dir)
    
    # Get prediction
    prediction = predictor.predict_category(resume_path=resume_path)
    
    if 'error' in prediction:
        return prediction
    
    # Calculate match score
    match_result = predictor.calculate_match_score(
        prediction['processed_text'],
        job_description
    )
    
    return {
        'prediction': prediction,
        'match': match_result
    }


if __name__ == "__main__":
    print("Prediction Module - Testing")
    print("=" * 50)
    
    # Note: This requires a trained model
    print("This module requires a trained model to work.")
    print("Please run model_training.py with your dataset first.")
