# AI Resume Screening System

A complete AI-powered resume screening and classification system built with Python, Machine Learning, and Streamlit.

## 🚀 Features

### Core Features
- **Resume Classification**: Automatically categorize resumes into job roles using ML
- **Text Extraction**: Extract text from PDF and DOCX files
- **NLP Preprocessing**: Clean, tokenize, and preprocess resume text
- **Multiple ML Algorithms**: Support for Naive Bayes, Logistic Regression, SVM, and Random Forest
- **Job Description Matching**: Match candidates to job descriptions using cosine similarity
- **Candidate Ranking**: Rank multiple candidates based on relevance scores

### ML Pipeline
- **Feature Extraction**: TF-IDF and Count Vectorization
- **Model Training**: Automated training with cross-validation
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Probability Scores**: Confidence levels for predictions

### User Interface
- **Streamlit Web App**: Clean, interactive UI
- **Single Resume Analysis**: Analyze one resume at a time
- **Batch Processing**: Process and rank multiple resumes
- **Visual Analytics**: Charts and progress bars for scores

## 📁 Project Structure

```
resume-screening-system/
├── app.py                    # Streamlit web application
├── data_preprocessing.py     # Text extraction and preprocessing
├── model_training.py         # ML model training and evaluation
├── prediction.py             # Resume prediction and matching
├── train_model.py           # Training script with sample data
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── data/                    # Dataset storage
│   └── resume_dataset.csv
└── models/                  # Trained model storage
    ├── vectorizer.joblib
    ├── classifier.joblib
    └── metadata.joblib
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project
```bash
cd resume-screening-system
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
The application will automatically download required NLTK data on first run.

## 🚀 Usage

### Quick Start - Train and Run

1. **Train the Model** (with sample dataset):
```bash
python train_model.py
```

Or with specific options:
```bash
# Train with specific algorithm
python train_model.py --algorithm logistic_regression

# Compare multiple algorithms
python train_model.py --compare

# Use your own dataset
python train_model.py --dataset data/your_dataset.csv
```

2. **Launch the Web App**:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Web Interface

#### Single Resume Analysis
1. Select "📤 Single Resume Analysis" from the sidebar
2. Upload a PDF or DOCX resume
3. (Optional) Enter a job description for matching
4. Click "🔍 Analyze Resume"
5. View the predicted category, confidence score, and match percentage

#### Batch Processing
1. Select "📊 Batch Processing" from the sidebar
2. Upload multiple resumes
3. Enter a job description (required for ranking)
4. Click "🔍 Rank Candidates"
5. View ranked list with match scores

#### Model Performance
1. Select "📈 Model Performance" from the sidebar
2. View evaluation metrics and cross-validation results

## 📊 Dataset

### Using the Sample Dataset
The system includes a sample dataset generator for demonstration. Run:
```bash
python train_model.py
```

### Using Your Own Dataset
Prepare a CSV file with two columns:
- `resume_text`: The resume content (text)
- `category`: The job category/label

Example:
```csv
resume_text,category
"Python machine learning expert...",Data Scientist
"Full stack JavaScript developer...",Web Developer
"HR specialist with 5 years...",HR
```

Train with your dataset:
```bash
python train_model.py --dataset path/to/your/dataset.csv
```

### Downloading from Kaggle
For a real-world dataset, download the [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset):

1. Download and extract the dataset
2. Preprocess to match the required format
3. Train the model:
```bash
python train_model.py --dataset data/Resume.csv
```

## 🧠 Machine Learning Details

### Algorithms Supported
1. **Naive Bayes** - Fast, works well with text
2. **Logistic Regression** - Good baseline, interpretable
3. **SVM (Support Vector Machine)** - Effective for high-dimensional data
4. **Random Forest** - Ensemble method, robust

### Feature Extraction
- **TF-IDF** (Term Frequency-Inverse Document Frequency)
- **Count Vectorizer** (Bag of Words)
- Configurable n-gram range (default: 1-2)
- Maximum 5000 features

### Text Preprocessing Pipeline
1. Convert to lowercase
2. Remove URLs, emails, phone numbers
3. Remove special characters and digits
4. Tokenize text
5. Remove stopwords
6. Lemmatization (reduce words to base form)

### Evaluation Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: Quality of positive predictions
- **Recall**: Coverage of actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Cross-Validation**: 5-fold CV for robustness

## 🔧 API Usage (Programmatic)

### Predict Single Resume
```python
from prediction import quick_predict

result = quick_predict('path/to/resume.pdf')
print(f"Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Match Resume to Job
```python
from prediction import match_resume_to_job

result = match_resume_to_job(
    'path/to/resume.pdf',
    'Job description text here...'
)
print(f"Match Score: {result['match']['match_score']:.2%}")
```

### Rank Multiple Resumes
```python
from prediction import ResumePredictor

predictor = ResumePredictor('models')
predictor.load_model()

resumes = [
    {'name': 'Candidate 1', 'path': 'resume1.pdf'},
    {'name': 'Candidate 2', 'path': 'resume2.pdf'},
]

ranked = predictor.rank_resumes(resumes, "Job description...")
for r in ranked:
    print(f"#{r['rank']}: {r['name']} - {r['match_score']:.2%}")
```

## 📈 Match Score Calculation

The match score combines:
- **Text Similarity** (70%): Cosine similarity between resume and job description
- **Category Match** (30%): Whether predicted category aligns with job

Score Interpretation:
- **80-100%**: Excellent Match - Highly Recommended
- **60-79%**: Good Match - Recommended for Interview
- **40-59%**: Average Match - Consider for Interview
- **20-39%**: Below Average - May Not Be Suitable
- **0-19%**: Poor Match - Not Recommended

## 🔍 Troubleshooting

### Model Not Found Error
If you see "Model not found":
```bash
python train_model.py
```

### NLTK Data Issues
If NLTK data download fails:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### PDF/DOCX Extraction Issues
Ensure you have the required libraries:
```bash
pip install PyPDF2 python-docx
```

## 📝 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- nltk
- spacy
- PyPDF2
- python-docx
- streamlit
- matplotlib
- seaborn
- joblib

See `requirements.txt` for specific versions.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional ML algorithms (XGBoost, Neural Networks)
- Better text extraction (OCR for scanned PDFs)
- Entity extraction (skills, experience, education)
- More sophisticated matching algorithms
- Database integration
- REST API

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Scikit-learn for machine learning tools
- NLTK for natural language processing
- Streamlit for the web interface
- Kaggle for resume datasets

## 📧 Contact

For questions or issues, please open an issue on the project repository.

---

**Built with ❤️ using Python, Scikit-learn, and Streamlit**
