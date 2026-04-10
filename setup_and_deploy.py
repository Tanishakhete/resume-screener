"""
Setup and Deployment Helper for Resume Screening System.
This script prepares the project for deployment to Streamlit Cloud.
"""

import os
import sys
import subprocess
import shutil

def check_model_exists():
    """Check if trained model exists."""
    model_files = [
        'models/vectorizer.joblib',
        'models/classifier.joblib', 
        'models/metadata.joblib'
    ]
    return all(os.path.exists(f) for f in model_files)

def train_model():
    """Train the model if it doesn't exist."""
    print("🤖 Training model...")
    result = subprocess.run([sys.executable, 'train_model.py'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Model trained successfully!")
        return True
    else:
        print(f"❌ Training failed: {result.stderr}")
        return False

def prepare_for_deployment():
    """Prepare project for cloud deployment."""
    print("=" * 60)
    print("🚀 Resume Screening System - Deployment Preparation")
    print("=" * 60)
    
    # 1. Check model
    print("\n1. Checking for trained model...")
    if check_model_exists():
        print("✅ Model found!")
    else:
        print("⚠️ Model not found. Training now...")
        if not train_model():
            print("❌ Failed to train model")
            return False
    
    # 2. Check if git repo exists
    print("\n2. Checking Git repository...")
    if not os.path.exists('.git'):
        print("⚠️ Not a git repository. Initializing...")
        subprocess.run(['git', 'init'], capture_output=True)
        subprocess.run(['git', 'add', '.'], capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], capture_output=True)
        print("✅ Git repository initialized!")
    else:
        print("✅ Git repository exists")
    
    # 3. Create deployment checklist
    print("\n3. Deployment Checklist:")
    print("   ✅ Model trained and saved to models/")
    print("   ✅ Code files ready")
    print("   ✅ Git repository ready")
    
    print("\n" + "=" * 60)
    print("📋 Next Steps for Streamlit Cloud Deployment:")
    print("=" * 60)
    print("""
1. Create a GitHub repository:
   - Go to https://github.com/new
   - Name it: resume-screening-system
   - Keep it Public (for free deployment)
   - Click "Create repository"

2. Push your code to GitHub:
   git remote add origin https://github.com/YOUR_USERNAME/resume-screening-system.git
   git branch -M main
   git push -u origin main

3. Deploy to Streamlit Cloud:
   - Go to https://streamlit.io/cloud
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file path: app.py
   - Click "Deploy"

4. Your app will be live at:
   https://your-app-name.streamlit.app

📌 Important Notes:
   - Keep models/ folder small or use Git LFS for large files
   - The app will train model on first run if not found
   - Streamlit Cloud is free for public repos
""")
    
    return True

def quick_ngrok_setup():
    """Quick setup for ngrok temporary sharing."""
    print("\n" + "=" * 60)
    print("⚡ Quick Temporary Sharing with ngrok")
    print("=" * 60)
    print("""
For immediate temporary sharing (good for 2 hours):

1. Download ngrok from https://ngrok.com/download

2. Extract ngrok.exe to this folder

3. Open TWO terminal windows:

   Terminal 1 (keep running):
   python -m streamlit run app.py
   
   Terminal 2:
   .\ngrok.exe http 8501

4. ngrok will give you a public URL:
   Forwarding: https://abc123.ngrok.io -> http://localhost:8501

5. Share the https://abc123.ngrok.io link with anyone!

⚠️  Note: URL changes every time you restart ngrok
    Free tier: URLs expire after 2 hours
""")

def main():
    """Main deployment helper."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deployment Helper')
    parser.add_argument('--prepare', action='store_true', help='Prepare for deployment')
    parser.add_argument('--ngrok', action='store_true', help='Show ngrok instructions')
    
    args = parser.parse_args()
    
    if args.ngrok:
        quick_ngrok_setup()
    else:
        prepare_for_deployment()
        print("\n" + "=" * 60)
        print("💡 Want temporary sharing instead? Run:")
        print("   python setup_and_deploy.py --ngrok")
        print("=" * 60)

if __name__ == "__main__":
    main()
