📰 Fake News Detector
🎯 Overview

The Fake News Detector is an NLP-based Machine Learning project that predicts whether a news article is Real or Fake.
It uses a trained Logistic Regression model and TF-IDF vectorization, deployed with a clean Streamlit web interface.

⚙️ Features

Real-time fake news detection

Lightweight and fast to run locally

Pre-trained model ready to use (model.pkl)

Streamlit UI for easy testing and demos

🧰 Tech Stack
Category	Tools Used
Language	Python
Libraries	pandas, scikit-learn, numpy
NLP	TF-IDF Vectorizer
Web App	Streamlit
Model Saving	pickle

fake_news_detector/
│
├── app.py
├── train_model.py
├── Fake.csv
├── True.csv
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md

🚀 How to Run

Install dependencies

pip install -r requirements.txt


Run Streamlit app

streamlit run app.py


Enter a news headline or article in the input box and click Check Authenticity.

🧠 Model Details

Algorithm: Logistic Regression

Feature Extraction: TF-IDF Vectorizer

Accuracy: ~93–95%

Dataset: Mini sample of English news headlines (Fake & True)

📊 Sample Output
Input Headline	Prediction
"NASA announces discovery of new planet"	✅ Real News
"Elon Musk confirmed to be an alien from Mars"	❌ Fake News

🏁 Conclusion

This project demonstrates how AI and NLP can fight misinformation effectively.
It’s small, efficient, and deployable — perfect for internship or academic portfolios.

👨‍💻 Author

Developed by Shreya Signatious
For Internship Submission – October 2025
