# Applicant Role-Fit Classifier

> A machine learning project born from a cheer determination: Applying to dozens of jobs and having no idea which ones I actually stood a chance at.

## Why I built this

I started this project as a job applicant, not as a data scientist looking for a portfolio piece. I was tired of spending hours tailoring applications for roles I'd never hear back from, while occasionally landing interviews for jobs I almost didn't apply to.

So I asked a simple question: **what if I could predict, before applying, whether I was actually a strong fit for a role?**

That question turned into this project — a multi-class classification model that takes an applicant's profile (experience, skills, education, certifications, soft-skill scores, and more) and predicts one of three outcomes:

- **Best Suited for Role** — a strong, high-confidence fit
- **Best Match for Role** — a solid fit worth pursuing
- **Will Get Interview** — meets the bar for a first conversation, but may need more to go further

The goal isn't to discourage anyone from applying. It's to help applicants (and recruiters) prioritize where time and effort go.

## What's inside

| File | Description |
|---|---|
| `applicant_classifier.py` | Full pipeline — data loading, preprocessing, training, evaluation, and a Flask API for serving predictions |
| `applicant_dataset_1000.csv` | Balanced synthetic dataset (1,000 records, ~333 per class) |
| `applicant_dataset_imbalanced.csv` | Realistic imbalanced dataset (550 / 300 / 150 split) reflecting a typical hiring funnel |
| `model.pkl` | Trained Random Forest pipeline (generated after running the script) |
| `le_edu.pkl` | Label encoder for the education feature |

## How it works

The model is trained on 12 features covering experience, education, skills match, project history, certifications, soft skills (communication, leadership, culture fit), salary history, location match, and interview track record.

**Approach: Supervised, multi-class classification**

This is a supervised learning problem because every record in the training data has a known outcome label — the model learns the relationship between an applicant's profile and the eventual result. It's multi-class (not binary) because there are three possible outcomes, and it's handled as a single target column rather than three separate models.

**Model: Random Forest Classifier**

After comparing Random Forest, Gradient Boosting, and Decision Tree, Random Forest came out on top with a macro-average AUC of 0.978 and 90% test accuracy — strong performance across all three categories, with prediction errors that stay close (a "Best Match" misclassified as "Best Suited" rather than something wildly off).

**Handling real-world imbalance**

Real hiring data isn't evenly distributed — most applicants clear the interview bar, fewer are a strong match, and very few are the best possible fit. This project includes an imbalanced dataset and demonstrates two standard techniques for handling it: SMOTE (synthetic oversampling) and class-weight balancing, both applied correctly so the test set always reflects reality.

Feature Engineering: Using TF-IDF Vectorizer.

Model Evaluation: Evaluated using Precision, Recall, and F1-Score to handle class imbalances in applicants data. 

Hyperparameter Tuning: GridSearchCV 

## Getting started

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib flask imbalanced-learn

# Train the model
python applicant_classifier.py

# Serve predictions via API
python applicant_classifier.py --serve
```

Once running, send a POST request to `/predict` with an applicant's profile and get back a predicted category plus confidence scores for all three classes.

## What's next

This started as a personal tool, but the same approach could plug into an applicant tracking system, a browser extension that scores job postings as you browse, or a personal dashboard that tracks which types of roles you're consistently strong matches for over time.

If you're an applicant who's also tired of the spray-and-pray approach to job hunting — feel free to fork this, plug in your own profile data, and see where you actually stand a chance.

## Disclaimer

This model is trained on simulated data: Applicant Dataset and is intended as a decision-support tool, not a replacement for human judgment in hiring or job searching. Real-world deployment would require training on real, anonymized, and ethically sourced data, along with careful bias auditing.
