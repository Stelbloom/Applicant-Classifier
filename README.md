# Applicant Role‑Fit Classifier

> **From “spray and pray” job hunting to targeted, data‑driven applications.**

A machine learning project born from a simple but painful experience:  
*Applying to dozens of jobs and having no idea which ones I actually stood a chance at.*

This project builds a **multi‑class classification model** that takes an applicant’s profile (experience, skills, education, certifications, soft skills, and more) and predicts one of three outcomes:

- **Best Suited for Role** – a strong, high‑confidence fit  
- **Best Match for Role** – a solid fit worth pursuing  
- **Will Get Interview** – meets the bar for a first conversation, but may need more to go further  

The goal isn’t to discourage applications. It’s to help **applicants and recruiters prioritise where time and effort go.**

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)  
2. [Business Problem](#business-problem)  
3. [Project Objectives](#project-objectives)  
4. [Data Source](#data-source)  
5. [Key Features](#key-features)  
6. [Tools and Libraries](#tools-and-libraries)  
7. [Data Preprocessing](#data-preprocessing)  
8. [Exploratory Data Analysis](#exploratory-data-analysis)  
9. [Machine Learning Model](#machine-learning-model)  
10. [Evaluation Metrics](#evaluation-metrics)  
11. [Key Insights](#key-insights)  
12. [Recommendations & Use Cases](#recommendations--use-cases)  
13. [Getting Started](#getting-started)  
14. [Disclaimer](#disclaimer)  
15. [Conclusion](#conclusion)  

---

## 🔍 Project Overview

**Project title:** Applicant Role‑Fit Classifier  

I started this project as a **job applicant**, not as a data scientist hunting for a portfolio piece.

After spending hours tailoring applications for roles I’d never hear back from – and occasionally landing interviews for jobs I almost skipped – I asked:

> **What if I could predict, *before applying*, whether I’m actually a strong fit for a role?**

That question turned into this project: a **multi‑class classifier** that predicts one of three outcomes for a given applicant–role combination:

1. **Best Suited for Role**  
2. **Best Match for Role**  
3. **Will Get Interview**  

The model is designed as a **decision‑support tool** for:

- Applicants who want to prioritise their applications  
- Recruiters who need quick, explainable triage  
- Anyone interested in making the hiring funnel more efficient and transparent  

---

## 🧩 Business Problem

Job search today often feels like **spray‑and‑pray**:

- Applicants send dozens of applications with **little feedback** on fit  
- Recruiters are overwhelmed with CVs, many of which are **poor matches**  
- Time is wasted on both sides:  
  - Applicants tailoring CVs for roles they’re unlikely to get  
  - Recruiters screening profiles that should have been filtered earlier  

There’s a gap between **how roles are described** and **how well candidates fit those roles**, and applicants rarely have tools to see that clearly.

**Core problem:**  
> *Given an applicant’s profile and a role’s requirements, can we estimate how strong a fit this applicant is likely to be?*

Not to replace humans, but to **prioritise effort and attention**.

---

## 🎯 Project Objectives

The primary objectives of this project are to:

1. **Frame the problem** as a supervised, multi‑class classification task  
2. **Design a feature set** that captures an applicant’s real‑world profile, including:
   - Experience level  
   - Education and certifications  
   - Skill match and project history  
   - Soft skills (communication, leadership, culture fit)  
   - Salary history and location match  
   - Interview track record  
3. **Handle realistic class imbalance**, reflecting actual hiring funnels  
4. **Evaluate and compare** multiple classification algorithms  
5. **Build a production‑ready pipeline** for training and serving predictions via an API  
6. **Expose clear outputs** (“Best Suited”, “Best Match”, “Will Get Interview”) plus confidence scores for each class  

Target users:

- **Job applicants** who want to focus on realistic opportunities  
- **Recruiters / hiring teams** who need faster, more structured candidate triage  
- **Hiring tooling builders** experimenting with fit‑scoring and recommendation systems  

---

## 📂 Data Source

The project uses two core datasets:

- `applicant_dataset_1000.csv`  
  - **Balanced synthetic dataset**  
  - 1,000 records, roughly **~333 per class**  
  - Used for baseline modelling and clean evaluation  

- `applicant_dataset_imbalanced.csv`  
  - **Realistic imbalanced dataset**  
  - Approximate class split: **550 / 300 / 150**  
  - Reflects a typical hiring funnel where:
    - Many applicants are “interview‑worthy”  
    - Fewer are “best match”  
    - Very few are “best suited”  

All datasets are **simulated** to avoid privacy issues but are structured to imitate realistic applicant–role relationships.

---

## 🔑 Key Features

The model is trained on **12 core features** describing the applicant and their alignment to the role, including:

- **Experience & history**
  - Years of professional experience  
  - Number of relevant projects completed  
  - Past interview success rate / track record  

- **Education & credentials**
  - Highest education level (e.g. Bachelor, Master, PhD, bootcamp…)  
  - Professional certifications  

- **Skill & role match**
  - Skill match score (how well applicant skills map to role requirements)  
  - Domain match (e.g. data science, web dev, product)  

- **Soft skills & culture**
  - Communication score  
  - Leadership score  
  - Culture fit score  

- **Contextual factors**
  - Location match score / remote compatibility  
  - Salary alignment score or band match  

- **Target label**
  - `outcome_class` with 3 categories:
    - `Best Suited for Role`  
    - `Best Match for Role`  
    - `Will Get Interview`  

> In some experiments, text‑based features (e.g. free‑text skills, summary) are vectorised using **TF‑IDF** to enrich the feature space.

---

## 🛠 Tools and Libraries

- **Python**  
- **pandas**, **NumPy** – data handling and preprocessing  
- **scikit‑learn** – modelling, pipelines, hyperparameter tuning  
- **imbalanced‑learn (SMOTE)** – handling class imbalance  
- **Matplotlib**, **Seaborn** – EDA and visualisation  
- **joblib** – model and encoder persistence (`model.pkl`, `le_edu.pkl`)  
- **Flask** – simple REST API to serve predictions  

---

## 🧹 Data Preprocessing

_Subtitle: Turning raw applicant profiles into model‑ready features_

Key preprocessing steps implemented in `applicant_classifier.py`:

1. **Data loading & validation**
   - Load either balanced or imbalanced dataset  
   - Check for missing or inconsistent values  

2. **Feature engineering**
   - Construct aggregate scores (e.g. overall fit score) where useful  
   - Optionally compute TF‑IDF vectors for text‑based fields (e.g. skills text, profile summary)  

3. **Encoding & scaling**
   - **Label encoding** for `education` using `le_edu.pkl`  
   - One‑hot or ordinal encoding for categorical features where appropriate  
   - Standardisation / normalisation of numeric features when needed  

4. **Handling class imbalance**
   - For imbalanced runs:
     - **SMOTE** for synthetic minority oversampling  
     - **Class‑weight balancing** in algorithms that support it  
   - Always keeping the **test set untouched** to reflect real‑world class proportions  

5. **Train–test split**
   - Stratified splitting to maintain class proportions across train and test sets  

---

## 📊 Exploratory Data Analysis

_Subtitle: Understanding what “fit” looks like in the data_

EDA focuses on:

- Class distribution:
  - Proportion of `Best Suited`, `Best Match`, `Will Get Interview`  
  - Comparison between **balanced** and **imbalanced** datasets  

- Feature distributions:
  - Experience by class  
  - Education levels across outcomes  
  - Skill match scores vs. final outcome  
  - Soft skill scores and their relationship with “Best Suited” predictions  

- Correlations:
  - Which features show the strongest correlation with outcome class  
  - How combinations (e.g. high skill match + strong communication) shift probabilities  

**Suggested visuals for the README (if you export plots):**

```markdown
![Class Distribution](reports/figures/class_distribution.png)
*Distribution of outcome classes: Best Suited, Best Match, Will Get Interview.*

![Feature Importance](reports/figures/feature_importance.png)
*Top contributing features to the Random Forest classifier.*

![ROC Curves by Class](reports/figures/roc_by_class.png)
*One-vs-rest ROC curves showing strong separability across the three classes.*
```

---

## 🤖 Machine Learning Model

_Subtitle: Multi‑class prediction for applicant role fit_

### Approach: Supervised, multi‑class classification

This is a **supervised** learning problem:

- Every training record has a **known outcome label**  
- The model learns the relationship between **applicant profile features** and the **final fit category**  

It is **multi‑class** (not binary) because there are three mutually exclusive outcomes:

- `Best Suited for Role`  
- `Best Match for Role`  
- `Will Get Interview`  

These are handled as a **single target column** rather than three separate binary models.

### Algorithms evaluated

Several tree‑based models were compared:

- Decision Tree  
- Gradient Boosting  
- **Random Forest Classifier** (final choice)  

**Final model:**  
- **Random Forest Classifier**  
  - **Macro‑average AUC:** ~**0.978**  
  - **Test accuracy:** ~**90%**  
  - Errors tend to be *near misses* (e.g. “Best Match” classified as “Best Suited”), which is ideal for a prioritisation tool.

### Hyperparameter tuning

- **GridSearchCV** used to tune:
  - Number of trees (`n_estimators`)  
  - Maximum depth (`max_depth`)  
  - Minimum samples per split/leaf, etc.  

The best configuration is then persisted to `model.pkl` for reuse.

---

## 📏 Evaluation Metrics

To properly handle the **class imbalance** and the **multi‑class nature** of the problem, the following metrics are used:

- **Accuracy**  
  Overall proportion of correctly classified applicants across all three classes.

- **Precision (per class)**  
  For each outcome (e.g. “Best Suited”), how many predicted as that class truly belonged there?

- **Recall (per class)**  
  For each outcome, how many of the true instances of that class were correctly detected?

- **F1‑Score (per class + macro average)**  
  Harmonic mean of precision and recall; macro average treats all classes equally.

- **ROC‑AUC (macro)**  
  One‑vs‑rest ROC curves are used to compute a **macro‑average AUC**.  
  The final Random Forest achieved **~0.978 macro‑AUC**, indicating strong separability between classes.

Where relevant, **confusion matrices** are used to inspect *which* misclassifications happen (e.g. “Best Match” vs. “Best Suited”, rather than completely wrong predictions).

---

## 🔑 Key Insights

Some key takeaways from the modelling and analysis:

1. **High overall performance with balanced trade‑offs**  
   - ~90% test accuracy and macro‑AUC of ~0.978 show that the model reliably distinguishes between **Best Suited**, **Best Match**, and **Will Get Interview**.

2. **Imbalance handling matters**  
   - On the imbalanced dataset, **SMOTE** and **class weights** significantly improve recall for the rare “Best Suited” class without collapsing overall precision.

3. **Feature contributions make intuitive sense**  
   - Features like **experience**, **skill match score**, **projects completed**, and **soft skills (communication, leadership, culture fit)** contribute strongly to predicting “Best Suited for Role”.  
   - Location and salary alignment play a bigger role in “Will Get Interview” vs “Best Match” separations.

4. **Errors are usually “soft” misclassifications**  
   - Most misclassifications happen between **adjacent classes** (e.g. “Best Match” vs “Best Suited”), which is acceptable for a tool that’s meant to guide prioritisation rather than make hard yes/no decisions.

---

## 💡 Recommendations & Use Cases

_Subtitle: How this could be used in the real world_

1. **Applicant personal assistant**  
   - Use this model locally to score roles against your profile and **prioritise applications** where you’re “Best Suited” or “Best Match”.

2. **Browser extension for job boards**  
   - A simple extension could parse job postings, map them to model features, and show:
     - “Role Fit: Best Suited / Best Match / Will Get Interview”  
     - Confidence bars for each class  

3. **ATS (Applicant Tracking System) integration**  
   - Recruiters could use this as a **first‑pass triage tool** to:
     - Sort candidates by projected fit  
     - Identify “hidden gems” who are strong matches but might otherwise be overlooked  

4. **Career coaching dashboards**  
   - Track which types of roles consistently yield **“Best Suited”** predictions.  
   - Help applicants adjust their **skill development and positioning** to move more roles into that category over time.

> Important: In all cases, the model should be a **decision‑support tool**, not the final gatekeeper.

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib flask imbalanced-learn
```

### 2. Train the model

```bash
python applicant_classifier.py
```

This will:

- Load the dataset  
- Preprocess and engineer features  
- Train the Random Forest classifier  
- Save the trained pipeline to `model.pkl`  
- Save the education label encoder to `le_edu.pkl`  

### 3. Serve predictions via API

```bash
python applicant_classifier.py --serve
```

Once running, send a **POST** request to `/predict` with an applicant profile:

```json
{
  "years_experience": 5,
  "education": "Masters",
  "skill_match_score": 0.82,
  "projects_completed": 6,
  "certifications": 2,
  "communication_score": 4.5,
  "leadership_score": 4.0,
  "culture_fit_score": 4.2,
  "location_match": 1,
  "salary_alignment": 0.9,
  "past_interview_success": 0.7,
  "domain_match": "Data Science"
}
```

The API responds with:

- **Predicted class**: e.g. `"Best Match for Role"`  
- **Confidence scores** for all three classes  

---

## ⚖️ Disclaimer

This model is trained on **simulated data** (`applicant_dataset_1000.csv`, `applicant_dataset_imbalanced.csv`) and is intended as a **decision‑support tool**, not a replacement for human judgement in hiring or job searching.

Real‑world deployment would require:

- Training on **real, anonymised, ethically sourced** data  
- **Bias and fairness auditing** (e.g. across gender, ethnicity, age)  
- Clear **governance and transparency** about how predictions are used  

---

## ✅ Conclusion

The **Applicant Role‑Fit Classifier** is a concrete demonstration of how machine learning can:

- Help applicants **prioritise** where they invest their time  
- Help recruiters **triage** large candidate pools more effectively  
- Turn vague notions of “fit” into a more **structured, explainable signal**  

From a technical perspective, this project shows:

- Problem framing as **supervised multi‑class classification**  
- Handling of **imbalanced data** with SMOTE and class weights  
- Use of **Random Forests**, TF‑IDF, and GridSearchCV in a reproducible pipeline  
- Exposure of the model via a simple **Flask API** for real use  

If you’re an applicant tired of the spray‑and‑pray approach, feel free to **fork this repo**, plug in your own profile data, and start seeing where you actually stand a chance.

