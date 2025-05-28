# AI Hiring Bias Mitigation Framework

This project implements a machine learning framework to detect and mitigate gender and racial bias in AI-driven hiring systems using resume and job description data.

It applies:
- Preprocessing techniques (intersectional group encoding and reweighting)
- In-processing via fairness-aware model training
- Evaluation across intersectional groups
- Visualisation of fairness metrics like Selection Rate, Accuracy, and Statistical Disparity

---

## 📌 Features

- Generates intersectional group labels (e.g., Female-Asian)
- TF-IDF vectorisation of resumes and job descriptions
- Three model variations:
  - **Model A** – Resume + Job Description
  - **Model B** – Model A + Gender + Ethnicity
  - **Model C** – Model A + Intersectional Groups + Sample Reweighting
- Fairness evaluations per group
- Visualisations of model performance and fairness outcomes

---

## Requirements

- Python 3.8 or higher
- All required libraries are listed in `requirements.txt`

---

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/clnnhtrn/ai-hiring-bias-framework.git
cd ai-hiring-bias-framework
```

### Step 2 (Optional but Recommended): Create a Virtual Environment

```bash
python -m venv venv
# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs the following Python packages:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `scipy`

---

## Prepare Your Dataset

The dataset used in this framework is the Recruitment Dataset available on Kaggle (https://www.kaggle.com/datasets/surendra365/recruitement-dataset/data)
If you are using your own dataset, your dataset file must be named:

```
job_applicant_dataset.csv
```

It must be placed in the **root directory** of the project (same folder as the Python script).

### Required columns:
- `Resume` – text field
- `Job Description` – text field
- `Gender` – e.g., Male, Female
- `Ethnicity` – e.g., Asian, White, Black
- `Best Match` – binary label (1 = selected, 0 = not selected)

---

## Run the Framework

In the terminal, from the root project folder:

```bash
python main.py
```

---

## Output Files

After running the framework, the following files will be generated automatically:

| File | Description |
|------|-------------|
| `final_preprocessed_dataset.csv` | Fully processed dataset containing original data, TF-IDF features, and one-hot encoded intersectional groups |
| `framework_output_log.txt` | Text log with model metrics, fairness evaluation results, and comparison summaries |
| Matplotlib plots (displayed) | Visual comparisons for selection rate, accuracy, and disparity across groups and models |

---

## Project Structure

```
📁 ai-hiring-bias-framework/
├── job_applicant_dataset.csv         # <-- You provide this dataset
├── main.py                           # Main Python script 
├── requirements.txt                 # Python dependencies
├── README.md                        # Full usage guide (this file)
├── .gitignore                       # Ignored files (e.g., venv, logs)
├── processed_dataset.csv            # Generated processed dataset
├── framework_output_log.txt        # Generated model/fairness evaluation log
```

---

## License

This project is licensed under the **MIT License**.

You are free to use, modify, and share it — with proper attribution.

---

## Need Help?

If you have any issues setting up or running the code, feel free to:

- [Open an issue on GitHub](https://github.com/clnnhtrn/ai-hiring-bias-framework/issues)

---
