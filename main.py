# =============================================================================
# Imports
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import matplotlib.pyplot as plt

# =============================================================================
# Log To Output File
# =============================================================================
def log(text="", newline=True, mode='a'):
    with open("framework_output_log.txt", mode, encoding='utf-8') as f:
        f.write(str(text) + ("\n" if newline else ""))


# =============================================================================
# Data Loading and Preparation
# =============================================================================
def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    data['Intersectional_Group'] = data['Gender'] + '-' + data['Ethnicity']
    return data

# =============================================================================
# Feature Engineering
# =============================================================================
########## Text vectorisation ##########
def vectorize_text(data, text_cols):
    vectorizers = {}
    vectorized_parts = []
    for col in text_cols:
        vec = TfidfVectorizer(stop_words='english', max_features=3000)
        X = vec.fit_transform(data[col].fillna(""))
        vectorizers[col] = vec
        vectorized_parts.append(X)
    return hstack(vectorized_parts), vectorizers


########## Encode intersectional group ##########
def encode_intersectional(data):
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    group_encoded = enc.fit_transform(data[['Intersectional_Group']])
    return group_encoded, enc

########## Encode single column ##########
def encode_column(data, column_name):
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore') 
    encoded = enc.fit_transform(data[[column_name]])
    return encoded, enc


# =============================================================================
# Weight Computation
# =============================================================================
########## Compute reweighting based on intersectional group ##########
def compute_intersectional_weights(data):
    A = data['Intersectional_Group']
    Y = data['Best Match']
    N = len(data)
    P_A = A.value_counts(normalize=True).to_dict()
    P_Y = Y.value_counts(normalize=True).to_dict()
    P_AY = data.groupby(['Intersectional_Group', 'Best Match']).size() / N

    weights = []
    for a, y in zip(A, Y):
        pa = P_A[a]
        py = P_Y[y]
        pay = P_AY.loc[(a, y)] if (a, y) in P_AY else 0
        weight = (pa * py) / pay if pay > 0 else 0
        weights.append(weight)

    weights = np.array(weights)
    weights /= np.mean(weights)
    return weights

########## Compute reweighting based on a single attribute (Gender or Ethnicity) ##########
def compute_weights(data, attr):
    A = data[attr]
    Y = data['Best Match']
    N = len(data)
    P_A = A.value_counts(normalize=True).to_dict()
    P_Y = Y.value_counts(normalize=True).to_dict()
    P_AY = data.groupby([attr, 'Best Match']).size() / N

    weights = []
    for a, y in zip(A, Y):
        pa = P_A[a]
        py = P_Y[y]
        pay = P_AY.loc[(a, y)] if (a, y) in P_AY else 0
        weight = (pa * py) / pay if pay > 0 else 0
        weights.append(weight)

    weights = np.array(weights)
    weights /= np.mean(weights)
    return weights

# =============================================================================
# Model Training and Evaluation
# =============================================================================
######### Train and evaluate a model ##########
def train_and_evaluate(X_train, y_train, X_test, y_test, sample_weights=None):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred = model.predict(X_test)
    log(classification_report(y_test, y_pred))
    return y_pred, model

########## Evaluate model per intersectional group ##########
def evaluate_intersectional_fairness(data, y_true, y_pred):
    data['y_true'] = y_true
    data['y_pred'] = y_pred
    metrics = {}
    for group in data['Intersectional_Group'].unique():
        sub = data[data['Intersectional_Group'] == group]
        sr = np.mean(sub['y_pred'])
        acc = np.mean(sub['y_pred'] == sub['y_true'])
        positives = sub[sub['y_true'] == 1]
        metrics[group] = {'Selection Rate': sr, 'Accuracy': acc}
    return pd.DataFrame.from_dict(metrics, orient='index')

# =============================================================================
# Visualisation
# =============================================================================
# =============================================================================
# Additional Visualisation (Moved from run_full_framework)
# =============================================================================
def plot_additional_framework_visuals(fairness_a, fairness_b, fairness_c, fairness_g, fairness_e, data):
    # SR Difference Sorted from Model A
    sr_diff = abs(fairness_c['Selection Rate'] - fairness_a['Selection Rate'])
    sorted_index = sr_diff.sort_values(ascending=False).index
    plt.figure(figsize=(14, 6))
    plt.plot(fairness_a.loc[sorted_index]['Selection Rate'], marker='o', label='Model A')
    plt.plot(fairness_b.loc[sorted_index]['Selection Rate'], marker='o', label='Model B')
    plt.plot(fairness_c.loc[sorted_index]['Selection Rate'], marker='o', label='Model C')
    plt.title("Selection Rate by Intersectional Group (Sorted by difference from Model A)")
    plt.ylabel("Selection Rate")
    plt.xlabel("Intersectional Group")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Statistical Disparity Plot
    def compute_disparity(sr_series):
        median = sr_series.median()
        privileged = sr_series[sr_series > median].mean()
        unprivileged = sr_series[sr_series <= median].mean()
        return privileged, unprivileged

    priv_a, unpriv_a = compute_disparity(fairness_a['Selection Rate'])
    priv_b, unpriv_b = compute_disparity(fairness_b['Selection Rate'])
    priv_c, unpriv_c = compute_disparity(fairness_c['Selection Rate'])

    labels = ['Model A', 'Model B', 'Model C']
    priv_means = [priv_a, priv_b, priv_c]
    unpriv_means = [unpriv_a, unpriv_b, unpriv_c]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, priv_means, width, label='Privileged')
    plt.bar(x + width/2, unpriv_means, width, label='Unprivileged')
    plt.xticks(x, labels)
    plt.title("Statistical Disparity Across Models")
    plt.ylabel("Selection Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # SR Difference with Actual
    actual_sr = data.groupby('Intersectional_Group')['Best Match'].mean()
    sorted_index = sorted(actual_sr.index, key=lambda x: (x.split('-')[0], x.split('-')[1]))

    plt.figure(figsize=(14, 6))
    plt.plot(fairness_a.loc[sorted_index]['Selection Rate'], marker='o', label='Model A')
    plt.plot(fairness_b.loc[sorted_index]['Selection Rate'], marker='o', label='Model B')
    plt.plot(fairness_c.loc[sorted_index]['Selection Rate'], marker='o', label='Model C')
    plt.plot(actual_sr.loc[sorted_index], marker='o', linestyle='--', label='Actual SR')
    plt.title("Selection Rate by Group (Sorted by Ethnicity → Gender)")
    plt.ylabel("Selection Rate")
    plt.xlabel("Intersectional Group")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Side-by-side line plots
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

    axes[0].plot(fairness_a.loc[sorted_index]['Selection Rate'], marker='o', label='Model A')
    axes[0].plot(actual_sr.loc[sorted_index], marker='o', linestyle='--', label='Actual SR')
    axes[0].plot(fairness_c.loc[sorted_index]['Selection Rate'], marker='o', label='Model C (Intersectional)')
    axes[0].plot(fairness_g.loc[sorted_index]['Selection Rate'], marker='o', label='Model C (Gender-only)')
    axes[0].set_title("SR Comparison - Gender-only")
    axes[0].set_xlabel("Intersectional Group")
    axes[0].set_ylabel("Selection Rate")
    axes[0].set_xticks(range(len(sorted_index)))
    axes[0].set_xticklabels(sorted_index, rotation=90)
    axes[0].legend()

    axes[1].plot(fairness_a.loc[sorted_index]['Selection Rate'], marker='o', label='Model A')
    axes[1].plot(actual_sr.loc[sorted_index], marker='o', linestyle='--', label='Actual SR')
    axes[1].plot(fairness_c.loc[sorted_index]['Selection Rate'], marker='o', label='Model C (Intersectional)')
    axes[1].plot(fairness_e.loc[sorted_index]['Selection Rate'], marker='o', label='Model C (Ethnicity-only)')
    axes[1].set_title("SR Comparison - Ethnicity-only")
    axes[1].set_xlabel("Intersectional Group")
    axes[1].set_xticks(range(len(sorted_index)))
    axes[1].set_xticklabels(sorted_index, rotation=90)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


########## Demographic Distribution: Gender and Ethnicity (Bar Chart)##########
def plot_demographic_distribution(data):
    import matplotlib.pyplot as plt

    gender_counts = data['Gender'].value_counts()
    ethnicity_counts = data['Ethnicity'].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gender bar chart (left)
    axes[0].bar(gender_counts.index, gender_counts.values, color='steelblue')
    axes[0].set_title('Gender Distribution')
    axes[0].set_ylabel('Count')
    axes[0].set_xlabel('Gender')

    # Ethnicity bar chart (right)
    axes[1].bar(ethnicity_counts.index, ethnicity_counts.values, color='coral')
    axes[1].set_title('Ethnicity Distribution')
    axes[1].set_xlabel('Ethnicity')
    axes[1].tick_params(axis='x', rotation=90)

    plt.suptitle("Figure 1: Dataset Demographic Distribution – Gender and Ethnicity", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()


########## Plot Statistical Disparity Across Models (Bar Chart) #########
def plot_statistical_disparity(fairness_data, model_label="Model"):
    data = fairness_data.copy()

    if data.index.name == 'Intersectional Group':
        data = data.reset_index()

    data.set_index('Intersectional Group', inplace=True)

    median_sr = data['Selection Rate'].median()
    privileged = data[data['Selection Rate'] > median_sr]
    unprivileged = data[data['Selection Rate'] <= median_sr]

    sr_priv = privileged['Selection Rate'].mean()
    sr_unpriv = unprivileged['Selection Rate'].mean()
    disparity = sr_unpriv - sr_priv

    plt.figure(figsize=(6, 4))
    plt.bar(['Privileged', 'Unprivileged'], [sr_priv, sr_unpriv], color=['#4CAF50', '#2196F3'])
    plt.axhline(y=sr_priv, color='gray', linestyle='--', label=f"Privileged SR = {sr_priv:.2f}")
    plt.axhline(y=sr_unpriv, color='blue', linestyle='--', label=f"Unprivileged SR = {sr_unpriv:.2f}")
    plt.title(f"Statistical Disparity in Selection Rate ({model_label})")
    plt.ylabel("Selection Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    log(f"Statistical Disparity ({model_label}): {disparity:.4f} (Unprivileged - Privileged)")
    return disparity


########## Plot SR/Accuracy by Intersectional Group (Line Chart) ##########
def plot_fairness_comparison(metric_dfs, labels):
    merged = pd.concat(metric_dfs, axis=1, keys=labels)
    for metric in ['Selection Rate', 'Accuracy']:
        plt.figure(figsize=(12, 6))
        for label in labels:
            plt.plot(merged[label][metric], marker='o', label=label)
        plt.title(f"{metric} by Intersectional Group")
        plt.xlabel("Intersectional Group")
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()


########### Plot Selection Rate by Intersectional Group (Ethnicity - Gender) (Line Chart) ###########
def plot_sorted_comparison(fairness_a, fairness_b, fairness_c, sort_key="ethnicity_gender"):
    data = pd.DataFrame({
        'Model A': fairness_a['Selection Rate'],
        'Model B': fairness_b['Selection Rate'],
        'Model C': fairness_c['Selection Rate']
    })

    if sort_key == "gender_ethnicity":
        sorted_index = sorted(data.index, key=lambda x: (x.split('-')[1], x.split('-')[0]))
    elif sort_key == "ethnicity_gender":
        sorted_index = sorted(data.index, key=lambda x: (x.split('-')[0], x.split('-')[1]))
    elif sort_key == "ascending_sr":
        sorted_index = data.mean(axis=1).sort_values().index
    else:
        sorted_index = data.index

    data = data.loc[sorted_index]

    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data['Model A'], marker='o', label='Model A')
    plt.plot(data.index, data['Model B'], marker='o', label='Model B')
    plt.plot(data.index, data['Model C'], marker='o', label='Model C')
    plt.title('Selection Rate by Intersectional Group (Sorted)')
    plt.xlabel('Intersectional Group')
    plt.ylabel('Selection Rate')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

########### Plot Accuracy by Intersectional Group (Ethnicity - Gender) (Line Chart) ###########
def plot_accuracy_sorted_ethnicity_gender(fairness_a, fairness_b, fairness_c):
    # Sort by Ethnicity then Gender
    sorted_index = sorted(fairness_a.index, key=lambda x: (x.split('-')[0], x.split('-')[1]))

    plt.figure(figsize=(14, 6))
    plt.plot(fairness_a.loc[sorted_index]['Accuracy'], marker='o', label='Model A')
    plt.plot(fairness_b.loc[sorted_index]['Accuracy'], marker='o', label='Model B')
    plt.plot(fairness_c.loc[sorted_index]['Accuracy'], marker='o', label='Model C')
    plt.title("Accuracy by Intersectional Group (Sorted by Ethnicity → Gender)")
    plt.ylabel("Accuracy")
    plt.xlabel("Intersectional Group")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

########## Summary Table for Logging ##########
def summarise_fairness(fairness_dfs, labels):
    summary = []
    for df, label in zip(fairness_dfs, labels):
        avg_sr = df['Selection Rate'].mean()
        avg_acc = df['Accuracy'].mean()
        disparity = df['Selection Rate'].median() - avg_sr
        summary.append({
            'Model': label,
            'Avg Selection Rate': avg_sr,
            'Avg Accuracy': avg_acc,
            'Statistical Disparity': disparity
        })
    return pd.DataFrame(summary)



# =============================================================================
# Main Framework
# =============================================================================
def run_full_framework(filepath):
    # Load and prepare data
    data = load_and_prepare_data(filepath)
    log("========== FRAMEWORK LOG START ==========", mode='w')
    log(data['Gender'].value_counts())
    log("\n")
    log(data['Ethnicity'].value_counts())
    log("\n")
    log(data['Intersectional_Group'].value_counts())

    # Initial target variable
    y = data['Best Match']

    # Text vectorisation
    X_text, _ = vectorize_text(data, ['Resume', 'Job Description'])

    # ------------------------------------------------------------
    # Model A: Resume + Job Description
    # ------------------------------------------------------------
    log("\n--- MODEL A: Resume + Job Description Only ---")
    X_train_a, X_test_a, y_train, y_test = train_test_split(X_text, y, stratify=y, random_state=42)
    y_pred_a, _ = train_and_evaluate(X_train_a, y_train, X_test_a, y_test)
    data_a_test = data.iloc[y_test.index].copy()
    fairness_a = evaluate_intersectional_fairness(data_a_test, y_test, y_pred_a)
    log(f"\nIntersectional Fairness for Model A:\n{fairness_a.to_string()}")

    # ------------------------------------------------------------
    # Model B: Resume + Job Description + Gender + Ethnicity
    # ------------------------------------------------------------
    log("\n--- MODEL B: Model A + Gender + Ethnicity ---")
    gender_encoded, _ = encode_column(data, 'Gender')
    ethnicity_encoded, _ = encode_column(data, 'Ethnicity')
    X_b = hstack([X_text, gender_encoded, ethnicity_encoded])
    X_train_b, X_test_b, _, _ = train_test_split(X_b, y, stratify=y, random_state=42)
    y_pred_b, _ = train_and_evaluate(X_train_b, y_train, X_test_b, y_test)
    data_b_test = data.iloc[y_test.index].copy()
    fairness_b = evaluate_intersectional_fairness(data_b_test, y_test, y_pred_b)
    log(f"\nIntersectional Fairness for Model B:\n{fairness_b.to_string()}")

    # ------------------------------------------------------------
    # Model C: Resume + Job Description + Intersectional Reweighting
    # ------------------------------------------------------------
    log("\n--- MODEL C: Model A + Intersectional Reweighting ---")
    group_encoded_c, _ = encode_intersectional(data)
    X_c = hstack([X_text, group_encoded_c])
    weights = compute_intersectional_weights(data)
    X_train_c, X_test_c, _, _ = train_test_split(X_c, y, stratify=y, random_state=42)
    y_pred_c, _ = train_and_evaluate(X_train_c, y_train, X_test_c, y_test, sample_weights=weights[y_train.index])
    data_c_test = data.iloc[y_test.index].copy()
    fairness_c = evaluate_intersectional_fairness(data_c_test, y_test, y_pred_c)
    log("\nIntersectional Fairness After Mitigation:")
    log(fairness_c)

    # Save the fully processed dataset
    encoded_df = pd.DataFrame(group_encoded_c, columns=[f"group_{i}" for i in range(group_encoded_c.shape[1])])
    final_df = pd.concat([data.reset_index(drop=True), encoded_df], axis=1)
    final_df.to_csv("final_preprocessed_dataset.csv", index=False)

    # ------------------------------------------------------------
    # Model C (Gender-only Reweighting)
    # ------------------------------------------------------------
    log("\n--- MODEL C (Gender-only Reweighting) ---")
    gender_weights = compute_weights(data, 'Gender')
    X_train_g, X_test_g, _, _ = train_test_split(X_c, y, stratify=y, random_state=42)
    y_pred_g, _ = train_and_evaluate(X_train_g, y_train, X_test_g, y_test, gender_weights[y_train.index])
    data_g_test = data.iloc[y_test.index].copy()
    fairness_g = evaluate_intersectional_fairness(data_g_test, y_test, y_pred_g)
    log(fairness_g[['Selection Rate', 'Accuracy']].to_string())

    # ------------------------------------------------------------
    # Model C (Ethnicity-only Reweighting)
    # ------------------------------------------------------------
    log("\n--- MODEL C (Ethnicity-only Reweighting) ---")
    eth_weights = compute_weights(data, 'Ethnicity')
    X_train_e, X_test_e, _, _ = train_test_split(X_c, y, stratify=y, random_state=42)
    y_pred_e, _ = train_and_evaluate(X_train_e, y_train, X_test_e, y_test, eth_weights[y_train.index])
    data_e_test = data.iloc[y_test.index].copy()
    fairness_e = evaluate_intersectional_fairness(data_e_test, y_test, y_pred_e)
    log(fairness_e[['Selection Rate', 'Accuracy']].to_string())

    # ------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------
    
    plot_fairness_comparison([fairness_a, fairness_b, fairness_c], ["Model A", "Model B", "Model C"])
    plot_sorted_comparison(fairness_a, fairness_b, fairness_c, sort_key="ethnicity_gender")
    plot_accuracy_sorted_ethnicity_gender(fairness_a, fairness_b, fairness_c)
    plot_additional_framework_visuals(fairness_a, fairness_b, fairness_c, fairness_g, fairness_e, data)

    # ------------------------------------------------------------
    # Summary Logging
    # ------------------------------------------------------------
    summary_df = summarise_fairness(
        [fairness_a, fairness_b, fairness_c],
        ['Model A', 'Model B', 'Model C']
    )
    log("\nAverage Fairness Summary Across Models:")
    log(summary_df.to_string())

# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    filepath = "job_applicant_dataset.csv"
    run_full_framework(filepath)
