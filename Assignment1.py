import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import wilcoxon
import pulp

np.random.seed(42)
n_patients = 200
data = pd.DataFrame({
    "patient_id": np.arange(n_patients),
    "pain_baseline": np.random.randint(0, 10, size=n_patients),
    "urgency_baseline": np.random.randint(0, 10, size=n_patients),
    "frequency_baseline": np.random.randint(0, 10, size=n_patients),
    "pain_before_treatment": np.random.randint(0, 10, size=n_patients),
    "urgency_before_treatment": np.random.randint(0, 10, size=n_patients),
    "frequency_before_treatment": np.random.randint(0, 10, size=n_patients),
    "treated": np.random.choice([0, 1], size=n_patients)
})

treated_patients = data[data["treated"] == 1].reset_index(drop=True)
control_patients = data[data["treated"] == 0].reset_index(drop=True)

cov_matrix = data.iloc[:, 1:7].cov().values
inv_cov_matrix = np.linalg.inv(cov_matrix)

def compute_mahalanobis(p1, p2):
    diff = p1 - p2
    return np.sqrt(diff.T @ inv_cov_matrix @ diff)

cost_matrix = np.zeros((len(treated_patients), len(control_patients)))

for i, t_row in treated_patients.iterrows():
    for j, c_row in control_patients.iterrows():
        cost_matrix[i, j] = compute_mahalanobis(
            t_row.iloc[1:7].values, c_row.iloc[1:7].values
        )

problem = pulp.LpProblem("Balanced_Matching", pulp.LpMinimize)
x = pulp.LpVariable.dicts("Pair", [(i, j) for i in range(len(treated_patients)) for j in range(len(control_patients))],
                          cat=pulp.LpBinary)

problem += pulp.lpSum(x[i, j] * cost_matrix[i, j] for i in range(len(treated_patients)) for j in range(len(control_patients)))

for i in range(len(treated_patients)):
    problem += pulp.lpSum(x[i, j] for j in range(len(control_patients))) == 1

for j in range(len(control_patients)):
    problem += pulp.lpSum(x[i, j] for i in range(len(treated_patients))) <= 1

problem.solve()

matched_pairs = []
for i in range(len(treated_patients)):
    for j in range(len(control_patients)):
        if pulp.value(x[i, j]) == 1:
            matched_pairs.append((treated_patients.iloc[i]["patient_id"], control_patients.iloc[j]["patient_id"]))

matched_pairs_df = pd.DataFrame(matched_pairs, columns=["treated_patient", "control_patient"])
print("Matched Pairs:\n", matched_pairs_df)

treatment_effects = []
for treated_id, control_id in matched_pairs:
    treated_outcome = treated_patients[treated_patients["patient_id"] == treated_id][["pain_before_treatment", "urgency_before_treatment", "frequency_before_treatment"]].values.flatten()
    control_outcome = control_patients[control_patients["patient_id"] == control_id][["pain_before_treatment", "urgency_before_treatment", "frequency_before_treatment"]].values.flatten()
    treatment_effects.append(treated_outcome - control_outcome)

treatment_effects = np.array(treatment_effects)

pain_test = wilcoxon(treatment_effects[:, 0])
urgency_test = wilcoxon(treatment_effects[:, 1])
frequency_test = wilcoxon(treatment_effects[:, 2])

print("\nWilcoxon Test Results:")
print(f"Pain: p-value = {pain_test.pvalue}")
print(f"Urgency: p-value = {urgency_test.pvalue}")
print(f"Frequency: p-value = {frequency_test.pvalue}")
