import numpy as np
import pandas as pd

def generate_pupillometry_data(n=5000, seed=42):
    np.random.seed(seed)

    # -----------------------------
    # Patient identifiers
    # -----------------------------
    patient_id = np.arange(1, n + 1)

    site_id = np.random.choice(
        ["ICU_1", "ICU_2", "ICU_3"],
        size=n,
        p=[0.40, 0.35, 0.25]
    )

    # -----------------------------
    # Diagnosis groups
    # -----------------------------
    diagnosis = np.random.choice(
        ["TBI", "Stroke", "Other"],
        size=n,
        p=[0.20, 0.30, 0.50]
    )

    # -----------------------------
    # Demographics
    # -----------------------------
    sex = np.random.choice(["Male", "Female"], size=n, p=[0.5, 0.5])

    age = np.zeros(n)
    for i in range(n):
        if diagnosis[i] == "TBI":
            age[i] = np.random.normal(30, 8)   # 18–45 typical
        elif diagnosis[i] == "Stroke":
            age[i] = np.random.normal(68, 10)  # 55–85 typical
        else:
            age[i] = np.random.normal(50, 20)  # broad hospital population
    age = np.clip(age, 18, 90)

    # -----------------------------
    # Pupillometry features
    # -----------------------------
    # NPI varies by diagnosis
    npi = np.zeros(n)
    for i in range(n):
        if diagnosis[i] == "TBI":
            npi[i] = np.random.normal(3.2, 0.7)
        elif diagnosis[i] == "Stroke":
            npi[i] = np.random.normal(3.5, 0.6)
        else:
            npi[i] = np.random.normal(4.0, 0.5)
    npi = np.clip(npi, 0.5, 5.0)

    # True pupil size (baseline physiology)
    true_pupil_size = np.clip(np.random.normal(3.2, 0.5, n), 1.0, 7.0)

    # Measured pupil size (adds device noise)
    measured_pupil_size = true_pupil_size + np.random.normal(0, 0.2, n)
    measured_pupil_size = np.clip(measured_pupil_size, 1.0, 7.0)

    # Left/right pupils (independent variation)
    pupil_left = np.clip(np.random.normal(3.2, 0.7, n), 1.0, 7.0)
    pupil_right = np.clip(np.random.normal(3.2, 0.7, n), 1.0, 7.0)

    # Velocities + latency
    constriction_velocity = np.clip(np.random.normal(1.8, 0.4, n), 0.2, 4.0)
    dilation_velocity = np.clip(np.random.normal(0.9, 0.3, n), 0.1, 3.0)
    latency_ms = np.clip(np.random.normal(230, 40, n), 120, 500)

    # -----------------------------
    # GCS (corrected probabilities)
    # -----------------------------
    gcs = np.random.choice(
        np.arange(3, 16),
        size=n,
        p=[
            0.02, 0.02, 0.02,   # 3,4,5 severe
            0.05, 0.05, 0.05,   # 6,7,8 moderate
            0.08, 0.08, 0.08,   # 9,10,11 mild
            0.10, 0.10, 0.10,   # 12,13,14 normal
            0.25                # 15 normal
        ]
    )

    # Binary severe label
    gcs_severe = (gcs <= 8).astype(int)

    # -----------------------------
    # Severity label (multi-class)
    # -----------------------------
    severity = []
    for i in range(n):
        if diagnosis[i] == "TBI" and (npi[i] < 3.2 or gcs[i] <= 8):
            severity.append("severe")
        elif diagnosis[i] == "Stroke" and (npi[i] < 3.4 or gcs[i] <= 10):
            severity.append("moderate")
        else:
            if npi[i] < 3.0 or gcs[i] <= 7:
                severity.append("severe")
            elif 3.0 <= npi[i] < 3.5 or 8 <= gcs[i] <= 10:
                severity.append("moderate")
            else:
                severity.append("mild")

    # -----------------------------
    # Final DataFrame
    # -----------------------------
    df = pd.DataFrame({
        "patient_id": patient_id,
        "site_id": site_id,
        "age": age,
        "sex": sex,
        "diagnosis": diagnosis,
        "npi": npi,
        "true_pupil_size": true_pupil_size,
        "measured_pupil_size": measured_pupil_size,
        "pupil_left": pupil_left,
        "pupil_right": pupil_right,
        "constriction_velocity": constriction_velocity,
        "dilation_velocity": dilation_velocity,
        "latency_ms": latency_ms,
        "gcs": gcs,
        "gcs_severe": gcs_severe,
        "severity": severity
    })

    return df


def save_dataset(path="data/synthetic_pupillometry.csv", n=5000):
    df = generate_pupillometry_data(n=n)
    df.to_csv(path, index=False)
    print(f"Saved synthetic dataset to {path}")


if __name__ == "__main__":
    save_dataset()