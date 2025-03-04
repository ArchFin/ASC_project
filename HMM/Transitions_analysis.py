import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the file path
file_path = "/Users/a_fin/Desktop/Year 4/Project/Data/transitions_summary.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Remove rows where Start Time (s) is 0
df = df[df["Start Time (s)"] != 0].copy()

# Define transition labeling function
def classify_transition(row):
    from_state = row["From State"]
    to_state = row["To State"]
    duration = row["Duration (s)"]

    # Example classification logic (customize as needed)
    if from_state == 1 and to_state == 2:
        if duration < 100:
            return "Short 1→2"
        else:
            return "Long 1→2"
    elif from_state == 2 and to_state == 1:
        if duration < 100:
            return "Short 2→1"
        else:
            return "Long 2→1"
    else:
        return "Other"

# Apply the function to create a new column
df["Transition length"] = df.apply(classify_transition, axis=1)

# Save the cleaned file (optional)
df.to_csv("cleaned_transitions.csv", index=False)

# 1. Total count of each Transition Type
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Transition length', order=df['Transition length'].value_counts().index)
plt.title("Total Count of Each Transition Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Data/plot_count.png')

# 2. Interaction between Transition Type and Gradual/Abrupt
cross_tab = pd.crosstab(df['Transition length'], df['Transition type'])
plt.figure(figsize=(10, 6))
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
plt.title("Transition Type vs. Gradual/Abrupt Interaction")
plt.xlabel("Transition Type (Original)")
plt.ylabel("Custom Transition Type")
plt.tight_layout()
plt.savefig('/Users/a_fin/Desktop/Year 4/Project/Data/heatmap.png')


