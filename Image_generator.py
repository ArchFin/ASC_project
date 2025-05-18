import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml

# Load config
yaml_path = os.path.join(os.path.dirname(__file__), 'Breathwork.yaml')
with open(yaml_path, 'r') as file:
    config = yaml.safe_load(file)

# Load data
csv_path = config['filelocation_TET']
df = pd.read_csv(csv_path)
feelings = config['feelings']
savelocation = config['savelocation_TET']
colours = config.get('colours', {})

has_week = 'Week' in df.columns

def annotate_conditions(ax, time_array, df):
    if 'Condition' not in df.columns:
        return
    prev_condition = df['Condition'].iloc[0]
    start_time = time_array[0]
    for idx, (t, condition) in enumerate(zip(time_array, df['Condition'])):
        if condition != prev_condition or idx == len(time_array)-1:
            end_time = t
            ax.text((start_time + end_time)/2, ax.get_ylim()[1]*0.95,
                    prev_condition, ha='center', va='top', 
                    fontsize=6, color='black')
            start_time = t
            prev_condition = condition

time_jump = 28  # seconds between samples

# Specify the subject, week, and session you want to plot
subject_id = f"s17\\"  # Change as needed
week_id = 'week_3'         # Change as needed (if applicable)
session_id = 'run_07'      # Change as needed

if has_week:
    group = df[(df['Subject'] == subject_id) & (df['Week'] == week_id) & (df['Session'] == session_id)]
    heading = (subject_id, week_id, session_id)
else:
    group = df[(df['Subject'] == subject_id) & (df['Session'] == session_id)]
    heading = (subject_id, session_id)

if group.empty:
    print(f"No data found for {heading}")
else:
    fig, ax = plt.subplots()
    time_array = range(0, time_jump * group.shape[0], time_jump)
    # Plot feeling trajectories
    for i, feeling in enumerate(feelings):
        color = colours.get(i, None)
        ax.plot(time_array, group[feeling]*10, label=feeling, color=color)
    # Annotate conditions if present
    annotate_conditions(ax, time_array, group)
    # Finalize plot
    combined = ''.join(map(str, heading)).translate({ord(c): None for c in "\\'() "})
    ax.set_title(combined)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rating')
    ax.legend(title='Feeling', bbox_to_anchor=(1.05, 1), loc='upper left')
    save_path = os.path.join(savelocation, f'raw_feeling_trajectories_{combined}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
