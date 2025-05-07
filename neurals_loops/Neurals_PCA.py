#Neurals PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# 1. Load PSD spreadsheet
df = pd.read_excel('/Users/a_fin/Desktop/Year 4/Project/Data/psd_metrics_combined.xlsx')

# 2. Split out subject, week, run
df[['subject','week','run']] = df['Participant'].str.split('_', expand=True)

# 3. Clean up 
df['week']    = df['week'].str.replace(r'^wk', '', regex=True).astype(int)  # “wk1” → 1

df.drop(columns=['Participant'], inplace=True)



# 5. (Optional) save out
df.to_excel('/Users/a_fin/Desktop/Year 4/Project/Data/psd_metrics_combined.xlsx', index=False)








class principal_component_finder:
    def __init__(self, csv_file, feelings, no_dimensions, savelocation_TET):
        """
        Extracts the required data and performs PCA.
        """
        self.csv_file = csv_file[feelings]
        self.feelings = feelings
        self.savelocation = savelocation_TET
        self.no_dimensions = no_dimensions
        
        # Standardize the data
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.csv_file_TET)

        # Compute PCA
        pca = PCA(n_components=no_dimensions)
        self.principal_components = pca.fit(self.scaled_data).components_
        self.explained_variance_ratio = pca.explained_variance_ratio_

    def PCA_TOT(self):
        """
        Projects the data onto the principal components and plots bar charts for each component.
        Returns the principal components, explained variance ratio, and the transformed data.
        """

        # Project STANDARDIZED DATA 
        df_TET_feelings_prin = pd.DataFrame(self.scaled_data @ self.principal_components.T,  
                                            columns=[f"PC{i+1}" for i in range(self.no_dimensions)])

        # Plot bar charts for each principal component.
        for i in range(self.principal_components.shape[0]):
            plt.figure()
            plt.bar(self.feelings, self.principal_components[i])
            plt.title(f'Principal Component {i+1}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.savelocation, f'principal_component{i+1}.png'))
            plt.close()

        # Scatter plot of the first two principal components.
        plt.figure()
        plt.scatter(df_TET_feelings_prin.iloc[:, 0], df_TET_feelings_prin.iloc[:, 1], s=0.5)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Projection')
        # plt.xlim(-6, 6)
        # plt.ylim(-1, 2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.savelocation, 'all_data_in_PCA.png'))
        plt.close()

        # Bar chart for explained variance ratio.
        labels = [f'Principal Component {i+1}' for i in range(len(self.explained_variance_ratio))]
        plt.figure(figsize=(10, 6))
        plt.bar(labels, self.explained_variance_ratio, color='skyblue')
        plt.title('Explained Variance Ratio')
        plt.xlabel('Principal Components')
        plt.ylabel('Variance Explained')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.savelocation, 'explained_variance_ratio.png'))
        plt.close() 

        return self.principal_components, self.explained_variance_ratio, df_TET_feelings_prin

    def PCA_split(self, split_df_array):
        """
        Projects each split dataset onto the principal components and plots scatter plots.
        Returns a dictionary mapping split names to their PCA-transformed data.
        """
        if split_df_array == None:
            df_TET_feelings_prin_dict = None
        else:
            split_csv_TET = {item[0]: item[1][self.feelings] for item in split_df_array}
            df_TET_feelings_prin_dict = {name: df @ self.principal_components.T for name, df in split_csv_TET.items()}

            for key, value in df_TET_feelings_prin_dict.items():
                plt.figure()
                plt.scatter(value.iloc[:, 0], value.iloc[:, 1], s=0.5)
                plt.title(f'PCA Projection: {key}')
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                # plt.xlim(-6, 6)
                # plt.ylim(-1, 2)
                plt.tight_layout()
                plt.savefig(os.path.join(self.savelocation, f'PCA_{key}.png'))
                plt.close()

        return df_TET_feelings_prin_dict


