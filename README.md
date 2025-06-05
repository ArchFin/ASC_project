# ASC_project: Hidden Markov Model (HMM) Pipeline for Entropic Mind Analysis

## Overview
This project implements a robust pipeline for analyzing time-series data using a custom Hidden Markov Model (HMM) framework. The goal is to uncover hidden states underlying subjective feeling ratings, with a focus on understanding the concept of an "Entropic mind." The pipeline is designed for interpretability, reproducibility, and extensibility.

## Motivation
A HMM is a statistical model that describes a system assumed to be a Markov process with unobserved (hidden) states. It is ideal for time-series data where you observe emissions (outputs) but not the underlying state sequence. This makes HMMs powerful for modeling transitions in psychological or physiological states, such as those hypothesized in altered states of consciousness.

## Pipeline Structure

### 1. Data Input and Preprocessing
- **Input:** CSV file with time-series data (e.g., subjective ratings, conditions, session info).
- **Preprocessing:**
  - Data is split by subject/session/week as needed.
  - Features (feelings) are selected based on configuration in `Breathwork.yaml`.
  - Data is standardized (z-scored) for PCA and clustering.

### 2. Principal Component Analysis (PCA)
- **Purpose:** Reduce dimensionality and identify main axes of variation in feelings data.
- **Outputs:**
  - Bar plots of principal component loadings (which feelings drive each PC).
  - Scatter plot of data in PC1/PC2 space.
  - Explained variance ratio plot.
- **Rationale:** PCA helps visualize structure and ensures clustering is not dominated by noise or redundant features.

### 3. Custom HMM Clustering
- **Model:**
  - Uses a custom HMM with multivariate t-distribution emissions for robustness to outliers.
  - Includes an extra "transition" state to capture periods of change between stable states.
  - Parameters are initialized with KMeans and refined via EM-like training.
- **Key Decisions:**
  - t-distribution is chosen for its heavy tails, better modeling real-world data.
  - Transition state allows explicit modeling of non-stationary periods.
  - Multiple runs and state alignment ensure stability and reproducibility.

### 4. Visualization and Interpretation
- **State Sequence Plot:** Shows cluster assignment over time.
- **Gamma Heatmap:** Visualizes state membership probabilities (uncertainty, transitions).
- **Transition Matrix:** Heatmap of empirical transition probabilities.
- **State Duration Distribution:** Histogram of how long each state persists.
- **Cluster Feature Profiles:** Bar plots of mean feeling values per cluster.
- **PCA Projections:** Data and cluster centers in PC space.
- **Trajectory Plots:** Feeling time-courses with cluster/state shading and transition annotations.
- **Middle State Validation:** Checks if the dominant cluster matches the "neutral" state (smallest L2 norm).

### 5. Transition and Jump Analysis
- **Transition Annotation:** Each transition is classified as abrupt or gradual based on gamma slope.
- **Transition Metadata:** CSV output with timing, type, and condition frequencies before/during/after transitions.
- **Jump Analysis:**
  - Stability: Ratio of dominant cluster to others as a function of time step (jump).
  - Consistency: Agreement of clustering across downsampling.
  - Autocorrelation: Average ACF for each feeling.

### 6. Neural Decoder Integration
- **Purpose:** The neural decoder is an additional module designed to predict subjective feeling ratings or cluster assignments based on neural data.
- **Input:** Neural data in a compatible format (e.g., EEG, fMRI, or other time-series neural signals).
- **Preprocessing:**
  - Neural data is aligned with subjective ratings or cluster assignments using timestamps.
  - Features are extracted from the neural data (e.g., power spectral density, connectivity metrics).
  - Data is standardized and optionally reduced in dimensionality using PCA.
- **Model:**
  - A regression or classification model (e.g., linear regression, random forest, or neural network) is trained to map neural features to subjective ratings or cluster labels.
  - Cross-validation is used to evaluate model performance.
- **Outputs:**
  - Predicted ratings or cluster assignments.
  - Performance metrics (e.g., R^2, accuracy, confusion matrix).
  - Visualizations of feature importance or model predictions.

### How to Use the Neural Decoder
1. **Prepare Neural Data:** Ensure your neural data is preprocessed and aligned with the subjective ratings or cluster assignments.
2. **Update Configuration:** Add the neural data file path and relevant parameters to `Breathwork.yaml`.
3. **Run the Decoder Script:**
   ```bash
   python neural_decoder.py
   ```
4. **Review Outputs:** Check the output directory for performance metrics and visualizations.

### Extending the Neural Decoder
- **Add New Features:** Extract additional features from the neural data (e.g., time-frequency representations).
- **Change Models:** Experiment with different machine learning models or hyperparameters.
- **Integrate with Pipeline:** Use the decoder's predictions as additional inputs for the HMM pipeline or clustering analysis.

## Configuration File: `Breathwork.yaml`
The `Breathwork.yaml` file contains all the necessary configurations for the pipeline. Key sections include:
- **File Locations:** 
  - `filelocation_TET`: Path to the input CSV file.
  - `savelocation_TET`: Directory to save outputs.
- **Feelings and Differences:**
  - Lists of feelings and their vectorized differences to include in the analysis.
- **PCA and Clustering Parameters:**
  - `no_dimensions_PCA`: Number of dimensions for PCA.
  - `no_clust`: Number of clusters for KMeans.
  - `no_states`: Optimal number of HMM states.
- **Headers and Colors:**
  - `headers`: Column indices for subject, week, and session.
  - `colours`: Mapping of cluster indices to colors for visualization.

Ensure this file is correctly configured before running the pipeline.

## Outputs
- **CSV files:** Cluster assignments, transition metadata.
- **PNG plots:** All visualizations described above, saved to the configured output directory.
- **Pickle files:** For saving transition data for further analysis.
- **Text files:** Middle state validation results, cluster summaries.

## How to Run
1. **Install Dependencies:** Ensure you have Python 3.x and required libraries installed. Use `pip install -r requirements.txt` if a requirements file is provided.
2. **Configure Settings:** Update `Breathwork.yaml` with the correct file paths and parameters.
3. **Run the Main Script:**
   ```bash
   python HMM.py
   ```
4. **Review Outputs:** Check the output directory for visualizations, CSVs, and text files.

## File Structure
- `HMM_methods.py`: Core methods and classes for the pipeline.
- `HMM.py`: Main script to initialize and run the pipeline.
- `Breathwork.yaml`: Configuration file for paths, features, and parameters.
- `loop_etc.py`: Utility functions for batch processing or repeated analyses.
- `interative_etc.py`: Tools for interactive exploration, such as stepwise model fitting or parameter tuning.
- `README.md`: This guide.
- **Output Directory:** Contains PNGs, CSVs, pickles, and text files generated by the pipeline.

## Extending or Modifying the Pipeline
- **Add New Features:** Update the feelings list in `Breathwork.yaml` and ensure your CSV includes them.
- **Change Model Parameters:** Adjust the number of states, iterations, or t-distribution settings in the YAML or code.
- **Add Visualizations:** Extend the `Visualiser` or `CustomHMMClustering` classes.
- **Use with New Data:** Ensure your data matches the expected format and update paths in the YAML.

## Further Reading
- For more on HMMs: [Wikipedia - Hidden Markov Model](https://en.wikipedia.org/wiki/Hidden_Markov_model)
- For t-distribution HMMs: See literature on robust HMMs for time-series.
- For PCA: [Wikipedia - Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)

---
For questions or contributions, contact: archie.finney@yahoo.co.uk

