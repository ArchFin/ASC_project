import os
import glob
import pandas as pd
import scipy.io as sio
import numpy as np
import h5py
from pathlib import Path

def load_mat_file(mat_path):
    """
    Load a .mat file and extract data into a pandas DataFrame.
    Handles both old format (.mat) and new format (v7.3) files.
    
    Args:
        mat_path: Path to the .mat file
        
    Returns:
        DataFrame with the extracted data
    """
    try:
        # First try with scipy.io for older .mat files
        try:
            mat_data = sio.loadmat(mat_path)
            is_h5_format = False
        except NotImplementedError:
            # If that fails, use h5py for v7.3 files
            mat_data = h5py.File(mat_path, 'r')
            is_h5_format = True
        
        print(f"Loading {mat_path}")
        
        if is_h5_format:
            # Handle HDF5/v7.3 format
            data_keys = [key for key in mat_data.keys() if not key.startswith('#')]
            print(f"Available keys (HDF5): {data_keys}")
            
            # Try to find the main data structure
            main_data = None
            for key in data_keys:
                try:
                    data = mat_data[key]
                    if hasattr(data, 'shape') and np.prod(data.shape) > 1:
                        # Convert HDF5 dataset to numpy array
                        data_array = np.array(data)
                        print(f"Found data in key '{key}' with shape: {data_array.shape}")
                        main_data = data_array
                        break
                except Exception as e:
                    print(f"Could not read key '{key}': {e}")
                    continue
            
            # Close the HDF5 file
            mat_data.close()
        else:
            # Handle traditional .mat format
            data_keys = [key for key in mat_data.keys() if not key.startswith('__')]
            print(f"Available keys: {data_keys}")
            
            # Try to find the main data structure
            main_data = None
            for key in data_keys:
                data = mat_data[key]
                if isinstance(data, np.ndarray) and data.size > 1:
                    print(f"Found data in key '{key}' with shape: {data.shape}")
                    main_data = data
                    break
        
        if main_data is None:
            print(f"Warning: No suitable data found in {mat_path}")
            return None
        
        # Convert to DataFrame
        if main_data.ndim == 1:
            # 1D array - create single column
            df = pd.DataFrame({'data': main_data})
        elif main_data.ndim == 2:
            # 2D array - assume columns are features
            if main_data.shape[1] > main_data.shape[0]:
                # More columns than rows - transpose
                main_data = main_data.T
            
            # Define specific column names for the first 12 features
            specific_col_names = [
                'Aperture', 'Boredom', 'Clarity', 'Conflict', 'Dereification', 
                'Wakefulness', 'Emotion', 'Effort', 'Stability', 'MetaAwareness',
                'ObjectOrientation', 'Source'
            ]
            
            # Only keep the first 12 columns (remove features 13-17)
            if main_data.shape[1] > 12:
                main_data = main_data[:, :12]
                print(f"Trimmed data to first 12 features, new shape: {main_data.shape}")
            
            # Create column names - use specific names for first 12, generic for any additional
            if main_data.shape[1] <= 12:
                col_names = specific_col_names[:main_data.shape[1]]
            else:
                col_names = specific_col_names + [f'feature_{i+1}' for i in range(12, main_data.shape[1])]
            
            df = pd.DataFrame(main_data, columns=col_names)
        else:
            print(f"Warning: Data has {main_data.ndim} dimensions, flattening to 2D")
            # Reshape to 2D
            reshaped = main_data.reshape(main_data.shape[0], -1)
            
            # Apply same logic for multidimensional data
            specific_col_names = [
                'Aperture', 'Boredom', 'Clarity', 'Conflict', 'Dereification', 
                'Wakefulness', 'Emotion', 'Effort', 'Stability', 'MetaAwareness',
                'ObjectOrientation', 'Source'
            ]
            
            # Only keep the first 12 columns
            if reshaped.shape[1] > 12:
                reshaped = reshaped[:, :12]
                print(f"Trimmed reshaped data to first 12 features, new shape: {reshaped.shape}")
            
            if reshaped.shape[1] <= 12:
                col_names = specific_col_names[:reshaped.shape[1]]
            else:
                col_names = specific_col_names + [f'feature_{i+1}' for i in range(12, reshaped.shape[1])]
            
            df = pd.DataFrame(reshaped, columns=col_names)
        
        # Extract subject and session info from filename
        filename = Path(mat_path).stem
        parts = filename.split('_')
        
        # Add metadata columns
        df['filename'] = filename
        if len(parts) >= 2:
            df['subject_id'] = parts[0]
            df['session_type'] = parts[1] if len(parts) > 1 else 'unknown'
        
        # Add file path for reference
        df['source_file'] = mat_path
        
        return df
        
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return None

def convert_mat_directory_to_csv(input_dir, output_dir=None, pattern="*.mat"):
    """
    Convert all .mat files in a directory to CSV files.
    
    Args:
        input_dir: Directory containing .mat files
        output_dir: Directory to save CSV files (if None, uses input_dir)
        pattern: File pattern to match (default: "*.mat")
    """
    input_path = Path(input_dir)
    if output_dir is None:
        output_dir = input_path / "csv_converted"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .mat files
    mat_files = list(input_path.rglob(pattern))
    print(f"Found {len(mat_files)} .mat files")
    
    successful_conversions = 0
    all_dataframes = []
    
    for mat_file in mat_files:
        print(f"\nProcessing: {mat_file}")
        
        # Load and convert
        df = load_mat_file(str(mat_file))
        
        if df is not None:
            # Add to combined list (skip individual CSV saving)
            all_dataframes.append(df)
            successful_conversions += 1
        
    # Create combined CSV file
    if all_dataframes:
        print(f"\nCombining {len(all_dataframes)} files into single CSV...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_path = output_dir / "combined_all_files.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined CSV saved: {combined_path}")
        
        # Print summary statistics
        print(f"\nConversion Summary:")
        print(f"- Files processed: {len(mat_files)}")
        print(f"- Successful conversions: {successful_conversions}")
        print(f"- Combined data shape: {combined_df.shape}")
        print(f"- Combined data columns: {list(combined_df.columns)}")
        
        return combined_path
    else:
        print("No data was successfully converted")
        return None

def convert_single_mat_file(mat_path, csv_path=None):
    """
    Convert a single .mat file to CSV.
    
    Args:
        mat_path: Path to the .mat file
        csv_path: Output CSV path (if None, replaces .mat with .csv)
    """
    if csv_path is None:
        csv_path = Path(mat_path).with_suffix('.csv')
    
    df = load_mat_file(mat_path)
    
    if df is not None:
        df.to_csv(csv_path, index=False)
        print(f"Converted {mat_path} -> {csv_path}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return csv_path
    else:
        print(f"Failed to convert {mat_path}")
        return None

def convert_all_subjects_to_csv(base_meditation_dir, output_dir=None, pattern="*.mat"):
    """
    Find all subject directories containing '20-SubjExp' folders and convert their .mat files to a combined CSV.
    
    Args:
        base_meditation_dir: Base directory containing subject folders (e.g., '/Users/a_fin/Desktop/Year 4/Project/Meditation/DreemEEG')
        output_dir: Directory to save the combined CSV file
        pattern: File pattern to match (default: "*.mat")
    """
    base_path = Path(base_meditation_dir)
    if output_dir is None:
        output_dir = base_path / "csv_converted"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all subdirectories that contain a '20-SubjExp' folder
    subject_dirs = []
    for item in base_path.iterdir():
        if item.is_dir():
            subjexp_dir = item / "20-SubjExp"
            if subjexp_dir.exists() and subjexp_dir.is_dir():
                subject_dirs.append(subjexp_dir)
                print(f"Found subject directory: {item.name}")
    
    print(f"\nFound {len(subject_dirs)} subject directories with 20-SubjExp folders")
    
    if not subject_dirs:
        print("No subject directories with '20-SubjExp' folders found")
        return None
    
    all_dataframes = []
    total_successful_conversions = 0
    total_files_processed = 0
    
    # Process each subject directory
    for subjexp_dir in subject_dirs:
        subject_name = subjexp_dir.parent.name
        print(f"\n=== Processing subject: {subject_name} ===")
        
        # Find all .mat files in this subject's 20-SubjExp directory
        mat_files = list(subjexp_dir.rglob(pattern))
        print(f"Found {len(mat_files)} .mat files in {subject_name}")
        total_files_processed += len(mat_files)
        
        subject_successful = 0
        
        for mat_file in mat_files:
            print(f"Processing: {mat_file.name}")
            
            # Load and convert
            df = load_mat_file(str(mat_file))
            
            if df is not None:
                # Add subject identifier to the dataframe
                df['subject_directory'] = subject_name
                all_dataframes.append(df)
                subject_successful += 1
                total_successful_conversions += 1
        
        print(f"Subject {subject_name}: {subject_successful}/{len(mat_files)} files converted successfully")
    
    # Create combined CSV file
    if all_dataframes:
        print(f"\nCombining {len(all_dataframes)} files from {len(subject_dirs)} subjects into single CSV...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_path = output_dir / "combined_all_subjects.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined CSV saved: {combined_path}")
        
        # Print summary statistics
        print(f"\nConversion Summary:")
        print(f"- Subject directories processed: {len(subject_dirs)}")
        print(f"- Total files found: {total_files_processed}")
        print(f"- Successful conversions: {total_successful_conversions}")
        print(f"- Combined data shape: {combined_df.shape}")
        print(f"- Combined data columns: {list(combined_df.columns)}")
        
        # Show subject breakdown
        subject_counts = combined_df['subject_directory'].value_counts()
        print(f"\nData points per subject:")
        for subject, count in subject_counts.items():
            print(f"  {subject}: {count} rows")
        
        return combined_path
    else:
        print("No data was successfully converted from any subject")
        return None

def main():
    """
    Main function - can be customized for your specific needs
    """
    # Convert all subjects in base directory
    print("=== Converting all subjects in base directory ===")
    base_meditation_dir = '/Users/a_fin/Desktop/Year 4/Project/Meditation/DreemEEG'
    output_dir = '/Users/a_fin/Desktop/Year 4/Project/ASC_project/converted_csv'
    
    convert_all_subjects_to_csv(base_meditation_dir, output_dir)

if __name__ == "__main__":
    main()