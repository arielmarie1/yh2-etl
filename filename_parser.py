import os


def parse_filename_to_columns(filename: str, verbose: bool = True) -> dict:
    """
    Parse filename to extract metadata columns based on underscore separators.
    Expected formats: 
      - Timeseries: Duration_Scenario_results_rollinghorizon.csv
      - Plan: Duration_Scenario_results_PLAN.csv
    
    Args:
        filename (str): Full filename including path
        verbose (bool): If True, print warning messages for non-standard filenames
        
    Returns:
        dict: Dictionary with keys: Duration, Scenario
    """
    # Extract just the filename without path
    basename = os.path.basename(filename)
    
    # Remove the appropriate suffix based on file type
    if basename.endswith('_results_rollinghorizon.csv'):
        basename = basename[:-len('_results_rollinghorizon.csv')]
        file_type = 'timeseries'
    elif basename.endswith('_results_PLAN.csv'):
        basename = basename[:-len('_results_PLAN.csv')]
        file_type = 'plan'
    else:
        if verbose:
            print(f"Warning: Filename '{basename}' doesn't match expected patterns.")
        file_type = 'unknown'
    
    # Split by underscores
    parts = basename.split('_')
    
    # Expected 2 parts: Duration, Scenario
    if len(parts) >= 2:
        if verbose:
            print(f"Parsed {file_type} file: {basename}")
        return {
            'Duration': parts[0],
            'Scenario': parts[1]
        }
    else:
        # Fallback if filename doesn't match expected pattern
        if verbose:
            print(f"Warning: Filename '{basename}' doesn't match expected pattern. Using fallback values.")
        return {
            'Duration': parts[0] if len(parts) > 0 else 'Unknown',
            'Scenario': parts[1] if len(parts) > 1 else 'Unknown',


        } 