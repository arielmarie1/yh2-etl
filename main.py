import pandas as pd
import os
from dotenv import load_dotenv
import glob

from etl_processor import process_csv_to_pivot, process_plan_file
from filename_parser import parse_filename_to_columns
from database import DatabaseManager, create_connection_string, Timeseries, Plan

# Load environmental variables
load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', "yh2_files"),
    'username': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASS')
}


if __name__ == "__main__":
    # Validate database configuration (env file)
    DatabaseManager.ensure_db_config(DB_CONFIG)
    # Create database if doesn't already exist
    DatabaseManager.ensure_database_exists(DB_CONFIG)
    # Create connection to database
    connection_string = create_connection_string(**DB_CONFIG)
    db_manager = DatabaseManager(connection_string)

    try:
        db_manager.create_tables()
        print("Tables created successfully.")
    finally:
        db_manager.close()
    
    # Get all CSV files in the data folder
    all_csv_files = glob.glob('data/*.csv')
    
    # Separate files by type
    timeseries_files = [f for f in all_csv_files if f.endswith('_results_rollinghorizon.csv')]
    plan_files = [f for f in all_csv_files if f.endswith('_results_PLAN.csv')]
    
    print(f"Found {len(all_csv_files)} total CSV files:")
    print(f"  - Timeseries files: {len(timeseries_files)}")
    print(f"  - Plan files: {len(plan_files)}")
    
    for file in timeseries_files:
        print(f"    Timeseries: {os.path.basename(file)}")
    for file in plan_files:
        print(f"    Plan: {os.path.basename(file)}")
    print()
    
    # Process Timeseries files
    timeseries_combined_df = None
    timeseries_files_processed = 0
    
    if timeseries_files:
        print(f"\n{'='*60}")
        print(f"PROCESSING TIMESERIES FILES")
        print(f"{'='*60}")
        
        for csv_file in timeseries_files:
            print(f"\n{'-'*40}")
            print(f"Processing: {os.path.basename(csv_file)}")
            print(f"{'-'*40}")
            
            try:
                # Process the CSV and get the pivoted table
                pivoted_df = process_csv_to_pivot(
                    csv_file,
                    remove_duplicates=True,
                    remove_zeros=True,
                    verbose=True,
                )
                
                # Parse filename to extract metadata columns
                metadata = parse_filename_to_columns(csv_file, verbose=True)
                print(f"Extracted metadata from filename:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
                
                # Add metadata columns to each row
                for key, value in metadata.items():
                    pivoted_df[key] = value
                
                # Print first 10 rows
                print(f"\nFirst 10 rows of pivoted data (with metadata):")
                print(pivoted_df.head(10))
                
                # Accumulate into combined dataframe
                if timeseries_combined_df is None:
                    timeseries_combined_df = pivoted_df.copy()
                else:
                    timeseries_combined_df = pd.concat([timeseries_combined_df, pivoted_df], ignore_index=True)
                
                timeseries_files_processed += 1
                print(f"✓ Added {len(pivoted_df)} rows from {os.path.basename(csv_file)}")
                print(f"  Combined total: {len(timeseries_combined_df)} rows")
                
            except FileNotFoundError:
                print(f"Error: File '{csv_file}' not found.")
            except Exception as e:
                print(f"Error processing file {csv_file}: {e}")
                continue
    
    # Process Plan files
    plan_combined_df = None
    plan_files_processed = 0
    
    if plan_files:
        print(f"\n{'='*60}")
        print(f"PROCESSING PLAN FILES")
        print(f"{'='*60}")
        
        for csv_file in plan_files:
            print(f"\n{'-'*40}")
            print(f"Processing: {os.path.basename(csv_file)}")
            print(f"{'-'*40}")
            
            try:
                # Process the plan CSV (no pivoting needed)
                plan_df = process_plan_file(csv_file, verbose=True)
                
                # Parse filename to extract metadata columns
                metadata = parse_filename_to_columns(csv_file, verbose=True)
                print(f"Extracted metadata from filename:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
                
                # Add metadata columns to each row
                for key, value in metadata.items():
                    plan_df[key] = value
                
                # Print first 10 rows
                print(f"\nFirst 10 rows of plan data (with metadata):")
                print(plan_df.head(10))
                
                # Accumulate into combined dataframe
                if plan_combined_df is None:
                    plan_combined_df = plan_df.copy()
                else:
                    plan_combined_df = pd.concat([plan_combined_df, plan_df], ignore_index=True)
                
                plan_files_processed += 1
                print(f"✓ Added {len(plan_df)} rows from {os.path.basename(csv_file)}")
                print(f"  Combined total: {len(plan_combined_df)} rows")
                
            except FileNotFoundError:
                print(f"Error: File '{csv_file}' not found.")
            except Exception as e:
                print(f"Error processing file {csv_file}: {e}")
                continue
    
    # Export results
    print(f"\n{'='*60}")
    print(f"EXPORTING RESULTS")
    print(f"{'='*60}")
    
    # Export Timeseries data
    if timeseries_combined_df is not None and len(timeseries_combined_df) > 0:
        print(f"\nTimeseries Summary:")
        print(f"  Files processed: {timeseries_files_processed}")
        print(f"  Total rows: {len(timeseries_combined_df)}")
        print(f"  Shape: {timeseries_combined_df.shape}")
        print(f"  Columns: {list(timeseries_combined_df.columns)}")
        
        # Export to CSV
        timeseries_filename = "timeseries.csv"
        timeseries_combined_df.to_csv(timeseries_filename, index=False)
        print(f"✓ Exported timeseries data to: {timeseries_filename}")
        
        # Show sample
        print(f"\nSample of timeseries data (first 5 rows):")
        print(timeseries_combined_df.head())
    
    # Export Plan data
    if plan_combined_df is not None and len(plan_combined_df) > 0:
        print(f"\nPlan Summary:")
        print(f"  Files processed: {plan_files_processed}")
        print(f"  Total rows: {len(plan_combined_df)}")
        print(f"  Shape: {plan_combined_df.shape}")
        print(f"  Columns: {list(plan_combined_df.columns)}")
        
        # Export to CSV
        plan_filename = "plan.csv"
        plan_combined_df.to_csv(plan_filename, index=False)
        print(f"✓ Exported plan data to: {plan_filename}")
        
        # Show sample
        print(f"\nSample of plan data (first 5 rows):")
        print(plan_combined_df.head())
    
    # Upload to database
    if (timeseries_combined_df is not None and len(timeseries_combined_df) > 0) or \
       (plan_combined_df is not None and len(plan_combined_df) > 0):
        answer = input("\nReplace existing rows for matching scenarios? (y/N): ").strip().lower()
        mode = "replace" if answer == "y" else "skip"
        try:
            # Create database connection
            conn_string = create_connection_string(**DB_CONFIG)
            db_manager = DatabaseManager(conn_string)
            db_manager.create_tables()

            # Insert timeseries data
            if timeseries_combined_df is not None and len(timeseries_combined_df) > 0:
                print("\nUploading timeseries data...")
                db_manager.insert_timeseries_data(timeseries_combined_df, mode=mode)

            # Insert PLAN data
            if plan_combined_df is not None and len(plan_combined_df) > 0:
                print("\nUploading plan data...")
                db_manager.insert_plan_data(plan_combined_df, mode=mode)
            print("✓ Data successfully uploaded to PostgreSQL!")
        except Exception as e:
            print(f"Error uploading to database: {e}")
        finally:
            if 'db_manager' in locals():
                db_manager.close()
    else:
        print("No data to upload.")
    
    if ((timeseries_combined_df is None or timeseries_combined_df.empty) and
            (plan_combined_df is None or plan_combined_df.empty)):
        print("No data was processed successfully.")
