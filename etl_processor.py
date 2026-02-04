import pandas as pd


def _remove_duplicate_columns(df: pd.DataFrame, verbose: bool = True, remove_duplicates: bool = True, remove_zeros: bool = True) -> pd.DataFrame:
    """
    Remove duplicate columns within the same Component based on identical values across all rows.
    Columns are considered duplicates only if:
      - They belong to the same Component (parsed from the column name pattern 'Component_Zone.Parameter'), and
      - Their entire column values are identical.
    Keeps the first occurrence within each Component group.

    Remove zeros:
      - Columns that are constant all-zeros or all-ones can be preserved or removed.

    Processing order per Component:
      1) Identify constant (all-zeros/ones) columns and either keep or delete based on remove_zeros
      2) If remove_duplicates is True, deduplicate the remaining candidates; otherwise keep them all
    """
    if df.empty or df.shape[1] <= 1:
        if verbose:
            print("Duplicate check: DataFrame empty or has <=1 non-Time columns; skipping duplicate/zeros removal.")
        return df

    if verbose:
        print(f"Duplicate/Zeros removal: Starting with {df.shape[1]} non-Time columns")
        print(f"  Settings -> remove_duplicates={remove_duplicates}, remove_zeros={remove_zeros}")

    # Parse Component from column names. If pattern doesn't match, use full name to avoid cross-component merging.
    components = {}
    for col in df.columns:
        try:
            underscore_idx = col.find('_')
            dot_idx = col.find('.')
            if underscore_idx != -1 and dot_idx != -1 and underscore_idx < dot_idx:
                component_key = col[:underscore_idx]
            else:
                component_key = col
        except Exception:
            component_key = col
        components.setdefault(component_key, []).append(col)

    if verbose:
        print(f"Duplicate/Zeros removal: Found {len(components)} component groups")
        for comp_key, cols in components.items():
            print(f"  - Component '{comp_key}': {len(cols)} cols")

    columns_to_keep = []
    total_removed = 0
    for component_key, cols in components.items():
        if len(cols) == 1:
            columns_to_keep.append(cols[0])
            continue

        group_df = df[cols]

        # 1) Handle constants first (always excluded from dedup consideration)
        constant_cols = []
        candidate_cols_for_dedup = []
        for col in group_df.columns:
            series = group_df[col]
            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_all_zero = bool(is_numeric and series.eq(0).all())
            is_all_one = bool(is_numeric and series.eq(1).all())
            if is_all_zero or is_all_one:
                constant_cols.append(col)
            else:
                candidate_cols_for_dedup.append(col)

        kept_in_group = []
        removed_in_group = []

        if constant_cols:
            if remove_zeros:
                removed_in_group.extend(constant_cols)
                if verbose:
                    print(f"  > Component '{component_key}': deleting constant columns (all-zeros/ones): {len(constant_cols)}")
                    for c in constant_cols:
                        print(f"    delete constant: {c}")
            else:
                kept_in_group.extend(constant_cols)
                if verbose:
                    print(f"  > Component '{component_key}': preserving constant columns (all-zeros/ones): {len(constant_cols)}")
                    for c in constant_cols:
                        print(f"    keep constant: {c}")

        # 2) Deduplicate remaining candidates only if enabled
        if candidate_cols_for_dedup:
            if remove_duplicates:
                sub_df = group_df[candidate_cols_for_dedup]
                signature_to_columns = {}
                for col in sub_df.columns:
                    sig = pd.util.hash_pandas_object(sub_df[col], index=False).sum()
                    signature_to_columns.setdefault(sig, []).append(col)

                if verbose and any(len(v) > 1 for v in signature_to_columns.values()):
                    print(f"  > Component '{component_key}': detected identical-value groups (excluding constants)")

                for sig, same_cols in signature_to_columns.items():
                    if len(same_cols) == 1:
                        kept_in_group.append(same_cols[0])
                        continue
                    kept_col = same_cols[0]
                    dup_cols = same_cols[1:]
                    kept_in_group.append(kept_col)
                    removed_in_group.extend(dup_cols)
                    if verbose:
                        print(f"    keep: {kept_col}")
                        for rc in dup_cols:
                            print(f"      - duplicate deleted: {rc}")

                # Fallback if necessary
                if not kept_in_group and candidate_cols_for_dedup:
                    duplicated_mask = sub_df.T.duplicated(keep='first')
                    kept_in_group = list(sub_df.columns[~duplicated_mask.values])
                    removed_in_group = list(sub_df.columns[duplicated_mask.values])
                    if verbose and removed_in_group:
                        print(f"  > Component '{component_key}': keeping {len(kept_in_group)}, removing {len(removed_in_group)} duplicates")
                        for rc in removed_in_group:
                            print(f"    * removed duplicate column: {rc}")
            else:
                # Dedup disabled: keep all candidate columns
                kept_in_group.extend(candidate_cols_for_dedup)
                if verbose:
                    print(f"  > Component '{component_key}': dedup disabled, keeping all {len(candidate_cols_for_dedup)} candidate columns")

        columns_to_keep.extend(kept_in_group)
        total_removed += len(removed_in_group)

    if verbose:
        print(f"Duplicate/Zeros removal: Removed {total_removed} columns; {len(columns_to_keep)} remain")

    # Preserve original column order
    columns_to_keep_ordered = [c for c in df.columns if c in set(columns_to_keep)]
    return df[columns_to_keep_ordered]


def process_csv_to_pivot(csv_path, sep=';', start_date='2019-01-01 00:00:00', remove_duplicates: bool = True,
                         remove_zeros: bool = True, verbose: bool = True):
    """
    Process a CSV file and convert it to a pivoted format.
    
    Args:
        csv_path (str): Path to the CSV file
        sep (str): CSV separator (default: ';')
        start_date (str): Start date for time conversion (default: '2019-01-01 00:00:00')
        remove_duplicates (bool): If True, remove duplicate columns within the same Component
        remove_zeros (bool): If True, remove columns that are all zeros or all ones
        verbose (bool): If True, print progress/debug information
    
    Returns:
        pandas.DataFrame: Pivoted DataFrame
    """
    if verbose:
        print(f"Reading CSV file: {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path, sep=sep)
    df = df.loc[:, ~df.columns.astype(str).str.match(r'^Unnamed')]
    
    if verbose:
        print(f"Original dataframe shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    
    # Validate required Time column
    if 'Time' not in df.columns:
        raise KeyError("Expected time column 'Time' not found in CSV")

    # Convert Time from seconds to datetime
    start_timestamp = pd.Timestamp(start_date)
    if verbose:
        print(f"Converting 'Time' (seconds) to datetime using start_date={start_date}")
    df['Time'] = start_timestamp + pd.to_timedelta(df['Time'], unit='s')
    if verbose:
        print("Sample of converted Time:")
        print(df['Time'].head())

    # Optionally remove duplicate and/or constant columns (excluding the Time identifier)
    if remove_duplicates or remove_zeros:
        non_id_cols = [c for c in df.columns if c != 'Time']
        if verbose:
            print(
                f"Checking duplicates/zeros among {len(non_id_cols)} non-Time columns (remove_duplicates={remove_duplicates}, remove_zeros={remove_zeros})")
        df_non_id = _remove_duplicate_columns(
            df[non_id_cols],
            verbose=verbose,
            remove_duplicates=remove_duplicates,
            remove_zeros=remove_zeros
        ) if non_id_cols else df[non_id_cols]
        df = pd.concat([df[['Time']], df_non_id], axis=1)
        if verbose:
            print(f"After duplicate/zeros removal shape: {df.shape}")
    else:
        if verbose:
            print("Duplicate and zeros removal disabled.")

    # Melt the dataframe to convert from wide to long format
    df_melted = df.melt(
        id_vars=['Time'],
        var_name='Column',
        value_name='Value'
    )

    if verbose:
        print(f"Melted dataframe shape: {df_melted.shape}")
        print("Sample melted rows:")
        print(df_melted.head())

    # Split the Column names into 3 separate columns
    df_melted[['Component', 'Zone', 'Parameter']] = df_melted['Column'].str.extract(r'^([^_]+)_([^.]+)\.(.+)$')

    # Reset the index to make Time a regular column
    df_melted = df_melted.reset_index(drop=True)

    if verbose:
        print(f"\nTime range: {df_melted['Time'].min()} to {df_melted['Time'].max()}")
        print("Sample time values:")
        print(df_melted['Time'].head())

    return df_melted[['Time', 'Component', 'Zone', 'Parameter', 'Value']]


def process_plan_file(csv_path, sep=';', verbose: bool = True):
    """
    Process a plan CSV file:

    Args:
        csv_path (str): Path to the CSV file
        sep (str): CSV separator (default: ';')
        verbose (bool): If True, print progress/debug information

    Returns:
        pandas.DataFrame: DataFrame with original columns plus metadata
    """
    # Split Model into two columns for Component_Zone
    def split_model(val):
        val = str(val).strip()
        if "_" in val:
            component, zone = val.split("_", maxsplit=1)
        else:
            component, zone = val, "Central"
        return component, zone

    # Read the CSV file
    df = pd.read_csv(csv_path, sep=sep)
    df = df.loc[:, ~df.columns.astype(str).str.match(r'^Unnamed', case=False)]
    comp_zone = df['Model'].apply(split_model)
    df.insert(0, 'Component', comp_zone.apply(lambda x: x[0]))
    df.insert(1, 'Zone', comp_zone.apply(lambda x: x[1]))
    df = df.drop(columns=['Model'])

    if verbose:
        print(f"Plan file shape: {df.shape}")
        print(f"Columns (after processing): {list(df.columns)}")
        print(df.head(10))

    return df
