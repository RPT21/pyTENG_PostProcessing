import os
import re
import numpy as np
import pandas as pd

# %% --------------------------------------------------------------------------
# PARSE CUSTOM TIME TO SECONDS
# -----------------------------------------------------------------------------

def LTIME_to_seconds(LTIME):
    
    conversor = {"d":86400,
                 "h": 3600,
                 "m": 60,
                 "s": 1,
                 "ms": 1e-3,
                 "us": 1e-6,
                 "ns": 1e-9}

    units = re.split(r'\d+', LTIME)[1:]
    numbers_str = re.findall(r'\d+', LTIME)
    numbers = [int(number) for number in numbers_str]
    
    total_time = 0

    for number, unit in zip(numbers, units):
        total_time += number * conversor[unit]
    
    return total_time

# %% --------------------------------------------------------------------------
# RENAME RAWDATA DICTIONARIES
# -----------------------------------------------------------------------------

MotColumnsRenames = {
    'Time(s)': 'Time',
    'MC SW Overview - Actual Position(mm)': 'Position',
    'MC SW Force Control - Measured Force(N)': 'Force',
    'MC SW Force Control - Target Force(N)': 'TargetForce',
    'LINMOT_MOVING_BOOL': 'IsMoving_Bool',
    'LINMOT_UP_AND_DOWN_BOOL': 'Motor_Up_Down_Bool'
}

DaqColumnsRenames = {
    'Time (s)': 'Time',
    'LinMot_Enable': 'IsMoving_Bool',
    'LinMot_Up_Down': 'Motor_Up_Down_Bool'
}

# %% --------------------------------------------------------------------------
# VALIDATE BINARY COLUMN FUNCTION
# -----------------------------------------------------------------------------

def validate_binary_column(df, column):
    # Check that all te values are {0, 1}
    if not df[column].isin([0, 1]).all():
        # Identify which are the values that are not 0 or 1
        wrong_values = df[~df[column].isin([0, 1])][column].unique()
        raise ValueError(f"Error: The column '{column}' contains not binary values: {wrong_values}")


# %% --------------------------------------------------------------------------
# SYNCHRONIZE DATAFRAMES FUNCTION
# -----------------------------------------------------------------------------

def synchronize_dataframes(dataframes_list, time_col='Time (s)', filter_time=True, binary_cols=("LinMot_Enable", "LinMot_Up_Down")):
    """
    Synchronizes a list of DataFrames to the highest sampling rate found among them.
    It does a temporal boundary alignment as well (make all data have the same physical duration).
    IMPORTANT: All dataframes have to start at the same timestep (for example, time = 0) and have different columns

    Args:
        dataframes_list (list): List of Pandas DataFrames.
        time_col (str): The name of the time column (must be present in all DFs).

    Returns:
        List of Pandas DataFrames having the same time column as an index
    """

    if not dataframes_list:
        return []

    # =================================================================
    # STEP 0: Temporal Boundary Alignment: all data represent the same physical duration of the experiment
    # =================================================================

    # Check that timestamps are correct:
    for df in dataframes_list:
        if not (df[time_col].is_monotonic_increasing and df[time_col].is_unique):
            raise ValueError("The time column must be strictly increasing for interpolation.")

    # Check that all dataframes start at the same time
    valor_ref = dataframes_list[0].iloc[0][time_col]
    all_equal = all(df.iloc[0][time_col] == valor_ref for df in dataframes_list)
    if not all_equal:
        raise Exception("The dataframes don't start at the same time")

    if filter_time:
        # Find the elapsed time for each dataframe and select the smallest one
        print("Analysing time duration...")
        min_end_time = float('inf')
        for i, df in enumerate(dataframes_list):
            current_time = df[time_col].iloc[-1]
            print(f"  - DataFrame {i}: Time duration = {current_time:.6f} s")
            if current_time < min_end_time:
                min_end_time = current_time
        print(f"The experiment ended at time: {min_end_time:.4f} seconds.")

        # Filter the dataframes to make them have the same time duration
        for i in range(len(dataframes_list)):
            # Filter the dataframes to the min_end_time timestamp (ignore old index [drop=True] and create a new one)
            dataframes_list[i] =\
                dataframes_list[i][dataframes_list[i][time_col] <= min_end_time].copy().reset_index(drop=True)
        print("All dataframes now have the same time duration")

    # =================================================================
    # STEP 1: Find the Highest Sampling Rate DataFrame
    # =================================================================
    min_time_step = float('inf')
    master_time_index = None

    print("Analyzing sampling rates...")

    for i, df in enumerate(dataframes_list):

        current_step = df[time_col].diff().mean()
        print(f"  - DataFrame {i}: Average timestep = {current_step:.6f} s")

        if current_step < min_time_step:
            # Save the timestamp of the highest sampling rate DataFrame
            master_time_index = df[time_col].values
            min_time_step = current_step

    print(f"Target sampling step: {min_time_step:.6f} s")

    # =================================================================
    # STEP 2: Resample and Interpolate all DataFrames
    # =================================================================
    synced_dataframes = []

    for df in dataframes_list:

        # 1. Prepare the DF: remove duplicates and set the time column as the index
        df_temp = df.drop_duplicates(subset=[time_col]).set_index(time_col)

        # 2. Reindex and Interpolate
        # - union(): Merges the dataframe's timestamp with the higher sampling rate timestamp
        # - interpolate(method='index'): Uses the actual time values to calculate data in the new timesteps
        # - loc[master_time_index]: Keeps only the points belonging to the master timestamp
        df_sync = (
            df_temp
            .reindex(df_temp.index.union(master_time_index))
            .interpolate(method='index')
            .loc[master_time_index]
        )

        # 3. Ensure binary columns to be int data type
        found_binary_columns = [col for col in binary_cols if col in df_sync.columns]
        if found_binary_columns:
            # Apply rounding and cast to integer for all matching columns simultaneously (Vectorization)
            df_sync[found_binary_columns] = df_sync[found_binary_columns].round().astype(int)

        synced_dataframes.append(df_sync)

    return synced_dataframes

# %% --------------------------------------------------------------------------
# MERGE DAQ RAWDATA FROM DIFFERENT TASKS INTO A SINGLE DATAFRAME
# -----------------------------------------------------------------------------

def merge_DAQ_data(folder_path, time_col='Time (s)'):
    files = [str(f) for f in os.listdir(folder_path) if f.endswith('.pkl')]

    dataframes = []

    for file in files:
        try:
            df = pd.read_pickle(os.path.join(folder_path, file))
            dataframes.append(df)
        except Exception as e:
            raise Exception(f'Error reading DAQ file {file}: {e}.')

    # Synchronize dataframes
    synced_dataframes = synchronize_dataframes(dataframes, time_col=time_col)

    # Concatenate directly (since all DFs share the same master time index)
    df = pd.concat(synced_dataframes, axis=1)

    # Restore the time index as a standard column
    df.index.name = time_col
    df = df.reset_index()

    # Check synchronization columns:
    if not ("LinMot_Enable" in df.columns and "LinMot_Up_Down" in df.columns):
        raise Exception("Error, LinMot_Up_Down and LinMot_Enable are not present in the dataframes")

    # Validate that they have valid data
    for col in df.columns:
        if col in ["LinMot_Enable", "LinMot_Up_Down"]:
            validate_binary_column(df, col)

    # Calculate the differences between adjacent values
    diff = df["LinMot_Enable"].diff()

    # Search indices
    up_index = diff.index[diff == 1].tolist()
    down_index = diff.index[diff == -1].tolist()

    if len(up_index) != 1 and len(down_index) != 1:
        raise Exception("Error, LinMot_Enable start and end position not found")
    else:
        up_index = up_index[0] - 1
        down_index = down_index[0]

    print("Found rising edge and falling edge positions in LinMot_Enable")
    print(f"Rising edge (from 0 to 1): {up_index}")
    print(f"Falling edge (from 1 to 0): {down_index}")

    # Filter dataframes using the calculated indices and take the first row as the time reference value
    df_final = df.loc[up_index : down_index].reset_index(drop=True)
    df_final[time_col] -= df_final[time_col].iloc[0]
    return df_final

# %% --------------------------------------------------------------------------
# LOAD MOTOR RAWDATA
# -----------------------------------------------------------------------------

def LoadMotorFile(ExpPath):
    '''
    Loads and processes a motor CSV file.
    
    Parameters
    ----------
    ExpPath : str
        Path to the experiment folder.
    
    Returns
    -------
    pd.DataFrame or None
        Processed motor data or None if there is an error.
    '''

    # Find files with CSV extension
    files = [f for f in os.listdir(ExpPath) if f.endswith('.csv')]
    if len(files) == 0:
        print(f'No motor data file found in {ExpPath}.')
        return None
    elif len(files) > 1:
        print(f'More than one motor data file found in {ExpPath}, the program cannot proceed.')
        return None

    # Define the MotorFile path
    MotorFile = os.path.join(ExpPath, files[0])
    MotorFile = os.path.normpath(MotorFile)

    try:
        dfMot = pd.read_csv(MotorFile, header=0, index_col=False,
                            delimiter=',', decimal='.')
    except Exception as e:
        print(f'Error reading Motor file {MotorFile}: {e}.')
        return None
    
    # Check if required source columns exists, and then rename them
    for col in MotColumnsRenames.keys():
        if col not in dfMot.columns:
            print(f'Column {col} not found in {MotorFile}.')
            return None
    
    # Rename defined columns
    dfMot = dfMot.rename(columns=MotColumnsRenames)
    
    # Ensure columns have the correct data types
    dfMot = dfMot.astype({
        'Position': float,
        'Force': float,
        'TargetForce': float,
        'IsMoving_Bool': int,
        'Motor_Up_Down_Bool': int
    })
    
    # Corrections Time
    dfMot['Time'] = dfMot['Time'].apply(LTIME_to_seconds).astype(float)
    dfMot['Time'] -= dfMot['Time'].iloc[0]
    
    # Corrections Position
    dfMot['Position'] -= dfMot['Position'].min()

    # Corrections Force and TargetForce
    dfMot['Force'] = -dfMot['Force']
    dfMot['TargetForce'] = -dfMot['TargetForce']
    
    return dfMot

# %% --------------------------------------------------------------------------
# LOAD DAQ RAWDATA
# -----------------------------------------------------------------------------

def LoadDAQData(ExpPath):
    '''
    Loads and processes a DAQ pickle file.
    
    Parameters
    ----------
    ExpPath : str
        Path to the experiment folder.
    
    Returns
    -------
    pd.DataFrame or None
        Processed DAQ data or None if there is an error.
    '''

    # Read the DAQ Data
    dfDaq = merge_DAQ_data(ExpPath)
    
    # Check if required source columns exists, and then rename them
    for col in DaqColumnsRenames.keys():
        if col not in dfDaq.columns:
            print(f'Column {col} not found in the experiment folder {ExpPath}.')
            return None
    
    # Rename defined columns
    dfDaq = dfDaq.rename(columns=DaqColumnsRenames)
    
    # Ensure columns have the correct data types
    dfDaq = dfDaq.astype({
        'Time': float,
        'IsMoving_Bool': int,
        'Motor_Up_Down_Bool': int
    })

    return dfDaq


# %% --------------------------------------------------------------------------
# FIND CYCLES FUNCTION
# -----------------------------------------------------------------------------

def FindCycles(df):
    '''
    Identifies start and end indices of operational cycles based on state changes.
    
    Parameters
    ----------
    dataframe : Pandas Dataframe with two binary synchronization columns (Enable/Disable, Move_Up/Move_Down)
    
    Returns
    -------
    list of [start, end]
        List of cycles represented by their star and end indices.
    '''
    state_series = df['IsMoving_Bool'] + df['Motor_Up_Down_Bool']
    cycles = []
    prev_state = state_series.iloc[0]
    start = None
    
    for i, s in enumerate(state_series[1:], start=1):
        if s != prev_state:
            if s == 2:
                if start is not None:
                    cycles.append([start, i - 1])
                start = i
            elif s == 0 and start is not None:
                cycles.append([start, i - 1])
                start = None
            prev_state = s
        
    if start is not None:
        cycles.append([start, len(state_series) - 1])
    
    return cycles


# %% --------------------------------------------------------------------------
# LOAD AND SYNCHRONIZE RAWDATA FILES
# -----------------------------------------------------------------------------

def ExtractCycles(ExpPath):
    '''
    Loads and synchronizes Motor and DAQ data files, returning combined cycles data.
    
    Parameters
    ----------
    ExpPath : str
        Path to the experiment directory containing the data files.
    
    Returns
    -------
    Cycles : list[pd.DataFrame]
        List of DataFrames, each containing data for a single cycle.
    '''
    # Load Motor data
    dfMot = LoadMotorFile(ExpPath)
    if dfMot is None:
        return []
    
    # Load DAQ data
    dfDaq = LoadDAQData(ExpPath)
    if dfDaq is None:
        return []
    
    # Motor sampling rate
    MotFs = 1 / dfMot['Time'].diff().mean()
    print(f'Motor sampling rate: {MotFs}.')
    
    # DAQ sampling rate
    DaqFs = 1 / dfDaq['Time'].diff().mean()
    print(f'DAQ sampling rate: {DaqFs}.')

    ### Synchronize motor and DAQ data ###

    # Calculate the differences between adjacent values
    diff = dfMot['IsMoving_Bool'].diff()

    # Search indices
    up_index = diff.index[diff == 1].tolist()
    down_index = diff.index[diff == -1].tolist()

    if len(up_index) != 1 and len(down_index) != 1:
        raise Exception("Error, LinMot_Enable start and end position not found")
    else:
        up_index = up_index[0] - 1
        down_index = down_index[0]

    print("Found rising edge and falling edge positions in LinMot_Enable")
    print(f"Rising edge (from 0 to 1): {up_index}")
    print(f"Falling edge (from 1 to 0): {down_index}")

    # Filter Motor dataframe using the calculated indices and reset time reference
    dfMot = dfMot.loc[up_index : down_index].copy().reset_index(drop=True)
    dfMot['Time'] -= dfMot['Time'].iloc[0]

    # Adjust the motor timestamp
    dfMot_time = dfMot['Time'].iloc[-1] - dfMot['Time'].iloc[0]
    dfDaq_time = dfDaq['Time'].iloc[-1] - dfDaq['Time'].iloc[0]
    print("Adjusting DAQ and Motor timestamps")
    print("Motor time: ", dfMot_time)
    print("DAQ time: ", dfDaq_time)
    print("Difference: ", abs(dfMot_time - dfDaq_time), "seconds")
    dfMot['Time'] = dfMot['Time'] * (dfDaq_time / dfMot_time)

    # Synchronize dataframes
    [dfDaq, dfMot] = synchronize_dataframes([dfDaq, dfMot],
                                            time_col='Time',
                                            filter_time=False,
                                            binary_cols=['IsMoving_Bool', 'Motor_Up_Down_Bool'])

    # Restore the time index as a standard column
    for df in [dfMot, dfDaq]:
        df.index.name = 'Time'
        df.reset_index(inplace=True)

    # Finding cycles
    MotCycles = FindCycles(dfMot)
    DaqCycles = FindCycles(dfDaq)
    if len(MotCycles) != len(DaqCycles):
        print(f'Different number of cycles: Motor={len(MotCycles)}, DAQ={len(DaqCycles)}. Using minimum.')
    nCycles = min(len(MotCycles), len(DaqCycles))
    
    Cycles = []
    for idx in range(nCycles):

        # Obtain data from the idx cycle interval (we are iterating for each cycle interval)
        dfcyM = dfMot.iloc[MotCycles[idx][0] : MotCycles[idx][1] + 1].reset_index(drop=True)
        dfcyD = dfDaq.iloc[DaqCycles[idx][0] : DaqCycles[idx][1] + 1].reset_index(drop=True)

        # Find the first index where (State == 1) in the cycle interval for the Motor
        mask = dfcyM['Motor_Up_Down_Bool'].eq(1)
        if mask.any():
            i = mask.idxmax()
        else:
            # Incomplete cycle
            continue

        # Find the first index where (State == 1) in the cycle interval for the DAQ
        mask = dfcyD['Motor_Up_Down_Bool'].eq(1)
        if mask.any():
            j = mask.idxmax()
        else:
            # Incomplete cycle
            continue

        # Interpolate the data columns using DAQ sampling rate as the reference
        for col in dfcyM.columns:
            if col == 'Time' or col == 'State':
                continue

            # Create in the DAQ a new column to store the motor data
            dfcyD[col] = np.zeros(len(dfcyD), dtype=float)

            # Down-phase interpolation
            valid_mask_down = dfcyM[col].iloc[:i].notna()
            xp = dfcyM['Time'].iloc[:i][valid_mask_down].values
            fp = dfcyM[col].iloc[:i][valid_mask_down].values
            x = dfcyD['Time'].iloc[:j].values

            if np.all(np.diff(xp) > 0) and len(xp) > 1:
                dfcyD.loc[dfcyD.index[:j], col] = np.interp(x, xp, fp)

            # Up-phase interpolation
            valid_mask_up = dfcyM[col].iloc[i:].notna()
            xp = dfcyM['Time'].iloc[i:][valid_mask_up].values
            fp = dfcyM[col].iloc[i:][valid_mask_up].values
            x = dfcyD['Time'].iloc[j:].values

            if np.all(np.diff(xp) > 0) and len(xp) > 1:
                dfcyD.loc[dfcyD.index[j:], col] = np.interp(x, xp, fp)
        
        # Keep the resulting cycle
        Cycles.append(dfcyD)
    
    return Cycles
