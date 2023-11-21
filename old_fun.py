def aggregate_load_data(df):
    
    agg_by = "StartTime"

    # if 'Z' not in df.StartTime or 'Z' not in df.EndTime:
    #     raise ValueError("Different timezone offset than the assumed one")
    
    check_offsets(df)

    if len(np.unique(df.AreaID)) > 1:
        raise Exception("AreaID not unique.")
    
    df['StartTime'] = df.StartTime.str.split('+').str[0]
    df['EndTime'] = df.EndTime.str.split('+').str[0]  
    df['StartTime'] = pd.to_datetime(df.StartTime)
    df['EndTime'] = pd.to_datetime(df.EndTime)

    if sum(df['EndTime'].isna()) > 0 or sum(df['StartTime'].isna()) > 0:
        raise Exception("There are some NAs in timestamps")

    df = replenish_zeros(df)
    
    aggregation_bound = timedelta(hours=1)
    cols = [agg_by,'Load']
    df_agg = df.loc[df['EndTime'] - df['StartTime'] < aggregation_bound,cols] # dataframe rows that require aggregation
    
    if df.loc[df['EndTime'] - df['StartTime'] > aggregation_bound,cols].shape[0] > 0:
        raise Exception("Some rows have an interval of more than 1 hour.") # Any rows have a higher

    if df_agg.shape[0] > 0:
        df_good = df.loc[~df.index.isin(df_agg.index),cols]
        df_good.set_index(agg_by,drop=True,inplace=True)
        df_agg = df_agg.resample("H", on = agg_by).sum()
        df = pd.concat([df_good,df_agg])
    else:
        df = df[cols]
        df.set_index(agg_by,drop=True,inplace=True)
    
    df.sort_index(inplace=True)
    
    if len(np.unique(df.index)) != df.shape[0]:
        raise Exception("Non-unique indices.")

    return df


def aggregate_gen_data(df):

    agg_by = "StartTime"
    
    check_offsets(df)

    if len(np.unique(df.AreaID[~df.AreaID.isna()].astype(str))) > 1:
        raise Exception("AreaID not unique.")
    
    if sum(df.UnitName != "MAW") > 0:
        raise Exception("Other unit than MAW")
    #df.quantity = np.where(df.UnitName != "MAW", df.)
    if sum(df.UnitName.isna()) > 0:
        raise Exception("No unit name")

    df['StartTime'] = df.StartTime.str.split('+').str[0]
    df['EndTime'] = df.EndTime.str.split('+').str[0]  
    df['StartTime'] = pd.to_datetime(df.StartTime)
    df['EndTime'] = pd.to_datetime(df.EndTime)

    df = replenish_zeros(df)
    #len([x for x in rng15 if x not in rngdata])

    aggregation_bound = timedelta(hours=1)
    cols = [agg_by,'quantity']
    df_agg = df.loc[df['EndTime'] - df['StartTime'] < aggregation_bound,cols] # dataframe rows that require aggregation
    intervals = df['EndTime'] - df['StartTime']
    #if len(np.unique(intervals).tolist()) > 1:
    #    raise Exception("Varying intervals across the dataset")
    if df.loc[intervals > aggregation_bound,cols].shape[0] > 0:
        raise Exception("Some rows have an interval of more than 1 hour.") # Any rows have a higher

    if df_agg.shape[0] > 0:
        df_good = df.loc[~df.index.isin(df_agg.index),cols]
        df_good.set_index(agg_by,drop=True,inplace=True)
        df_agg = df_agg.resample("H", on = agg_by).sum()
        df = pd.concat([df_good,df_agg])
    else:
        df = df[cols]
        df.set_index(agg_by,drop=True,inplace=True)
    
    if len(np.unique(df.index)) != df.shape[0]:
        df = df.resample("H").sum()
        #val_cnt = df.index.value_counts()
        #weird = val_cnt[val_cnt>1].index.tolist()
        #df[df.index.isin(weird[0:2])]
        #raise Exception("Non-unique indices.")
    df.sort_index(inplace=True)
    
    return df


def create_df(path_main):
    exclude_years = [2021,2023]
    load_files = [x for x in os.listdir(path_main) if 'load' in x.lower()]
    gen_files = sorted([x for x in os.listdir(path_main) if 'gen' in x.lower()])

    # Aggregate load data

    file_no = 0
    for file in load_files:
        col_name = file.split(".")[0]; col_name = col_name.split("_")[1]+"_"+col_name.split("_")[0]
        # file = load_files[0]
        file_path = f"{path_main}/{file}"
        #df = read_data(file_path)
        #test_ts_completeness(df)
        if file_no == 0:
            df = read_data(file_path)
            df_loads = aggregate_load_data(df)
            df_loads.columns = [col_name]
            file_no += 1
        else:
            df_aux = aggregate_load_data(read_data(file_path))
            df_aux.columns = [col_name]
            df_loads = df_loads.join(df_aux,how='outer')
    df_loads = df_loads[~df_loads.index.year.isin(exclude_years)]

    # Aggregate gen data
    file_no = 0
    for file in gen_files:
        _, country, var, = file.split(".")[0].split("_")
        col_name = f"{country}_gen_{var}"
        if var in cnst.green_energy.keys():
            # file = load_files[0]
            file_path = f"{path_main}/{file}"
            if file_no == 0:
                df_gens = aggregate_gen_data(read_data(file_path))
                df_gens.columns = [col_name]
                file_no += 1
            else:
                df_aux = aggregate_gen_data(read_data(file_path))
                df_aux.columns = [col_name]
                df_gens = df_gens.join(df_aux,how='outer')
    df_gens = df_gens[~df_gens.index.year.isin(exclude_years)]

    return df_loads.join(df_gens,how='outer')


def replenish_zeros(df):
    
    col = 'quantity' if 'quantity' in df.columns.tolist() else 'Load'
    if df[col].isna().sum()>0:
        raise Exception("Found some NAs")
    
    # if sum(df.quantity==0)>0:
    #     raise Exception("Found some zeros")
    
    #     idxs = df[df.quantity==0].index
    #     df[(df.index > idxs[0]-3) & (df.index < idxs[0]+3)]
    
    return df