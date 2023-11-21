import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
#from src import constants as cnst
import constants as cnst

def create_aux_daterng(startdate="1/1/2022",finishdate="1/1/2023",freq="15Min"):
    daterng = [x for x in pd.date_range(start=pd.to_datetime(startdate),end=pd.to_datetime(finishdate),freq=freq)]
    daterng = daterng[:-1]
    return daterng

def read_data(file_path):
    # TODO: Load data from CSV file

    df = pd.read_csv(file_path)
    df.reset_index(drop=True,inplace=True)

    return df

def create_df2(path_main):
    #exclude_years = [2021,2023]
    load_files = [x for x in os.listdir(path_main) if 'load' in x.lower()]
    gen_files = sorted([x for x in os.listdir(path_main) if 'gen' in x.lower()])

    # Aggregate load data
    df_loads = pd.DataFrame(data=None,index=create_aux_daterng(freq="1H"))
    for file in load_files:
        #print(f"--------------{file[5:7]}----------------")
        col_name = file.split(".")[0]; col_name = col_name.split("_")[1]+"_"+col_name.split("_")[0]
        file_path = f"{path_main}/{file}"
        df = read_data(file_path)
        df = aggregate_data2(df)
        df.columns = [col_name]
        df_loads = df_loads.join(df,how='left')

    # Aggregate gen data
    df_gens = pd.DataFrame(data=None,index=create_aux_daterng(freq="1H"))
    for file in gen_files:
        #print(f"--------------{file[4:6]}----------------")
        _, country, var, = file.split(".")[0].split("_")
        col_name = f"{country}_gen_{var}"
        if var in cnst.green_energy.keys():
            file_path = f"{path_main}/{file}"
            df = read_data(file_path)
            df = aggregate_data2(df)
            df.columns = [col_name]
            df_gens = df_gens.join(df,how='left')
        #import time; time.sleep(.2)

    return df_loads.join(df_gens,how='left')

def aggregate_data2(df):
    #agg_by = "StartTime"
    value = 'quantity' if 'quantity' in df.columns else 'Load'

    df['StartTime'] = df.StartTime.str.split('+').str[0]
    df['EndTime'] = df.EndTime.str.split('+').str[0]  
    df['StartTime'] = pd.to_datetime(df.StartTime)
    df['EndTime'] = pd.to_datetime(df.EndTime)

    df['Interval'] = df['EndTime'] - df['StartTime']; df.Interval = df['Interval'].apply(lambda x: f"{(x.seconds/60)/60}").astype(float)
    #df.Interval.value_counts()

    #print(np.unique(df.Interval).tolist())

    df['StartHour'] = df['StartTime'].apply(lambda x: x.replace(minute=0,second=0,microsecond=0))
    
    # Check if all rows have the same UnitName
    if sum(df["UnitName"].astype(str) != "MAW") > 0:
        raise ValueError("Some rows have a unit different than 'MAW'.")
    
    if value =='quantity':
        if len(np.unique(df["PsrType"])) > 1:
            raise ValueError("Varying PsrTypes.")

    #if len(np.unique(df.Interval)) > 1:
    df_list = []

    for inter in np.unique(df.Interval):
        if inter > 1:
            raise Exception("Interval greater than 1 hour.")
        #print(f"Interval: {inter}")
        freq = '1H' if inter * 60 == 60 else f"{int(inter*60)}Min"
        df_aux = df[df.Interval == inter].copy()
        df_perfect = pd.DataFrame(data=None,index=create_aux_daterng(freq=freq))
        df_aux = df_aux[["StartTime",value]].groupby(["StartTime"],as_index=False).sum(min_count=1)
        df_aux.columns = ["StartTime","MySum"]
        #df_aux = df_aux[["StartTime",value]].groupby(["StartTime"],as_index=False).agg(MySum = (value,'sum'))
        df_aux['StartHour'] = df_aux['StartTime'].apply(lambda x: x.replace(minute=0,second=0,microsecond=0))
        
        hours_summary = df_aux[["StartHour","MySum"]].groupby(["StartHour"],as_index=False).agg(MyCount = ("MySum",'count'))
        hours_summary = hours_summary[hours_summary.StartHour.apply(lambda x: x.year) == 2022]
        keep_hours = np.unique(hours_summary.StartHour)
        interpolate_hours = hours_summary[hours_summary.MyCount < int(1/inter)].StartHour.tolist()

        df_aux = df_aux[["StartTime","MySum"]].set_index("StartTime",drop=True)
        df_perfect = df_perfect.join(df_aux,how='left')
        #print(f"Before interpolation - {df_perfect.MySum.isna().sum(axis=0)} missing for interval: {inter}")
        df_perfect['StartHour'] = pd.Series(df_perfect.index).apply(lambda x: x.replace(minute=0,second=0,microsecond=0)).to_numpy()

        for hour in interpolate_hours:
            df_tmp = df_perfect[df_perfect["StartHour"] == hour].drop(['StartHour'],axis=1).copy()
            df_tmp.interpolate(method='linear', limit_direction='both',inplace=True)
            df_perfect.loc[df_perfect.index.isin(df_tmp.index),"MySum"] = df_tmp
        #print(f"After interpolation - {df_perfect.MySum.isna().sum(axis=0)} missing for interval: {inter}")

        df_perfect = df_perfect[df_perfect.StartHour.isin(keep_hours)]
        df_list.append(df_perfect)
    
    df = pd.concat(df_list)
    if sum(df.MySum.isna()) > 0:
        raise Exception("Take a look")
    df = df[["StartHour","MySum"]].groupby(["StartHour"],as_index=True).sum(min_count=1)
    df.columns = ["MySum"]
    #df = df.groupby(["StartHour"],as_index=True).agg(MySum = ('MySum','sum'))
    #print(f"After groupby(hour) - {df.MySum.isna().sum(axis=0)} missing")
    #import time;time.sleep(0.5)
    return df

def check_offsets(df):
    start_tz_offsets = np.unique(df.StartTime.str.split('+').str[-1]).tolist()
    end_tz_offsets = np.unique(df.EndTime.str.split('+').str[-1]).tolist()

    if len(start_tz_offsets) > 1 or len(end_tz_offsets) > 1:
        raise ValueError("Different timezone offsets")
    
    if '00:00' not in start_tz_offsets[0] or '00:00' not in end_tz_offsets[0]:
        raise Exception("There is time offset different than 0.")

def impute_NA(data,cols):
    data.sort_index(inplace=True)

    for col in cols:
        vect = data[col].copy()
        
        na_idxs = vect[vect.isna()].index
        for idx in na_idxs:
            NA_pos = vect.index.get_loc(idx)
                     
            aux1 = vect[:NA_pos+1]
            aux2 = vect[NA_pos:]
            if aux1.isna().sum() < aux1.shape[0] and aux2.isna().sum() < aux2.shape[0]:
                prev_value_idx = pd.DataFrame(aux1).apply(pd.Series.last_valid_index).values[0]
                next_value_idx = pd.DataFrame(aux2).apply(pd.Series.first_valid_index).values[0]
                impute_idxs = pd.Index([prev_value_idx,next_value_idx])
                vect[NA_pos] = vect[vect.index.isin(impute_idxs)].mean()
        
        data[col] = vect
    return data

def concat_similar_gen(df):
    countries = [x for x in cnst.country_ids.keys()]
    for key in cnst.recode_gen.keys():
        vars = cnst.recode_gen[key]
        for country in countries:
            cols = [y for y in df.columns for x in vars if country in y and x in y]
            if len(cols)>0:
                df[f"{country}_gen_{key}"] = df[cols].sum(axis=1,min_count=1)
                df.drop(cols,axis=1,inplace=True)
    return df

def aggregate_gen(df):
    countries = [x for x in cnst.country_ids.keys()]
    agg_gen_cols = [f"{country}_gen" for country in countries]
    for country in countries:
        cols = [y for y in df.columns if country in y and 'gen' in y]
        if len(cols)>0:
            name = [x for x in agg_gen_cols if country in x][0]
            df[name] = df[cols].sum(axis=1,min_count=1)
            df.drop(cols,axis=1,inplace=True)
    return df

def get_gen_surplus_labels(df):
    """Calculate surplus"""
    countries = [x for x in cnst.country_ids.keys()]
    surplus_cols = [f"{x}_surplus" for x in countries]

    df_tmp = aggregate_gen(df.copy())

    for country in countries:
        gen_col = [y for y in df_tmp.columns if country in y and 'gen' in y][0]
        load_col = [y for y in df_tmp.columns if country in y and 'load' in y][0]
        name = [x for x in surplus_cols if country in x][0]
        df_tmp[name] = df_tmp[gen_col] - df_tmp[load_col]
        #df_tmp[name] = np.where((~df_tmp[gen_col].isna()) & (df_tmp[gen_col]!=0) & \
        #                        (~df_tmp[load_col].isna()) & (df_tmp[load_col]!=0)
        #                        ,df_tmp[gen_col] - df_tmp[load_col],np.NaN)

    #df_tmp['label1'] = np.where(df_tmp[surplus_cols].isna().sum(axis=1) == 0,df_tmp[surplus_cols].astype(float).idxmax(axis=1).str[:2],np.NaN)
    df_tmp['label'] = df_tmp[surplus_cols].astype(float).idxmax(axis=1).str[:2]
    
    #import copy
    #checkout = copy.deepcopy(surplus_cols)
    #checkout.extend(["label","label1"])
    #df_tmp[checkout][df_tmp.label1 != df_tmp.label]

    return df.join(df_tmp['label'],how='left')

def backfill_zeros(df,columns):
    df.sort_index(inplace=True)
    df[columns] = df[columns].fillna(df[columns].mask(df[columns].bfill().notna(),0))
    return df

def drop_rows(df):
    load_cols = [x for x in df.columns if 'load' in x]
    countries = [x for x in cnst.country_ids.keys()]
    agg_gen_cols = [f"{country}_gen" for country in countries]

    df_aux = aggregate_gen(df.copy())
    df_aux = df_aux[df_aux[load_cols].isna().sum(axis=1) == 0]
    df_aux = backfill_zeros(df_aux,agg_gen_cols)
    #df_aux[agg_gen_cols] = df_aux[agg_gen_cols].fillna(df_aux[agg_gen_cols].mask(df_aux[agg_gen_cols].bfill().notna(),0))
    df_aux.dropna(inplace=True)
    return df[df.index.isin(df_aux.index)]

def clean_data(df):
    #df = data.copy()
    load_cols = [x for x in df.columns if 'load' in x]
    countries = [x for x in cnst.country_ids.keys()]

    agg_gen_cols = [f"{country}_gen" for country in countries]
    all_cols = load_cols.copy(); all_cols.extend(agg_gen_cols)

    df = concat_similar_gen(df)
    df = drop_rows(df)

    # Fill out gen-specific columns with zeros
    gen_cols = [x for x in df.columns if 'gen' in x]
    df_tmp = df[gen_cols].copy(); df_tmp.fillna(0,inplace=True); df.loc[:,gen_cols] = df_tmp

    # Get labels
    df = get_gen_surplus_labels(df)

    df = aggregate_gen(df)

    # TODO: Handle outliers etc

    return df

def preprocess_data(df):
    # TODO: Generate new features, transform existing features, resampling, etc.
    # Time of day feature
    # Season feature
    #df["green_energy.."]

    df_processed = df

    return df_processed

def split_data(df):
    """Split data into train and test"""
    train = .8
    df.sort_index(inplace=True)
    df_train = df[:round(train*df.shape[0])]
    df_test = df[round(train*df.shape[0]):]

    return df_train, df_test


def save_data(df,path):
    # TODO: Save processed data to a CSV file
    
    df_train, df_test = split_data(df)
    df_train.to_csv(f"{path}/train.csv")
    df_test.to_csv(f"{path}/test.csv")
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/raw_data.csv',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file):
    path_main = './data'

    df = create_df2(path_main)

    df_clean = clean_data(df) # not quite done yet - look into outliers etc.

    df_processed = preprocess_data(df_clean)

    save_data(df_processed, path_main)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file)