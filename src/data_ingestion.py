import argparse
import datetime
import numpy as np
import pandas as pd
import constants as cnst
#from functions import path_exists
from utils import perform_get_request, xml_to_load_dataframe, xml_to_gen_data

def get_load_data_from_entsoe(regions,periodStart='202302240000', periodEnd='202303240000', output_path='./data'):
    
    # TODO: There is a period range limit of 1 year for this API. Process in 1 year chunks if needed

    # URL of the RESTful API
    url = 'https://web-api.tp.entsoe.eu/api'

    years = np.unique([int(periodStart[:4]),int(periodEnd[:4])]).tolist()

    # General parameters for the API
    # Refer to https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html#_documenttype
    params = {
        'securityToken': '1d9cd4bd-f8aa-476c-8cc1-3442dc91506d',
        'documentType': 'A65',
        'processType': 'A16',
        'outBiddingZone_Domain': 'FILL_IN', # used for Load data
        'periodStart': 'FILL_IN', # in the format YYYYMMDDHHMM
        'periodEnd': 'FILL_IN' # in the format YYYYMMDDHHMM
    }

    # Loop through the regions and get data for each region
    for region, area_code in regions.items():
        print(f'Fetching data for {region}...')
        params['outBiddingZone_Domain'] = area_code
          
        periodStart_aux = periodStart
        for year in range(min(years),max(years)+1):
            if year < max(years):
                periodEnd_aux = datetime.datetime(year+1, 1, 1).strftime('%Y%m%d%H%M')
            else:
                periodEnd_aux = periodEnd
            
            params['periodStart'] = periodStart_aux
            params['periodEnd'] = periodEnd_aux

            # Use the requests library to get data from the API for the specified time range
            response_content = perform_get_request(url, params)
            
            if year == min(years):
                df = xml_to_load_dataframe(response_content)
            else:
                df = pd.concat([df,xml_to_load_dataframe(response_content)])
                df.reset_index(drop=True,inplace=True)

            periodStart_aux = periodEnd_aux
        # Save the DataFrame to a CSV file
        df.to_csv(f'{output_path}/load_{region}.csv', index=False)

    return

def get_gen_data_from_entsoe(regions, periodStart='202302240000', periodEnd='202303240000', output_path='./data'):
    
    # TODO: There is a period range limit of 1 day for this API. Process in 1 day chunks if needed
    
    #years = np.unique([int(periodStart[:4]),int(periodEnd[:4])]).tolist()

    # URL of the RESTful API
    url = 'https://web-api.tp.entsoe.eu/api'

    # General parameters for the API
    params = {
        'securityToken': 'fb81432a-3853-4c30-a105-117c86a433ca',
        'documentType': 'A75',
        'processType': 'A16',
        'outBiddingZone_Domain': 'FILL_IN', # used for Load data
        'in_Domain': 'FILL_IN', # used for Generation data
        'periodStart': 'FILL_IN', # in the format YYYYMMDDHHMM
        'periodEnd': 'FILL_IN' # in the format YYYYMMDDHHMM
    }

    # Loop through the regions and get data for each region
    for region, area_code in regions.items():
        print(f'Fetching data for {region}...')
        params['outBiddingZone_Domain'] = area_code
        params['in_Domain'] = area_code
        
        # Create a list of day-tuples
        days_list = pd.date_range(periodStart,periodEnd,freq='d').strftime('%Y%m%d%H%M')
        days_list = [(days_list[ii],days_list[ii+1]) for ii in range(len(days_list)-1)]
        
        iter = 1
        for periodStart_aux, periodEnd_aux in days_list:
            params['periodStart'] = periodStart_aux
            params['periodEnd'] = periodEnd_aux

            # Use the requests library to get data from the API for the specified time range
            response_content = perform_get_request(url, params)

            # Response content is a string of XML data
            if (periodStart_aux, periodEnd_aux) == days_list[0]:
                dfs = xml_to_gen_data(response_content)
                dfs = {key: [dfs[key]] for key in dfs.keys()}
            else:
                dfs_tmp = xml_to_gen_data(response_content)
                for key in dfs_tmp.keys():
                    if key in dfs.keys():
                        dfs[key].append(dfs_tmp[key])
                    elif key in cnst.green_energy.keys():
                        dfs[key] = [dfs_tmp[key]]
            print(f"Ukończono {iter} z {len(days_list)}")
            iter += 1

        for key in dfs.keys():
            dfs[key] = pd.concat(dfs[key])   
            dfs[key].reset_index(drop=True,inplace=True)

        # Save the dfs to CSV files
        for psr_type, df in dfs.items():
            # Save the DataFrame to a CSV file
            df.to_csv(f'{output_path}/gen_{region}_{psr_type}.csv', index=False)
    
    return


def parse_arguments():
    parser = argparse.ArgumentParser(description='Data ingestion script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--start_time', 
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), 
        default=datetime.datetime(2022, 1, 1), 
        help='Start time for the data to download, format: YYYY-MM-DD'
    )
    parser.add_argument(
        '--end_time', 
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), 
        default=datetime.datetime(2023, 1, 1), 
        help='End time for the data to download, format: YYYY-MM-DD'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='./data',
        help='Name of the output file'
    )
    return parser.parse_args()

def main(start_time, end_time, output_path):
    
    regions = {
         'HU': '10YHU-MAVIR----U',
         'IT': '10YIT-GRTN-----B',
         'PO': '10YPL-AREA-----S',
         'SP': '10YES-REE------0',
         'UK': '10Y1001A1001A92E',
         'DE': '10Y1001A1001A83F',
         'DK': '10Y1001A1001A65H',
         'SE': '10YSE-1--------K',
         'NL': '10YNL----------L',
    }

    # Transform start_time and end_time to the format required by the API: YYYYMMDDHHMM
    start_time = start_time.strftime('%Y%m%d%H%M')
    end_time = end_time.strftime('%Y%m%d%H%M')
    
    # Get Load data from ENTSO-E
    get_load_data_from_entsoe(regions, start_time, end_time, output_path)

    # Get Generation data from ENTSO-E
    get_gen_data_from_entsoe(regions, start_time, end_time, output_path)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.start_time, args.end_time, args.output_path)