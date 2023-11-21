# https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html#_available_parameters
data_api = "https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html"
green_energy = {"B01":"Biomass","B09":"Geothermal","B10":"Hydro Pumped Storage","B11":"Hydro Run-of-river and poundage","B12":"Hydro Water Reservoir",
                "B13":"Marine","B15":"Other renewable","B16":"Solar","B18":"Wind Offshore","B19":"Wind Onshore"}#"B17":"Waste",
# Double check B10
recode_gen = {'Biomass': ['B01'], 'Geothermal': ['B09'], 'Hydro': ['B10', 'B11', 'B12'], 'Marine': ['B13'],
              'Other': ['B15'], 'Solar': ['B16'], 'Wind': ['B18','B19']}

country_ids = {
"SP": 0, # Spain
"UK": 1, # United Kingdom
"DE": 2, # Germany
"DK": 3, # Denmark
"HU": 5, # Hungary
"SE": 4, # Sweden
"IT": 6, # Italy
"PO": 7, # Poland
"NL": 8 # Netherlands
}

# TODO:
# - group by 'StartTime' and by 'hour'
# - impute NAs before aggregating?

# Ostatecznie powinienem mieć 2210 obserwacji
# coś musi być nie tak przy agregacji danych do godziny?
