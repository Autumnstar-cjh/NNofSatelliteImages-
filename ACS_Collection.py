import pandas as pd
import censusdata
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


var_list = [
# population
            'B01003_001E',
            'B01001_001E', 'B01001_002E', 'B01001_026E',
            'B01002_001E',
# households
            'B11001_001E',
# race
            'B02001_001E', 'B02001_002E', 'B02001_003E', 'B02001_004E', 'B02001_005E',
# income info (a lot of NAs)
            'B06010_001E', 'B06010_002E', 'B06010_003E', 'B06010_004E', 'B06010_005E', 'B06010_006E', 'B06010_007E', 'B06010_008E', 'B06010_009E', 'B06010_010E', 'B06010_011E',
            'B06011_001E',
# travel
#             'B08101_001E', 'B08101_009E', 'B08101_017E', 'B08101_025E', 'B08101_033E', 'B08101_041E', 'B08101_049E',
#             'B08015_001E',
            'B08301_001E', 'B08301_002E', 'B08301_010E', 'B08301_016E', 'B08301_018E', 'B08301_019E', 'B08301_021E',
# education
            'B15001_001E',
            'B15001_017E', 'B15001_018E', 'B15001_025E', 'B15001_026E', 'B15001_033E', 'B15001_034E', 'B15001_041E', 'B15001_042E',
            'B15001_058E', 'B15001_059E', 'B15001_066E', 'B15001_067E', 'B15001_074E', 'B15001_075E', 'B15001_082E', 'B15001_083E',
            'B15003_001E', 'B15003_022E', 'B15003_023E', 'B15003_025E',
# income info (more complete)
            'B19013_001E', 'B19301_001E',
# employement
            'B23025_001E', 'B23025_002E', 'B23025_007E',
# properties
            'B25002_001E', 'B25002_002E', 'B25002_003E',
            'B25064_001E',
#             'B25031_002E', 'B25031_003E', 'B25031_004E', 'B25031_005E', 'B25031_006E', 'B25031_007E',
#             'B25111_002E', 'B25111_003E', 'B25111_004E', 'B25111_005E', 'B25111_006E', 'B25111_007E', 'B25111_008E', 'B25111_009E', 'B25111_010E', 'B25111_011E',
            'B25075_001E', 'B25077_001E',
# imputation
            'B99082_001E'
           ]

var_names = [
# population
             'pop_total',
             'sex_total', 'sex_male', 'sex_female',
             'age_median',
# hosueholds
             'households',
# race
             'race_total', 'race_white', 'race_black', 'race_native', 'race_asian',
# income info (a lot of NAs)
             'inc_total_pop', 'inc_no_pop', 'inc_with_pop', 'inc_pop_10k', 'inc_pop_1k_15k', 'inc_pop_15k_25k', 'inc_pop_25k_35k', 'inc_pop_35k_50k', 'inc_pop_50k_65k', 'inc_pop_65k_75k', 'inc_pop_75k',
             'inc_median_ind',
# travel
#              'travel_total_to_work', 'travel_single_driving_to_work', 'travel_carpool_to_work', 'travel_public_transit_to_work', 'travel_walking_to_work', 'travel_cycling_to_work', 'travel_work_from_home',
#              'vehicle_total',
             'travel_total_to_work', 'travel_driving_to_work', 'travel_pt_to_work', 'travel_taxi_to_work', 'travel_cycle_to_work', 'travel_walk_to_work', 'travel_work_from_home',
# education
             'edu_total_pop',
             'bachelor_male_25_34', 'master_phd_male_25_34', 'bachelor_male_35_44', 'master_phd_male_35_44', 'bachelor_male_45_64', 'master_phd_male_45_64',  'bachelor_male_65_over', 'master_phd_male_65_over',
             'bachelor_female_25_34', 'master_phd_female_25_34', 'bachelor_female_35_44', 'master_phd_female_35_44', 'bachelor_female_45_64', 'master_phd_female_45_64',  'bachelor_female_65_over', 'master_phd_female_65_over',
             'edu_total', 'edu_bachelor', 'edu_master', 'edu_phd',
# income info (more complete)
             'inc_median_household', 'inc_per_capita',
# employement
            'employment_total_labor', 'employment_employed', 'employment_unemployed',
# properties
             'housing_units_total', 'housing_units_occupied', 'housing_units_vacant',
             'rent_median',
#              'rent_0_bedroom', 'rent_1_bedroom', 'rent_2_bedroom', 'rent_3_bedroom', 'rent_4_bedroom', 'rent_5_bedroom',
#              'rent_built_2014', 'rent_built_2010', 'rent_built_2000', 'rent_built_1990', 'rent_built_1980', 'rent_built_1970', 'rent_built_1960', 'rent_built_1950', 'rent_built_1940', 'rent_built_1930',
             'property_value_total', 'property_value_median',
# imputation
            'vehicle_total_imputed'
            ]


# US mainland states and FIPS id.
# Now I see that the FIPS order follows the alphabet
# Check: https://www.usgs.gov/faqs/what-constitutes-united-states-what-are-official-definitions
# Check: https://www.mercercountypa.gov/dps/state_fips_code_listing.htm
# I use only the "Conterminous United States" - excluding Alaska and Hawaii. - total 49 states (including DC)
states_and_fips = {
       '01':'AL', # Alabama
#        '02':'AK', # Alaska
       '04':'AZ',
       '05':'AR', # Arkansas
       '06':'CA',
       '08':'CO',
       '09':'CT',
       '10':'DE',
       '11':'DC',
       '12':'FL',
       '13':'GA',
#        '15':'HI', # Hawaii
       '16':'ID', # IDAHO
       '17':'IL',
       '18':'IN',
       '19':'IA',
       '20':'KS',
       '21':'KY',
       '22':'LA',
       '23':'ME',
       '24':'MD',
       '25':'MA',
       '26':'MI',
       '27':'MN',
       '28':'MS',
       '29':'MO',
       '30':'MT',
       '31':'NE',
       '32':'NV',
       '33':'NH',
       '34':'NJ',
       '35':'NM',
       '36':'NY',
       '37':'NC',
       '38':'ND',
       '39':'OH',
       '40':'OK',
       '41':'OR',
       '42':'PA',
       '44':'RI',
       '45':'SC',
       '46':'SD',
       '47':'TN',
       '48':'TX',
       '49':'UT',
       '50':'VT',
       '51':'VA',
       '53':'WA',
       '54':'WV',
       '55':'WI',
       '56':'WY'
}

# print
print(len(states_and_fips.keys()))

# download data for the 49 states.
starting_time = time.time()
us_state_ct = {}

for key_ in states_and_fips.keys():
    print(key_)
    state_ct = censusdata.download('acs5', 2019, censusdata.censusgeo([('state', key_), ('tract', '*')]), var_list)
    us_state_ct[key_] = state_ct

# check the time
end_time = time.time()
processing_time = end_time-starting_time
print(processing_time, ' seconds') # about 3 min.


# add the FIPS info. Change the idx.
def add_fips(df_state):
    state_fips = []
    county_fips = []
    tract_fips = []
    full_ct_fips = []

    for i in range(df_state.shape[0]):
        state_fips.append(df_state.index[i].params()[0][1])
        county_fips.append(df_state.index[i].params()[1][1])
        tract_fips.append(df_state.index[i].params()[2][1])
        full_ct_fips.append(df_state.index[i].params()[0][1]
                            + df_state.index[i].params()[1][1]
                            + df_state.index[i].params()[2][1])

    df_state['state_fips'] = state_fips
    df_state['county_fips'] = county_fips
    df_state['tract_fips'] = tract_fips
    df_state['full_ct_fips'] = full_ct_fips

    df_state.reset_index(drop=True, inplace=True)
    return df_state

# quick preprocessing: adding state name, replacing column names, and adding FIPS info.
for key_ in us_state_ct.keys():
    us_state_ct[key_].columns = var_names
    us_state_ct[key_]['state'] = states_and_fips[key_]
    us_state_ct[key_] = add_fips(us_state_ct[key_])

# check the na, MA example
pd.set_option('display.max_rows', 500)
print(np.sum(us_state_ct['25'].isna()))
pd.set_option('display.max_rows', 10)

for key_ in us_state_ct.keys():
    print(key_, us_state_ct[key_].shape)

with open('us_state_ct_raw.pickle', 'wb') as f:
    pickle.dump(us_state_ct, f)

with open('var_names.pickle', 'wb') as f:
    pickle.dump(var_names, f)

with open('states_and_fips.pickle', 'wb') as f:
    pickle.dump(states_and_fips, f)
