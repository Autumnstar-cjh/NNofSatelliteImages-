import pandas as pd
import censusdata # to download ACS data
import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

with open('us_state_ct_raw.pickle', 'rb') as f:
    us_state_ct = pickle.load(f)

with open('var_names.pickle', 'rb') as f:
    var_names = pickle.load(f)


us_state_ct['36'].describe()

# Note that NAs are marked as -99999999. And many zeros exist in pop, inc, etc.
# For example in NY

# checking negative
epsilon = -1000
for var in var_names:
    print("Count of negative and NA ", var, np.sum(us_state_ct['36'][var] < epsilon))

print()

epsilon = 0.0001
for var in var_names:
    print("Count of negative, zero, and NA ", var, np.sum(us_state_ct['36'][var] < epsilon))

# data imputation.
from sklearn.impute import KNNImputer

for key_ in us_state_ct.keys():
    # Use only the cts with none-zero population.
    us_state_ct[key_] = us_state_ct[key_].loc[us_state_ct[key_].pop_total > 0.0001, :].reset_index(drop = True)

    # replace the negative values (NA notations) by np.nan.
    # the following four vars are those vars with negative values. However, this list could change.
    var_list_to_replace_negative_values = ['age_median', 'inc_median_household', 'rent_median', 'property_value_median']
    for var in var_list_to_replace_negative_values:
        us_state_ct[key_].loc[us_state_ct[key_][var] < -100, var] = np.nan

    # impute the NAs with KNN.
    imp = KNNImputer(missing_values=np.nan, n_neighbors=5)

    # only impute the numeric values
    imputing_vars = list(us_state_ct[key_].dtypes[us_state_ct[key_].dtypes != 'object'].index)

    # imputing data
    imp.fit(us_state_ct[key_][imputing_vars])
    us_state_ct[key_][imputing_vars] = imp.transform(us_state_ct[key_][imputing_vars])

# Need to use per capita or per household info.
# I need to first lift the denominator variables by one unit to avoid weird inf and nan in division.

var_list_to_be_lifted_by_one = ['pop_total', 'sex_total', 'households', 'race_total',
                                'travel_total_to_work', 'edu_total', 'housing_units_total', 'property_value_total',
                                'vehicle_total_imputed']

for var in var_list_to_be_lifted_by_one:
    for key_ in us_state_ct.keys():
        try:
            us_state_ct[key_].loc[us_state_ct[key_][var] == 0.0, var] += 1.0
        except KeyError:
            pass



for key_ in us_state_ct.keys():
    us_state_ct[key_]['household_size_avg'] = us_state_ct[key_]['pop_total']/us_state_ct[key_]['households']
    us_state_ct[key_]['sex_male_ratio'] = us_state_ct[key_]['sex_male']/us_state_ct[key_]['sex_total']
    us_state_ct[key_]['race_white_ratio'] = us_state_ct[key_]['race_white']/us_state_ct[key_]['race_total']
    us_state_ct[key_]['race_black_ratio'] = us_state_ct[key_]['race_black']/us_state_ct[key_]['race_total']
    us_state_ct[key_]['race_native_ratio'] = us_state_ct[key_]['race_native']/us_state_ct[key_]['race_total']
    us_state_ct[key_]['race_asian_ratio'] = us_state_ct[key_]['race_asian']/us_state_ct[key_]['race_total']
    us_state_ct[key_]['travel_driving_ratio'] = us_state_ct[key_]['travel_driving_to_work']/us_state_ct[key_]['travel_total_to_work']
    us_state_ct[key_]['travel_pt_ratio'] = us_state_ct[key_]['travel_pt_to_work']/us_state_ct[key_]['travel_total_to_work']
    us_state_ct[key_]['travel_taxi_ratio'] = us_state_ct[key_]['travel_taxi_to_work']/us_state_ct[key_]['travel_total_to_work']
    us_state_ct[key_]['travel_cycle_ratio'] = us_state_ct[key_]['travel_cycle_to_work']/us_state_ct[key_]['travel_total_to_work']
    us_state_ct[key_]['travel_walk_ratio'] = us_state_ct[key_]['travel_walk_to_work']/us_state_ct[key_]['travel_total_to_work']
    us_state_ct[key_]['travel_work_home_ratio'] = us_state_ct[key_]['travel_work_from_home']/us_state_ct[key_]['travel_total_to_work']
    us_state_ct[key_]['edu_bachelor_ratio'] = us_state_ct[key_]['edu_bachelor']/us_state_ct[key_]['edu_total']
    us_state_ct[key_]['edu_master_ratio'] = us_state_ct[key_]['edu_master']/us_state_ct[key_]['edu_total']
    us_state_ct[key_]['edu_phd_ratio'] = us_state_ct[key_]['edu_phd']/us_state_ct[key_]['edu_total']
    us_state_ct[key_]['edu_higher_edu_ratio'] = us_state_ct[key_]['edu_bachelor_ratio'] + us_state_ct[key_]['edu_master_ratio'] + us_state_ct[key_]['edu_phd_ratio']
    us_state_ct[key_]['employment_unemployed_ratio'] = us_state_ct[key_]['employment_unemployed']/us_state_ct[key_]['employment_total_labor']
    us_state_ct[key_]['vehicle_per_capita'] = us_state_ct[key_]['vehicle_total_imputed']/us_state_ct[key_]['pop_total']
    us_state_ct[key_]['vehicle_per_household'] = us_state_ct[key_]['vehicle_total_imputed']/us_state_ct[key_]['households']
    us_state_ct[key_]['vacancy_ratio'] = us_state_ct[key_]['housing_units_vacant']/us_state_ct[key_]['housing_units_total']

# check
us_state_ct['06'].shape

# check
us_state_ct['12'].shape

with open('us_state_ct_processed.pickle', 'wb') as f:
    pickle.dump(us_state_ct, f)

us_state_ct['25'].to_csv("./Mass_ct.csv")

