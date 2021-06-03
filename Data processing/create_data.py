import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler
import flow_data as fd

# Returns base_characteristics, body_mass_6
def base_characteristics():
    base_charac = pd.read_csv("Data/widetable_basecharacteristics_full.csv")

    ids = base_charac.iloc[1:,:11] # Basic features
    ids.columns = ids.iloc[0] 
    ids = ids[1:]
    ids.drop(["roottable_intervention_group_name_item", "roottable_birth_year_value","roottable_height_cm_value","roottable_pre_pai_est_value", "roottable_screening_mean_right_sbp_value","roottable_screening_mean_right_dbp_value","roottable_screening_mean_left_sbp_value", "roottable_screening_mean_left_dbp_value", "roottable_age_value","roottable_sex_item"], axis=1, inplace=True)

    body_mass_6 = base_charac.iloc[:,11:] # Features with pre and post tests
    first_row = body_mass_6.iloc[0,:]

    drop_indices = [] # Columns with irrelevant tests
    for i in range(len(first_row)):
        if  first_row[i] == "Post-test day 1" or first_row[i] == "Post-test day 2" or first_row[i] == "Post-test day 3" or first_row[i] == "Pre-test day 1" or first_row[i] == "Pre-test day 2" or first_row[i] == "Pre-test day 3" :
            drop_indices.append(i)
    body_mass_6.drop(body_mass_6.columns[drop_indices], axis=1, inplace=True)
    body_mass_6.drop(["clinical_visits_body_mass_index_value", "clinical_visits_bsa_out_text","clinical_visits_ecw_ratio_value","clinical_visits_height_value","clinical_visits_pre_vfa_value","clinical_visits_pre_whr_value","clinical_visits_skeletal_muscle_mass_value","clinical_visits_test_day_date_date","clinical_visits_waist_circ_cm1_value","clinical_visits_waist_circ_cm2_value","clinical_visits_waist_circ_cm_value"], axis=1, inplace=True)
    body_mass_6 = body_mass_6.iloc[2:,:]
    body_mass_6["roottable_case_id_text"] = ids
    body_mass_6.columns = ['clinical_visits_mid_body_mass_value','roottable_case_id_text']
    body_mass_6.roottable_case_id_text = pd.to_numeric(body_mass_6.roottable_case_id_text)
    body_mass_6.clinical_visits_mid_body_mass_value = pd.to_numeric(body_mass_6.clinical_visits_mid_body_mass_value)

    base_charac1 = base_charac.iloc[1:,:11] #Basic features
    base_charac1.columns = base_charac1.iloc[0] 
    base_charac1 = base_charac1[1:]
    base_charac1.drop(["roottable_intervention_group_name_item", "roottable_birth_year_value","roottable_height_cm_value","roottable_pre_pai_est_value", "roottable_screening_mean_right_sbp_value","roottable_screening_mean_right_dbp_value","roottable_screening_mean_left_sbp_value", "roottable_screening_mean_left_dbp_value"], axis=1, inplace=True)

    base_charac2 = base_charac.iloc[:,11:] #Features with pre and post tests
    first_row = base_charac2.iloc[0,:]

    drop_indices = [] #Columns with irrelevant tests
    for i in range(len(first_row)):
        if first_row[i] == "Mid-test" or first_row[i] == "Post-test day 1" or first_row[i] == "Post-test day 2" or first_row[i] == "Post-test day 3" or first_row[i] == "Pre-test day 1" or first_row[i] == "Pre-test day 3" :
            drop_indices.append(i)

    base_charac2.drop(base_charac2.columns[drop_indices], axis=1, inplace=True)
    base_charac2.columns = ["clinical_visits_body_mass_index_value","clinical_visits_body_mass_value","clinical_visits_bsa_out_text", "clinical_visits_ecw_ratio_value", "clinical_visits_height_value","clinical_visits_pre_vfa_value","clinical_visits_pre_whr_value","clinical_visits_skeletal_muscle_mass_value","clinical_visits_test_day_date_date","clinical_visits_waist_circ_cm1_value","clinical_visits_waist_circ_cm2_value","clinical_visits_waist_circ_cm_value"] #Set column names
    base_charac2.drop(["clinical_visits_ecw_ratio_value","clinical_visits_skeletal_muscle_mass_value","clinical_visits_bsa_out_text","clinical_visits_test_day_date_date","clinical_visits_waist_circ_cm1_value", "clinical_visits_waist_circ_cm2_value","clinical_visits_waist_circ_cm_value","clinical_visits_pre_whr_value","clinical_visits_pre_vfa_value"], axis=1, inplace=True) #Drop irrelevant columns
    base_charac2.drop([0,1], axis=0, inplace=True)

    ids = base_charac1.iloc[:,0]
    base_charac2.loc[:,"roottable_case_id_text"] = ids

    base_characteristics = pd.merge(base_charac1, base_charac2, how="inner", on=["roottable_case_id_text"]) #Merge on patient ID

    # Change sex column from string to 0/1 = male/female
    for i in range(base_characteristics.shape[0]):
        if base_characteristics.iloc[i,2] == "Male":
            base_characteristics.iloc[i,2] = 0
        elif base_characteristics.iloc[i,2] == "Female":
            base_characteristics.iloc[i,2] = 1

    return base_characteristics, body_mass_6

# Returns main, mid, post
def main_analysis():
    main = pd.read_csv("Data/widetable_mainanalysis_full.csv")

    # Find mid values
    first_row = main.iloc[0,:]
    drop_indices = [] # Columns with irrelevant tests
    for i in range(len(first_row)):
        if first_row[i] == "Post-test day 1" or first_row[i] == "Post-test day 2" or first_row[i] == "Post-test day 3" or first_row[i] == "Pre-test day 1" or first_row[i] == "Pre-test day 2" or first_row[i] == "Pre-test day 3" :
            drop_indices.append(i)
    mid = main.drop(main.columns[drop_indices], axis=1)
    mid.columns = ["patient_id","group","clinical_visits_dbp1_hr_value","clinical_visits_dbp2_hr_value","clinical_visits_dbp3_hr_value","clinical_visits_dbp4_hr_value","clinical_visits_dbp5_hr_value","clinical_visits_mean_dbp_text","clinical_visits_mean_sbp_text","clinical_visits_mid_24h_dbp_mean_value",'clinical_visits_pre_24h_dbp_sd_value','clinical_visits_pre_24h_hr_mean_value','clinical_visits_pre_24h_hr_sd_value','clinical_visits_pre_24h_map_sd_text','clinical_visits_pre_24h_map_value','clinical_visits_pre_24h_pp_mean_value','clinical_visits_pre_24h_pp_sd_text','clinical_visits_mid_24h_sbp_mean_value','clinical_visits_pre_24h_sbp_sd_value','clinical_visits_pre_abpm_24h_ok_text','clinical_visits_pre_abpm_24h_samples_value','clinical_visits_pre_abpm_asleep_ok_text','clinical_visits_pre_abpm_asleep_samples_value','clinical_visits_pre_abpm_asleep_time_time','clinical_visits_pre_abpm_awake_ok_text','clinical_visits_pre_abpm_awake_samples_value','clinical_visits_pre_abpm_awake_time_time','clinical_visits_pre_asleep_dbp_mean_value','clinical_visits_pre_asleep_dbp_sd_value','clinical_visits_pre_asleep_hr_mean_value','clinical_visits_pre_asleep_hr_sd_value','clinical_visits_pre_asleep_map_mean_value','clinical_visits_pre_asleep_map_sd_text','clinical_visits_pre_asleep_pp_mean_value','clinical_visits_pre_asleep_pp_sd_text','clinical_visits_pre_asleep_sbp_mean_value','clinical_visits_pre_asleep_sbp_sd_value','clinical_visits_pre_awake_dbp_mean_value','clinical_visits_pre_awake_dbp_sd_value','clinical_visits_pre_awake_hr_mean_value','clinical_visits_pre_awake_hr_sd_value','clinical_visits_pre_awake_map_mean_value','clinical_visits_pre_awake_map_sd_text','clinical_visits_pre_awake_pp_mean_value','clinical_visits_pre_awake_pp_sd_text','clinical_visits_pre_awake_sbp_mean_value','clinical_visits_pre_awake_sbp_sd_value','clinical_visits_pre_pvw_entrytable_tabledata_hr_car','clinical_visits_pre_pvw_entrytable_tabledata_hr_fem','clinical_visits_pre_pvw_entrytable_tabledata_pvw_vel','clinical_visits_pre_pvw_entrytable_tabledata_pwv_std','clinical_visits_pre_pwv_cca_ssn_value','clinical_visits_pre_pwv_distance_value','clinical_visits_pre_pwv_ssn_cfa_value','clinical_visits_time_meal_time', 'clinical_visits_time_out_text','clinical_visits_time_time_time']
    mid.drop(["group","clinical_visits_dbp1_hr_value","clinical_visits_dbp2_hr_value","clinical_visits_dbp3_hr_value","clinical_visits_dbp4_hr_value","clinical_visits_dbp5_hr_value","clinical_visits_mean_dbp_text","clinical_visits_mean_sbp_text","clinical_visits_pre_24h_dbp_sd_value","clinical_visits_pre_24h_hr_mean_value","clinical_visits_pre_24h_hr_sd_value","clinical_visits_pre_24h_map_sd_text","clinical_visits_pre_24h_map_value","clinical_visits_pre_24h_pp_mean_value","clinical_visits_pre_24h_pp_sd_text","clinical_visits_pre_24h_sbp_sd_value","clinical_visits_pre_abpm_24h_ok_text","clinical_visits_pre_abpm_24h_samples_value",'clinical_visits_pre_abpm_asleep_ok_text','clinical_visits_pre_abpm_asleep_samples_value','clinical_visits_pre_abpm_asleep_time_time','clinical_visits_pre_abpm_awake_ok_text','clinical_visits_pre_abpm_awake_samples_value','clinical_visits_pre_abpm_awake_time_time','clinical_visits_pre_asleep_dbp_mean_value','clinical_visits_pre_asleep_dbp_sd_value','clinical_visits_pre_asleep_hr_mean_value','clinical_visits_pre_asleep_hr_sd_value','clinical_visits_pre_asleep_map_mean_value','clinical_visits_pre_asleep_map_sd_text','clinical_visits_pre_asleep_pp_mean_value','clinical_visits_pre_asleep_pp_sd_text','clinical_visits_pre_asleep_sbp_mean_value','clinical_visits_pre_asleep_sbp_sd_value','clinical_visits_pre_awake_dbp_mean_value','clinical_visits_pre_awake_dbp_sd_value','clinical_visits_pre_awake_hr_mean_value','clinical_visits_pre_awake_hr_sd_value','clinical_visits_pre_awake_map_mean_value','clinical_visits_pre_awake_map_sd_text','clinical_visits_pre_awake_pp_mean_value','clinical_visits_pre_awake_pp_sd_text','clinical_visits_pre_awake_sbp_mean_value','clinical_visits_pre_awake_sbp_sd_value','clinical_visits_pre_pvw_entrytable_tabledata_hr_car','clinical_visits_pre_pvw_entrytable_tabledata_hr_fem','clinical_visits_pre_pvw_entrytable_tabledata_pvw_vel','clinical_visits_pre_pvw_entrytable_tabledata_pwv_std','clinical_visits_pre_pwv_cca_ssn_value','clinical_visits_pre_pwv_distance_value','clinical_visits_pre_pwv_ssn_cfa_value','clinical_visits_time_meal_time', 'clinical_visits_time_out_text','clinical_visits_time_time_time'], axis=1, inplace=True)
    mid = mid.iloc[2:,:]
    mid.patient_id = pd.to_numeric(mid.patient_id)
    mid.clinical_visits_mid_24h_dbp_mean_value = pd.to_numeric(mid.clinical_visits_mid_24h_dbp_mean_value)
    mid.clinical_visits_mid_24h_sbp_mean_value = pd.to_numeric(mid.clinical_visits_mid_24h_sbp_mean_value)

    # Find post values
    first_row = main.iloc[0,:]
    drop_indices = [] # Columns with irrelevant tests
    for i in range(len(first_row)):
        if first_row[i] == "Mid-test" or first_row[i] == "Post-test day 1" or first_row[i] == "Post-test day 3" or first_row[i] == "Pre-test day 1" or first_row[i] == "Pre-test day 2" or first_row[i] == "Pre-test day 3" :
            drop_indices.append(i)
    post = main.drop(main.columns[drop_indices], axis=1)
    post.columns = ["patient_id", "group","clinical_visits_cpet_vo2max_value",	"clinical_visits_dbp1_hr_value",	"clinical_visits_dbp2_hr_value","clinical_visits_dbp3_hr_value","clinical_visits_dbp4_hr_value",	"clinical_visits_dbp5_hr_value","clinical_visits_mean_dbp_text","clinical_visits_mean_sbp_text","clinical_visits_post_24h_dbp_mean_value","clinical_visits_pre_24h_dbp_sd_value","clinical_visits_pre_24h_hr_mean_value","clinical_visits_pre_24h_hr_sd_value","clinical_visits_pre_24h_map_sd_text","clinical_visits_pre_24h_map_value",	"clinical_visits_pre_24h_pp_mean_value","clinical_visits_pre_24h_pp_sd_text","clinical_visits_post_24h_sbp_mean_value","clinical_visits_pre_24h_sbp_sd_value","clinical_visits_pre_abpm_24h_ok_text",	"clinical_visits_pre_abpm_24h_samples_value",	"clinical_visits_pre_abpm_asleep_ok_text","clinical_visits_pre_abpm_asleep_samples_value","clinical_visits_pre_abpm_asleep_time_time","clinical_visits_pre_abpm_awake_ok_text","clinical_visits_pre_abpm_awake_samples_value",	"clinical_visits_pre_abpm_awake_time_time","clinical_visits_pre_asleep_dbp_mean_value","clinical_visits_pre_asleep_dbp_sd_value","clinical_visits_pre_asleep_hr_mean_value","clinical_visits_pre_asleep_hr_sd_value","clinical_visits_pre_asleep_map_mean_value",	"clinical_visits_pre_asleep_map_sd_text","clinical_visits_pre_asleep_pp_mean_value","clinical_visits_pre_asleep_pp_sd_text",	"clinical_visits_pre_asleep_sbp_mean_value","clinical_visits_pre_asleep_sbp_sd_value","clinical_visits_pre_awake_dbp_mean_value",	"clinical_visits_pre_awake_dbp_sd_value","clinical_visits_pre_awake_hr_mean_value","clinical_visits_pre_awake_hr_sd_value",	"clinical_visits_pre_awake_map_mean_value","clinical_visits_pre_awake_map_sd_text","clinical_visits_pre_awake_pp_mean_value",	"clinical_visits_pre_awake_pp_sd_text","clinical_visits_pre_awake_sbp_mean_value","clinical_visits_pre_awake_sbp_sd_value",	"clinical_visits_pre_pvw_entrytable_tabledata_hr_car","clinical_visits_pre_pvw_entrytable_tabledata_hr_fem",	"clinical_visits_pre_pvw_entrytable_tabledata_pvw_vel","clinical_visits_pre_pvw_entrytable_tabledata_pwv_std","clinical_visits_pre_pwv_cca_ssn_value","clinical_visits_pre_pwv_distance_value","clinical_visits_pre_pwv_ssn_cfa_value",	"clinical_visits_time_meal_time","clinical_visits_time_out_text","clinical_visits_time_time_time"]
    post.drop(["group","clinical_visits_cpet_vo2max_value","clinical_visits_dbp1_hr_value",	"clinical_visits_dbp2_hr_value","clinical_visits_dbp3_hr_value","clinical_visits_dbp4_hr_value","clinical_visits_pre_24h_dbp_sd_value",	"clinical_visits_dbp5_hr_value","clinical_visits_mean_dbp_text","clinical_visits_mean_sbp_text","clinical_visits_pre_24h_dbp_sd_value","clinical_visits_pre_24h_hr_mean_value","clinical_visits_pre_24h_sbp_sd_value",	"clinical_visits_pre_24h_hr_sd_value","clinical_visits_pre_24h_map_sd_text","clinical_visits_pre_24h_map_value","clinical_visits_pre_24h_pp_mean_value","clinical_visits_pre_24h_pp_sd_text","clinical_visits_pre_24h_sbp_sd_value","clinical_visits_pre_abpm_24h_ok_text","clinical_visits_pre_abpm_24h_samples_value","clinical_visits_pre_abpm_asleep_ok_text","clinical_visits_pre_abpm_asleep_samples_value","clinical_visits_pre_abpm_asleep_time_time","clinical_visits_pre_abpm_awake_ok_text","clinical_visits_pre_abpm_awake_samples_value",	"clinical_visits_pre_abpm_awake_time_time","clinical_visits_pre_asleep_dbp_mean_value","clinical_visits_pre_asleep_dbp_sd_value","clinical_visits_pre_asleep_hr_mean_value","clinical_visits_pre_asleep_hr_sd_value","clinical_visits_pre_asleep_map_mean_value","clinical_visits_pre_asleep_map_sd_text","clinical_visits_pre_asleep_pp_mean_value","clinical_visits_pre_asleep_pp_sd_text",	"clinical_visits_pre_asleep_sbp_mean_value","clinical_visits_pre_asleep_sbp_sd_value","clinical_visits_pre_awake_dbp_mean_value","clinical_visits_pre_awake_dbp_sd_value","clinical_visits_pre_awake_hr_mean_value","clinical_visits_pre_awake_hr_sd_value","clinical_visits_pre_awake_map_mean_value","clinical_visits_pre_awake_map_sd_text","clinical_visits_pre_awake_pp_mean_value","clinical_visits_pre_awake_pp_sd_text","clinical_visits_pre_awake_sbp_mean_value","clinical_visits_pre_awake_sbp_sd_value",	"clinical_visits_pre_pvw_entrytable_tabledata_hr_car","clinical_visits_pre_pvw_entrytable_tabledata_hr_fem",	"clinical_visits_pre_pvw_entrytable_tabledata_pvw_vel","clinical_visits_pre_pvw_entrytable_tabledata_pwv_std","clinical_visits_pre_pwv_cca_ssn_value","clinical_visits_pre_pwv_distance_value","clinical_visits_pre_pwv_ssn_cfa_value",	"clinical_visits_time_meal_time","clinical_visits_time_out_text","clinical_visits_time_time_time"], axis=1, inplace=True) #Drop irrelevant columns
    post = post.iloc[2:,:]
    post.patient_id = pd.to_numeric(post.patient_id)
    post.clinical_visits_post_24h_dbp_mean_value = pd.to_numeric(post.clinical_visits_post_24h_dbp_mean_value)
    post.clinical_visits_post_24h_sbp_mean_value = pd.to_numeric(post.clinical_visits_post_24h_sbp_mean_value)

    first_row = main.iloc[0,:]
    drop_indices = [] # Columns with irrelevant tests
    for i in range(len(first_row)):
        if first_row[i] == "Mid-test" or first_row[i] == "Post-test day 1" or first_row[i] == "Post-test day 2" or first_row[i] == "Post-test day 3" or first_row[i] == "Pre-test day 1" or first_row[i] == "Pre-test day 3" :
            drop_indices.append(i)
    main.drop(main.columns[drop_indices], axis=1, inplace=True)

    main.columns = ["roottable_case_id_text","group_name","clinical_visits_cpet_vo2max_value","clinical_visits_dbp1_hr_value","clinical_visits_dbp2_hr_value","clinical_visits_dbp3_hr_value","clinical_visits_dbp4_hr_value","clinical_visits_mean_dbp_text","clinical_visits_mean_sbp_text","clinical_visits_pre_24h_dbp_mean_value","clinical_visits_pre_24h_dbp_sd_value","clinical_visits_pre_24h_hr_mean_value","clinical_visits_pre_24h_hr_sd_value","clinical_visits_pre_24h_map_sd_text","clinical_visits_pre_24h_map_value","clinical_visits_pre_24h_pp_mean_value", "clinical_visits_pre_24h_pp_sd_text", "clinical_visits_pre_24h_sbp_mean_value","clinical_visits_pre_24h_sbp_sd_value","clinical_visits_pre_abpm_24h_ok_text","clinical_visits_pre_abpm_24h_samples_value","clinical_visits_pre_abpm_asleep_ok_text","clinical_visits_pre_abpm_asleep_samples_value","clinical_visits_pre_abpm_asleep_time_time", "clinical_visits_pre_abpm_awake_ok_text", "clinical_visits_pre_abpm_awake_samples_value","clinical_visits_pre_abpm_awake_time_time", "clinical_visits_pre_asleep_dbp_mean_value", "clinical_visits_pre_asleep_dbp_sd_value", "clinical_visits_pre_asleep_hr_mean_value", "clinical_visits_pre_asleep_hr_sd_value", "clinical_visits_pre_asleep_map_mean_value","clinical_visits_pre_asleep_map_sd_text", "clinical_visits_pre_asleep_pp_mean_value","clinical_visits_pre_asleep_pp_sd_text","clinical_visits_pre_asleep_sbp_mean_value", "clinical_visits_pre_asleep_sbp_sd_value","clinical_visits_pre_awake_dbp_mean_value",	"clinical_visits_pre_awake_dbp_sd_value","clinical_visits_pre_awake_hr_mean_value","clinical_visits_pre_awake_hr_sd_value",	"clinical_visits_pre_awake_map_mean_value","clinical_visits_pre_awake_map_sd_text","clinical_visits_pre_awake_pp_mean_value",	"clinical_visits_pre_awake_pp_sd_text","clinical_visits_pre_awake_sbp_mean_value","clinical_visits_pre_awake_sbp_sd_value",	"clinical_visits_pre_pvw_entrytable_tabledata_hr_car","clinical_visits_pre_pvw_entrytable_tabledata_hr_fem",	"clinical_visits_pre_pvw_entrytable_tabledata_pvw_vel","clinical_visits_pre_pvw_entrytable_tabledata_pwv_std","clinical_visits_pre_pwv_cca_ssn_value","clinical_visits_pre_pwv_distance_value","clinical_visits_pre_pwv_ssn_cfa_value",	"clinical_visits_time_meal_time","clinical_visits_time_out_text","clinical_visits_time_time_time"] #Set column names
    main.drop(["group_name","clinical_visits_dbp1_hr_value","clinical_visits_dbp2_hr_value","clinical_visits_dbp3_hr_value","clinical_visits_dbp4_hr_value","clinical_visits_mean_dbp_text","clinical_visits_mean_sbp_text", "clinical_visits_pre_24h_dbp_sd_value","clinical_visits_pre_24h_sbp_sd_value","clinical_visits_pre_abpm_24h_ok_text","clinical_visits_pre_abpm_24h_samples_value","clinical_visits_pre_abpm_asleep_ok_text","clinical_visits_pre_abpm_asleep_samples_value","clinical_visits_pre_abpm_asleep_time_time", "clinical_visits_pre_abpm_awake_ok_text", "clinical_visits_pre_abpm_awake_samples_value","clinical_visits_pre_abpm_awake_time_time", "clinical_visits_pre_24h_map_sd_text","clinical_visits_pre_24h_map_value","clinical_visits_pre_asleep_dbp_mean_value", "clinical_visits_pre_asleep_dbp_sd_value", "clinical_visits_pre_asleep_hr_mean_value", "clinical_visits_pre_asleep_hr_sd_value", "clinical_visits_pre_24h_pp_mean_value", "clinical_visits_pre_24h_pp_sd_text","clinical_visits_pre_asleep_map_mean_value","clinical_visits_pre_asleep_map_sd_text", "clinical_visits_pre_asleep_pp_mean_value","clinical_visits_pre_asleep_pp_sd_text","clinical_visits_pre_asleep_sbp_mean_value", "clinical_visits_pre_asleep_sbp_sd_value","clinical_visits_pre_awake_dbp_mean_value",	"clinical_visits_pre_awake_dbp_sd_value","clinical_visits_pre_awake_hr_mean_value","clinical_visits_pre_awake_hr_sd_value",	"clinical_visits_pre_awake_map_mean_value","clinical_visits_pre_awake_map_sd_text","clinical_visits_pre_awake_pp_mean_value",	"clinical_visits_pre_awake_pp_sd_text","clinical_visits_pre_awake_sbp_mean_value","clinical_visits_pre_awake_sbp_sd_value",	"clinical_visits_pre_24h_hr_mean_value","clinical_visits_pre_24h_hr_sd_value","clinical_visits_pre_pvw_entrytable_tabledata_hr_car","clinical_visits_pre_pvw_entrytable_tabledata_hr_fem",	"clinical_visits_pre_pvw_entrytable_tabledata_pvw_vel","clinical_visits_pre_pvw_entrytable_tabledata_pwv_std","clinical_visits_pre_pwv_cca_ssn_value","clinical_visits_pre_pwv_distance_value","clinical_visits_pre_pwv_ssn_cfa_value",	"clinical_visits_time_meal_time","clinical_visits_time_out_text","clinical_visits_time_time_time"], axis=1, inplace=True) #Drop irrelevant columns
    main.drop([0,1], axis=0, inplace=True)

    return main, mid, post

# Returns quest
def questionnaire():
    quest = pd.read_csv("Data/widetable_screen_postquestionnaire_full.csv")

    quest = quest.filter(regex=r'(roottable_case_id_text|roottable_pre_smoker_item|roottable_pre_snuff_item|roottable_pre_wine_value|roottable_pre_beer_value|roottable_pre_spirits_value|roottable_famhis_htn_item)', axis=1)

    # Change smoker column from string to 0/1/2 = Nei, jeg har aldri røykt/Nei, jeg har sluttet å røyke/Ja, sigaretter av og til (fest/ferie, ikke daglig)
    for i in range(quest.shape[0]):
        if quest.iloc[i,1] == "Nei, jeg har aldri røykt":
            quest.iloc[i,1] = 0
        elif quest.iloc[i,1] == "Nei, jeg har sluttet å røyke":
            quest.iloc[i,1] = 1
        elif quest.iloc[i,1] == "Ja, sigaretter av og til (fest/ferie, ikke daglig)":
            quest.iloc[i,1] = 2

    # Change sniffer column from string to 0/1/2/3 = Nei, aldri/Nei, men jeg har sluttet/Ja, av og til/Ja, daglig
    for i in range(quest.shape[0]):
        if quest.iloc[i,2] == "Nei, aldri":
            quest.iloc[i,2] = 0
        elif quest.iloc[i,2] == "Ja, men jeg har sluttet":
            quest.iloc[i,2] = 1
        elif quest.iloc[i,2] == "Ja, av og til":
            quest.iloc[i,2] = 2
        elif quest.iloc[i,2] == "Ja, daglig":
            quest.iloc[i,2] = 3

    # Change family history of hypertension column from string to 0/1/None = Nei/Ja/Vet ikke
    for i in range(quest.shape[0]):
        if quest.iloc[i,6] == "Nei":
            quest.iloc[i,6] = 0
        elif quest.iloc[i,6] == "Ja":
            quest.iloc[i,6] = 1
        elif quest.iloc[i,6] == "Vet ikke":
            quest.iloc[i,6] = None

    return quest

# Returns clinical
def clinical_visits():
    clinical = pd.read_csv("Data/widetable_clinicalvisits_full.csv")

    ids = clinical[["Unnamed: 0"]]
    hr = clinical.iloc[:,31]
    clinical = pd.merge(ids, hr, how="inner", left_on=ids.index, right_on=hr.index)
    clinical = clinical.iloc[2:,:]
    clinical.columns = ['index','patient_id','clinical_visits_cpre_hrmax_value']
    clinical.drop(['index'],axis=1,inplace=True)
    clinical.patient_id = pd.to_numeric(clinical.patient_id)

    return clinical

# Returns pai_data
def PAI():
    pai_data = pd.read_csv("Data/OutputTablePAI_added_days_final_190620.csv")

    first_row = pai_data.iloc[0,:]
    drop_indices = [] # Columns with irrelevant tests
    for i in range(len(first_row)):
        if first_row[i] == "Baseline" or first_row[i] == "PrePost":
            drop_indices.append(i)
    pai_data.drop(pai_data.columns[drop_indices], axis=1, inplace=True)
    pai_data = pai_data.iloc[2:,:3]
    pai_data.columns = ['patient_id','MeanPAIPerDay_MidPost', 'MeanPAIPerDay_PreMid']
    pai_data.patient_id = pd.to_numeric(pai_data.patient_id)
    pai_data.MeanPAIPerDay_PreMid = pd.to_numeric(pai_data.MeanPAIPerDay_PreMid)
    pai_data.MeanPAIPerDay_MidPost = pd.to_numeric(pai_data.MeanPAIPerDay_MidPost)

    return pai_data

# Returns full_set
def merge_all():
    base = base_characteristics()[0]
    main = main_analysis()[0]
    quest = questionnaire()
    clinical = clinical_visits()
    pai = PAI()

    full_set = pd.merge(base, main, how="inner", on=["roottable_case_id_text"]) #Merge on patient ID
    full_set = full_set.astype({'roottable_case_id_text': 'int'})
    full_set = pd.merge(full_set, quest, how="inner", on=["roottable_case_id_text"]) #Merge on patient ID
    full_set = pd.merge(full_set, clinical, how="left", left_on="roottable_case_id_text", right_on='patient_id')
    full_set.drop(['patient_id'], axis=1, inplace=True)
    full_set = pd.merge(full_set, pai, how="left", left_on="roottable_case_id_text", right_on='patient_id')
    full_set.drop(['patient_id'], axis=1, inplace=True)
    full_set = full_set.fillna(value=np.nan)
    full_set.replace(to_replace=['None'], value=np.nan, inplace=True)

    return full_set

# Returns full_set
def add_post_mid_bp(full_set):
    post = main_analysis()[2]
    mid = main_analysis()[1]
    body_mass = base_characteristics()[1]

    full_set = pd.merge(full_set, post, how="inner", left_on=["roottable_case_id_text"],right_on=["patient_id"])
    full_set.drop(['patient_id'], axis=1, inplace=True)
    full_set = pd.merge(full_set, mid, how="inner", left_on=["roottable_case_id_text"],right_on=["patient_id"])
    full_set.drop(['patient_id'], axis=1, inplace=True)
    full_set = pd.merge(full_set, body_mass, how="inner", on=["roottable_case_id_text"])

    return full_set

# Returns full_set
def add_cycles(full_set):
    ## PRE
    empty_list = [[0]*100] * full_set.shape[0]
    full_set['pre_finger_pressure_cycle'] = empty_list
    for p in [17.0,51.0,59.0,67.0,93.0,144.0,195.0,217.0,227.0,231.0,278.0,367.0,371.0,468.0,503.0,541.0,556.0,570.0,578.0,647.0,711.0,839.0,890.0,905.0,967.0]:
        pre_cycle = pd.read_csv("Data/FP_pre_cycle/Pre_finger_pressure_patient_"+str(p)+".csv", header=None)
        pre_cycle.columns = ['pressure_values']
        pressure_values = pre_cycle['pressure_values'].tolist()
        i = full_set.index[full_set['roottable_case_id_text'] == p].tolist()[0]
        full_set.at[i, 'pre_finger_pressure_cycle'] = pressure_values

    ## MID
    empty_list = [[0]*100] * full_set.shape[0]
    full_set['mid_finger_pressure_cycle'] = empty_list
    for p in [17.0,51.0,59.0,67.0,93.0,144.0,195.0,217.0,227.0,231.0,278.0,367.0,371.0,468.0,503.0,541.0,556.0,578.0,647.0,711.0,839.0,890.0,905.0,967.0]:
        mid_cycle = pd.read_csv("Data/FP_mid_cycle/Mid_finger_pressure_patient_"+str(p)+".csv", header=None)
        mid_cycle.columns = ['pressure_values']
        pressure_values = mid_cycle['pressure_values'].tolist()
        i = full_set.index[full_set['roottable_case_id_text'] == p].tolist()[0]
        full_set.at[i, 'mid_finger_pressure_cycle'] = pressure_values

    ## POST
    empty_list = [[0]*100] * full_set.shape[0]
    full_set['post_finger_pressure_cycle'] = empty_list
    for p in [17.0,51.0,59.0,67.0,144.0,217.0,231.0,278.0,367.0,468.0,503.0,541.0,556.0,570.0,578.0,647.0,839.0,890.0,905.0,967.0]:
        post_cycle = pd.read_csv("Data/FP_post_cycle/Post_finger_pressure_patient_"+str(p)+".csv", header=None)
        post_cycle.columns = ['pressure_values']
        pressure_values = post_cycle['pressure_values'].tolist()
        i = full_set.index[full_set['roottable_case_id_text'] == p].tolist()[0]
        full_set.at[i, 'post_finger_pressure_cycle'] = pressure_values

    return full_set

def addFlow(full_set):
    flow_data = fd.getData()
    pais = fd.findPais(flow_data)
    ids = list(full_set.roottable_case_id_text.values)
    pre_ids, mid_ids, post_ids, pre_flows, mid_flows, post_flows, pre_times, mid_times, post_times = fd.allFlows(ids)
    empty_list = [[0]*100] * full_set.shape[0]
    empty = [0.0] * full_set.shape[0]
    ## PRE
    full_set['pre_flow'] = empty_list
    full_set['true_time_pre_flow'] = empty
    for i in range(len(pre_ids)):
        pat = pre_ids[i]
        flow = pre_flows[i]
        flow_values = flow.tolist()
        ind = full_set.index[full_set['roottable_case_id_text'] == pat].tolist()[0]
        full_set.at[ind, 'pre_flow'] = flow_values
        full_set.at[ind, 'true_time_pre_flow'] = pre_times[i]
    ## MID
    full_set['mid_flow'] = empty_list
    full_set['true_time_mid_flow'] = empty
    for i in range(len(mid_ids)):
        pat = mid_ids[i]
        flow = mid_flows[i]
        flow_values = flow.tolist()
        ind = full_set.index[full_set['roottable_case_id_text'] == pat].tolist()[0]
        full_set.at[ind, 'mid_flow'] = flow_values
        full_set.at[ind, 'true_time_mid_flow'] = mid_times[i]
    ## POST
    full_set['post_flow'] = empty_list
    full_set['true_time_post_flow'] = empty
    for i in range(len(post_ids)):
        pat = post_ids[i]
        flow = post_flows[i]
        flow_values = flow.tolist()
        ind = full_set.index[full_set['roottable_case_id_text'] == pat].tolist()[0]
        full_set.at[ind, 'post_flow'] = flow_values
        full_set.at[ind, 'true_time_post_flow'] = post_times[i]
    return full_set

# Returns full_set
def correct_data_types(full_set):
    full_set["clinical_visits_height_value"] = pd.to_numeric(full_set["clinical_visits_height_value"])
    full_set["clinical_visits_body_mass_value"] = pd.to_numeric(full_set["clinical_visits_body_mass_value"])
    full_set["clinical_visits_body_mass_index_value"] = full_set.clinical_visits_body_mass_value/np.power(full_set.clinical_visits_height_value/100,2)
    full_set["clinical_visits_mid_body_mass_index_value"] = full_set.clinical_visits_mid_body_mass_value/np.power(full_set.clinical_visits_height_value/100,2)
    full_set["roottable_age_value"] = pd.to_numeric(full_set["roottable_age_value"])
    full_set["clinical_visits_cpet_vo2max_value"] = pd.to_numeric(full_set["clinical_visits_cpet_vo2max_value"])
    full_set["clinical_visits_pre_24h_dbp_mean_value"] = pd.to_numeric(full_set["clinical_visits_pre_24h_dbp_mean_value"])
    full_set["clinical_visits_pre_24h_sbp_mean_value"] = pd.to_numeric(full_set["clinical_visits_pre_24h_sbp_mean_value"])
    full_set["roottable_pre_wine_value"] = pd.to_numeric(full_set["roottable_pre_wine_value"])
    full_set["roottable_pre_beer_value"] = pd.to_numeric(full_set["roottable_pre_beer_value"])
    full_set["roottable_pre_spirits_value"] = pd.to_numeric(full_set["roottable_pre_spirits_value"])
    full_set["clinical_visits_cpre_hrmax_value"] = pd.to_numeric(full_set["clinical_visits_cpre_hrmax_value"])

    full_set.drop(['clinical_visits_body_mass_value','clinical_visits_mid_body_mass_value','clinical_visits_height_value','roottable_pre_spirits_value','roottable_famhis_htn_item'], axis=1, inplace=True)

    return full_set

def to_CSV(file):
    #file.to_csv(r'/Users/anineahlsand/iCloud Drive (arkiv)/Documents/Dokumenter/Documents/Skole/NTNU/Master/Code2021/Master2021/Data/full_data_set.csv',index=False)
    file.to_csv(r'/Users/anineahlsand/iCloud Drive (arkiv)/Documents/Dokumenter/Documents/Skole/NTNU/Master/Code2021/Master2021/Data/full_data_set_double.csv',index=False)
    print('File saved')

# Returns full_set_double
def split_week_6(full_set):
    full_set_double_1 = full_set.copy()
    full_set_double_1.drop(['MeanPAIPerDay_MidPost','true_time_post_flow','post_finger_pressure_cycle',
                            'post_flow','clinical_visits_post_24h_dbp_mean_value','clinical_visits_post_24h_sbp_mean_value',
                            'clinical_visits_mid_body_mass_index_value'], axis=1, inplace=True)
    full_set_double_1.columns = ['roottable_case_id_text', 'roottable_age_value', 'roottable_sex_item',
                                'clinical_visits_body_mass_index_value','clinical_visits_cpet_vo2max_value',
                                'clinical_visits_pre_24h_dbp_mean_value','clinical_visits_pre_24h_sbp_mean_value', 
                                'roottable_pre_smoker_item','roottable_pre_snuff_item', 'roottable_pre_wine_value',
                                'roottable_pre_beer_value', 'clinical_visits_cpre_hrmax_value','MeanPAIPerDay', 
                                'clinical_visits_post_24h_dbp_mean_value','clinical_visits_post_24h_sbp_mean_value',
                                'pre_finger_pressure_cycle','post_finger_pressure_cycle','pre_flow', 
                                'true_time_pre_flow', 'post_flow','true_time_post_flow']
    full_set_double_2 = full_set.copy()
    full_set_double_2.drop(['clinical_visits_body_mass_index_value','clinical_visits_pre_24h_dbp_mean_value',
                            'clinical_visits_pre_24h_sbp_mean_value','MeanPAIPerDay_PreMid','pre_finger_pressure_cycle',
                            'pre_flow','true_time_pre_flow'], axis=1, inplace=True)
    full_set_double_2.columns = ['roottable_case_id_text', 'roottable_age_value', 'roottable_sex_item','clinical_visits_cpet_vo2max_value', 
                                'roottable_pre_smoker_item','roottable_pre_snuff_item', 'roottable_pre_wine_value',
                                'roottable_pre_beer_value', 'clinical_visits_cpre_hrmax_value','MeanPAIPerDay', 
                                'clinical_visits_post_24h_dbp_mean_value','clinical_visits_post_24h_sbp_mean_value',
                                'clinical_visits_pre_24h_dbp_mean_value','clinical_visits_pre_24h_sbp_mean_value',
                                'pre_finger_pressure_cycle','post_finger_pressure_cycle', 'pre_flow', 'true_time_pre_flow', 
                                'post_flow','true_time_post_flow','clinical_visits_body_mass_index_value']
    full_set_double_1 = full_set_double_1[['roottable_case_id_text', 'roottable_age_value', 'roottable_sex_item','clinical_visits_body_mass_index_value','clinical_visits_cpet_vo2max_value', 'roottable_pre_smoker_item','roottable_pre_snuff_item', 'roottable_pre_wine_value','roottable_pre_beer_value', 'clinical_visits_cpre_hrmax_value','MeanPAIPerDay', 'clinical_visits_pre_24h_dbp_mean_value','clinical_visits_pre_24h_sbp_mean_value','pre_finger_pressure_cycle','pre_flow', 'true_time_pre_flow','clinical_visits_post_24h_dbp_mean_value','clinical_visits_post_24h_sbp_mean_value','post_finger_pressure_cycle','post_flow','true_time_post_flow']]
    full_set_double_2 = full_set_double_2[['roottable_case_id_text', 'roottable_age_value', 'roottable_sex_item','clinical_visits_body_mass_index_value','clinical_visits_cpet_vo2max_value', 'roottable_pre_smoker_item','roottable_pre_snuff_item', 'roottable_pre_wine_value','roottable_pre_beer_value', 'clinical_visits_cpre_hrmax_value','MeanPAIPerDay', 'clinical_visits_pre_24h_dbp_mean_value','clinical_visits_pre_24h_sbp_mean_value','pre_finger_pressure_cycle','pre_flow', 'true_time_pre_flow','clinical_visits_post_24h_dbp_mean_value','clinical_visits_post_24h_sbp_mean_value','post_finger_pressure_cycle','post_flow','true_time_post_flow']]

    full_set_double = pd.concat([full_set_double_1,full_set_double_2])
    full_set_double = full_set_double.sort_values(by=['roottable_case_id_text'])

    full_set_double.pre_finger_pressure_cycle = full_set_double.pre_finger_pressure_cycle.tolist()
    full_set_double.post_finger_pressure_cycle = full_set_double.post_finger_pressure_cycle.tolist()
    full_set_double.reset_index(drop=True, inplace=True)

    return full_set_double


data_set = merge_all()
data_set = add_post_mid_bp(data_set)
data_set = add_cycles(data_set)
data_set = addFlow(data_set)
data_set = correct_data_types(data_set)
#to_CSV(data_set)
data_set = split_week_6(data_set)
#to_CSV(data_set)


