import numpy as np
import os

# Vacancy heating ramp data for synthetic quenched/aged states (used by s07 & s08)
os.system("mkdir -p ./data/vacancy_jump_heating/")
os.system("cp -r /home/yifanc/MC_simulations/01_MC_20220328/38_MC_MTP_20230208/vacancy_jump_heating/post_processing/figures/details/WC_evolution_ramp_summary.json ./data/vacancy_jump_heating/")
os.system("cp -r /home/yifanc/MC_simulations/01_MC_20220328/38_MC_MTP_20230208/vacancy_jump_heating/post_processing/figures/details/WC_evolution_single_ramp_summary.json ./data/vacancy_jump_heating/")
os.system("cp -r /home/yifanc/MC_simulations/01_MC_20220328/38_MC_MTP_20230208/vacancy_jump_heating/post_processing/figures/details/s02_lat_vs_T_summary.json ./data/vacancy_jump_heating/")
os.system("cp -r /home/yifanc/MC_simulations/01_MC_20220328/38_MC_MTP_20230208/vacancy_jump_heating/post_processing/figures/single_ramp/details/vacancy_jumps_single_ramp_summary.json ./data/vacancy_jump_heating/")
os.system("cp -r /home/yifanc/MC_simulations/01_MC_20220328/38_MC_MTP_20230208/vacancy_jump_heating/post_processing/figures/details/WC_evolution_ramp_continue_1650_summary.json ./data/vacancy_jump_heating/")
os.system("cp -r /home/yifanc/MC_simulations/01_MC_20220328/38_MC_MTP_20230208/vacancy_jump_heating/post_processing/figures/details/WC_evolution_single_ramp_continue_1650_summary.json ./data/vacancy_jump_heating/")
os.system("cp -r /home/yifanc/MC_simulations/01_MC_20220328/38_MC_MTP_20230208/vacancy_jump_heating/post_processing/figures/single_ramp/details/vacancy_jumps_single_ramp_continue_1650_summary.json ./data/vacancy_jump_heating/")
