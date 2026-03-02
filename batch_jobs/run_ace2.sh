#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=50gb
#SBATCH --partition=gputest
#SBATCH --gres=gpu:1
#SBATCH --job-name=ace2-cpl
#SBATCH --output=logs/ace2-coupled-%A.txt

source ~/.bashrc
conda activate ace

cd /home/a/antonio/repos/ace

START_DATETIME="19510101"

# Note that start datetime here means first init datetime, not first target datetime like the Gencast code.
NUMYEARS=70
NUMSTEPS=$((NUMYEARS*366*4))

# --output-vars PRATEsfc LHTFLsfc \

# python -m run_ace --inference-config /home/a/antonio/repos/ace/configs/coupled_inference_config.yaml \
# --model-name ace2-forced \
# --model-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data \
# --output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions/ \
# --experiment-name ace2_forced_control_${NUMYEARS}years \
# --output-vars TMP2m \
# --initial-condition-path /network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/initial_conditions/ic_19510101.nc \
# --forcing-data-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/forcing_data/control_1951-2051 \
# --logging-dir /home/a/antonio/repos/ace/logs \
# --start-datetime "${START_DATETIME}-0" \
# --num-steps-per-initialisation ${NUMSTEPS};


python -m run_ace --inference-config /home/a/antonio/repos/ace/configs/coupled_inference_config.yaml \
--model-name ace2-forced \
--model-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data \
--output-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/predictions/ \
--experiment-name ace2_forced_hist_${NUMYEARS}years \
--output-vars TMP2m \
--initial-condition-path /network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/initial_conditions/ic_19510101.nc \
--forcing-data-dir /network/group/aopp/predict/HMC005_ANTONIO_EERIE/ace2_data/forcing_data/historical_1951-2021 \
--logging-dir /home/a/antonio/repos/ace/logs \
--start-datetime "${START_DATETIME}-0" \
--num-steps-per-initialisation ${NUMSTEPS};
