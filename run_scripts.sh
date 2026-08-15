#!/bin/bash

# configure color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# specify the scripts to run (example)
scripts=(
  "./scripts/vary_batch_size/replay_mp_default_train_bs_16_eval_bs_16.sh"
)

# print the list of scripts to run
echo -e "${YELLOW}Scripts to run:${NC}"
for script in "${scripts[@]}"; do
    echo "$script"
done
echo "======================================================================"

# configure the output directory
output_dir="./output/test_results/"

# create the output directory
mkdir -p "$output_dir"

# run each script
for script in "${scripts[@]}"; do
    # extract the script name (without the extension)
    script_name=$(basename "$script" .sh)

    # create a unique output directory for this script
    script_output_dir="${output_dir}/${script_name}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$script_output_dir"

    # log file path
    log_file="${script_output_dir}/${script_name}.log"

    echo -e "${GREEN}Running script: $script${NC}"

    # grant execution permission to the script
    chmod +x "$script"

    # run the script and save the output to the log file
    bash "$script" >> "$log_file" 2>&1
    
    # check the execution result
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Script execution completed: $script${NC}"
    else
        echo -e "${RED}Script execution failed: $script${NC}"
    fi

    # move the generated .pth files
    mv ./*.pth "$script_output_dir/" 2>/dev/null
    echo "Moved generated .pth files to $script_output_dir"
    
    echo "---------------------------------------"
done

echo -e "${GREEN}All scripts have been executed. Logs are saved in the $output_dir directory.${NC}"
