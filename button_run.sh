#!/bin/bash
# echo alias doic_button="bash -i /home/yzq/hzy/DOIC/button_run.sh" >> ~/.bashrc && source ~/.bashrc
# sudo chmod +x /home/yzq/hzy/DOIC/button_run.sh

cd /home/yzq/hzy/DOIC/Main

echo -e "\033[36m"
echo -e "Activating environment ..."
conda activate doic
echo "Finish activation"

echo ""

powerbtn_func="/home/yzq/hzy/DOIC/handle-powerbtn.sh"
save_dir="/home/yzq/hzy/DOIC/Results"
if_value_savedir=false
echo "Running with args:"
for arg in "$@"
do
    if [[ $arg =~ ^-- ]]
    then
        echo -e "> $arg"
        if [ "$arg" == "--save_dir" ]
        then
            if_value_savedir=true
        fi

    elif [[ $arg =~ ^- ]]
    then
        echo -n -e "> $arg"
        if [ "$arg" == "-s" ]
        then
            if_value_savedir=true
        fi

    else
        echo -e " $arg"
        if [ $if_value_savedir = true ]
        then
            save_dir=$(readlink -f "$arg")
        fi
    fi
done
sed -i "s|SAVE_PATH=.*|SAVE_PATH=$save_dir|" $powerbtn_func

echo ""
echo "Loading Script: /home/yzq/hzy/DOIC/Main/main_button.py"
echo -e "\033[0m"

python main_button.py "$@" 