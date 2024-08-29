#!/bin/bash
# echo alias doic="bash -i /home/yzq/hzy/DOIC/fast_run.sh" >> ~/.bashrc && source ~/.bashrc
# sudo chmod +x /home/yzq/hzy/DOIC/fast_run.sh

cd /home/yzq/hzy/DOIC/Main

echo -e "\033[36m"
echo -e "Activating environment ..."
conda activate doic
echo "Finish activation"

echo ""

echo "Running with args:"
for arg in "$@"
do
    if [[ $arg == -* ]]
    then
        echo -n -e "\n> $arg"
    else
        echo -e " $arg\n"
    fi
done

echo ""
echo "Loading Script in /home/yzq/hzy/DOIC/Main/main.py"
echo -e "\033[0m"

python main.py "$@"