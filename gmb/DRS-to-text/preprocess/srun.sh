
gpu=0
part=M_AND_I_GPU
execution="python oracle_unordered.py ../trn.json 456789123 > trn.oracle_unordered_456789123.json"
logfile="nohup.out"

nohup srun --gres gpu:${gpu} --mem 20G -p ${part} ${execution} 2>${logfile} &
