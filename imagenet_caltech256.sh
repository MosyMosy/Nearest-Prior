#!/bin/bash
#SBATCH --mail-user=Moslem.Yazdanpanah@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=imagenet_caltech256
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=1-00:00
#SBATCH --account=rrg-ebrahimi

nvidia-smi

source ~/ENV/bin/activate

echo "------------------------------------< Data preparation>----------------------------------"
echo "Copying the source code"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/Nearest-Prior .

echo "Copying the datasets"
date +"%T"
cp ~/scratch/dataset/imagenet_object_localization_patched2019.tar.gz .
cp ~/scratch/dataset/ILSVRC_val.zip .
cp ~/scratch/DA_Dataset/caltech256.zip .

echo "creating data directories"
date +"%T"
cd Nearest-Prior
cd dataset
tar -xzf $SLURM_TMPDIR/imagenet_object_localization_patched2019.tar.gz
unzip -q $SLURM_TMPDIR/ILSVRC_val.zip
unzip -q $SLURM_TMPDIR/caltech256.zip



echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cd Nearest-Prior

title=ImageNet_caltech256_with_new_regularization

python train.py --title $title --epochs 20 --batch_size 128 --learning_rate 0.01 --reg_weight 10 --sigma 1 --print_freq 100

echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
echo "--------------------------------------<backup the result>-----------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/Nearest-Prior/logs/$title ~/scratch/Nearest-Prior/logs/
cp -r $SLURM_TMPDIR/Nearest-Prior/wandb/ ~/scratch/Nearest-Prior/