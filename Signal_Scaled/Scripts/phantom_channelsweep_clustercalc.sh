#!/bin/bash

SIGNAL_NAMES=( "clustercalc_signal" )
#SIGNAL_NAMES=( "clustercalc_signal" )
NUM_CHANNELS_LIST=( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 )
#NUM_CHANNELS_LIST=( 1 3 5 7 9 11 13 15 17 19 )
#NUM_CHANNELS_LIST=( 2 4 6 8 10 12 14 16 18 20 )
#NUM_CHANNELS_LIST=( 1 )
IMAGE_SIZE=65
SIGNAL_PATH="../Data/Phantom/clustercalc/"
DATA_PATH="../Data/Phantom/clustercalc/"
SAVE_PATH="../Models/Phantom/"
STOP_METHOD=1
if [ $STOP_METHOD -eq 1 ]; then
	STOP_STRING='Val'
else
	STOP_STRING='Train'
fi

#declare -a arr=("element1", "element2", "element3")
#for i in "${arr}"


#TRAIN_IMAGES_LIST=( 250 500 1000 2000 5000 10000 15000 20000 25000 30000 )
TRAIN_IMAGES_LIST=( 250 500 1000 2000 5000 )

#echo $TRAIN_IMAGES_LIST
for j in "${SIGNAL_NAMES[@]}"
do
	SIGNAL_NAME="$j"
	for i in "${TRAIN_IMAGES_LIST[@]}"
	do
		for k in "${NUM_CHANNELS_LIST[@]}"
		do
			NUM_CHANNELS="$k"
			TRAIN_IMAGES="$i"
			BASE_NAME="TMI_${NUM_CHANNELS}LS_T_${TRAIN_IMAGES}_${STOP_STRING}Stop"
			DIR_LOCATION="${SAVE_PATH}${SIGNAL_NAME}/${BASE_NAME}"
                        if [ ! -d ${SAVE_PATH}${SIGNAL_NAME} ]; then
                                mkdir ${SAVE_PATH}${SIGNAL_NAME} || exit 1
                        fi
			if [ ! -d $DIR_LOCATION ]; then
				#echo "Making directory"
				mkdir $DIR_LOCATION || exit 1
			fi
			echo $DIR_LOCATION
			python3 tmi_aetsi_script_phantom.py --train_size $TRAIN_IMAGES --image_size $IMAGE_SIZE --signal_path $SIGNAL_PATH --signal_name $SIGNAL_NAME --data_path $DATA_PATH --num_channels $NUM_CHANNELS --validation $STOP_METHOD --save_path ${SAVE_PATH}${SIGNAL_NAME} > "${DIR_LOCATION}/stdout.txt" 2>"${DIR_LOCATION}/stderr.txt" || exit 1
		done
	done
done
#TRAIN_IMAGES=250
#BASE_NAME="TMI_4LS_T_${TRAIN_IMAGES}_${STOP_STRING}Stop/"
#DIR_LOCATION="${SAVE_PATH}${BASE_NAME}"

#echo $STOP_STRING
#echo $DIR_LOCATION

#mkdir $DIR_LOCATION
#mkdir ../Models/Lumpy/Circular/TMI_4LS_T_250_TrainStop/ || exit 1
#python3 tmi_aetsi_script.py --train_size 250 --image_size $IMAGE_SIZE --signal_path $SIGNAL_PATH --signal_name $SIGNAL_NAME$ --data_path $DATA_PATH --save_path $SAVE_PATH > ../Models/Lumpy/Circular/TMI_4LS_T_250_TrainStop/stdout.txt 2>../Models/Lumpy/Circular/TMI_4LS_T_250_TrainStop/stderr.txt || exit 1

exit 0
