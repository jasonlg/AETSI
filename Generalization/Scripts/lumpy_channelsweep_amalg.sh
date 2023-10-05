#!/bin/bash

#SIGNAL_NAMES=( "circular_signal_s3" "circular_signal_s5" "circular_signal_s7" )
SIGNAL_NAMES=( "elliptical_signal_sy1_5" )
#SIGNAL_NAMES=( "circular_signal_s3" )
#NUM_CHANNELS_LIST=( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 )
NUM_CHANNELS_LIST=( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 )
#NUM_CHANNELS_LIST=( 15 16 )
IMAGE_SIZE=64
SIGNAL_PATH="../Data/amalg/"
DATA_PATH="../Data/amalg/"
SAVE_PATH="../Models/amalg/"
STOP_METHOD=1
if [ $STOP_METHOD -eq 1 ]; then
	STOP_STRING='Val'
else
	STOP_STRING='Train'
fi

#declare -a arr=("element1", "element2", "element3")
#for i in "${arr}"


#TRAIN_IMAGES_LIST=( 250 500 1000 2000 5000 10000 15000 20000 25000 30000)
#TRAIN_IMAGES_LIST=( 5000 30000 )
TRAIN_IMAGES_LIST=( 5000 10000 30000 )

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
			python3 tmi_aetsi_script_lumpy.py --train_size $TRAIN_IMAGES --image_size $IMAGE_SIZE --signal_path $SIGNAL_PATH --signal_name $SIGNAL_NAME --data_path $DATA_PATH --num_channels $NUM_CHANNELS --validation $STOP_METHOD --save_path ${SAVE_PATH}${SIGNAL_NAME} > "${DIR_LOCATION}/stdout.txt" 2>"${DIR_LOCATION}/stderr.txt" || exit 1
		done
	done
done

exit 0
