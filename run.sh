#!/bin/bash

TARGET="bin/benchmark"

COUNT=5

POINTS_FULL=(576 1200 1500 2500 3200 3600 5000 6000 7600)
POINTS_SHORT=(1200)

for SIZE in "${POINTS_SHORT[@]}"
do
	for (( j=1; j<=${COUNT}; j++ )) ;
	do
		echo -e "\n\n ================= Iteration = $j ================="
		${TARGET} ${SIZE}

		if [[ $? != 0 ]]; then exit 1; fi 	# Выход если итерация завершилась с ошибкой

		sleep 1s 	# Задержка для "охлаждения"
	done
	# sleep 2m
done
