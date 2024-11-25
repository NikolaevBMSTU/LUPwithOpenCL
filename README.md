# LUPwithOpenCL


Works in Windows and Linux with C++17.

Главный файл main.cpp содержит список макросов для включения/выключения отдельных бенчмарков.

Сборка и пробный запуск осуществляется с помощью скрипта make.sh.

./make.sh

Можно указать размер матрицы для пробного запуска, например 7600х7600

./make.sh 7600

Последующие запуски осуществляются (по умолчанию) командой

bin/benchmark

Для тестирования разных вариантов LAPACK нужно собирать 

LAPACK_VENDOR=LAPACK ./make.sh

Для LAPACK в реализации OpenBLAS

LAPACK_VENDOR=OpenBLAS ./make.sh

Также возможно комбинирование LAPACK + BLAS от OpenBLAS

LAPACK_VENDOR=COMBINED ./make.sh

В папке external располагаются внешние библиотки

