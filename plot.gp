set encoding utf8

set terminal pngcairo size 1000,800 font "Times, 16" enhanced

if (!exists('inputfile')) inputfile = './cpu.result'
if (!exists('outputfile')) outputfile = 'result'
if (!exists('ycol')) ycol = 11;

# set title 'Виброперемещение основания подшипника'

set xlabel "Размерность матрицы, NxN"
set ylabel "Время решения, с"

set grid

# set logscale y
set xrange [0:8000]
set yrange [1e-1:*]

x0 = y0 = NaN

set output outputfile.".png"

# plot inputfile using (stringcolumn(3) eq "-O3" ? (y0=$4,x0=$5) : x0):(y0) with line title "CPU -O3"
plot '< sort -nk1 cpu.result' using 1:2 with line title "Intel Core i5-8250U, -O3", \
	'< sort -nk1 cpu.result' using 1:2:($2-100*$3):($2+100*$3) with yerrorbars title "Intel Core i5-8250U err, -O3", \
 '< sort -nk1 intel.result' using 1:2 with line title "Intel UHD 620", \
 '< sort -nk1 amd.result' using 1:2 with line title "AMD"

set logscale y
set xrange [0:2000]
set yrange [1e-6:*]

set output outputfile."_small".".png"

# plot inputfile using (stringcolumn(3) eq "-O3" ? (y0=$4,x0=$5) : x0):(y0) with line title "CPU -O3"
plot '< sort -nk1 cpu.result' using 1:2 with line title "Intel Core i5-8250U, -O3", \
 '< sort -nk1 intel.result' using 1:2 with line title "Intel UHD 620", \
 '< sort -nk1 amd.result' using 1:2 with line title "AMD"