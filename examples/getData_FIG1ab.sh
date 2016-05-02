function singleRun {
  A0=$1
  ZD=$2
  RT=$3
  NA=$4
  python main_inv_kernelReconstruction_EX1.py $A0 $ZD $RT $NA  > ./FIG1ab/kRec_a0${A0}_zD${ZD}_rTs${RT}_Na${NA}.dat 
}


# approximation in far field 
for Na in 5 11 15 21 25 31 35 41 45 51 55 61 65 71 75 81 85 91 95 101;
do
   singleRun 0.1 -0.5 0.10 $Na
done

# approximation in near field 
for Na in 3 5 7 9 11 13 15 17 21 23 25 27 29 31 33 35 37 41 45 51 55 61 65 71 75 81 85 91 95 101;
do
   singleRun 0.1 -0.02 0.20 $Na
done

