function singleRun {
  R=$1
  NA=$2
  FN=$3
  python main_inv_kernelReconstruction_EX2.py ./FIG2a/INPUT/${FN} $R $NA > ./FIG2a/kRec_R${R}_Na${NA}_${FN}
}

N=41 
R=0.10

singleRun $R $N topHat_zD-0.5_a0.10_a00.10.dat
