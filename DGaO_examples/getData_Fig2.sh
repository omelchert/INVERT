

function singleRun {
  ZD=$1
  python main_inv_EX2.py $ZD > ./Fig2/inv_a00.1_zD${ZD}.dat
}



singleRun -0.01
singleRun -0.02
singleRun -0.05
singleRun -0.1
singleRun -0.5
