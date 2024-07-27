
for MYEPS in 0 0.001 0.003 0.01 0.03 0.1
    for MYALGO in lp.py mip.py
        python $MYALGO ./data1.pth --perturbation $MYEPS  > ./output/$MYALGO.$MYEPS.txt 2>&1 &
    end
end
