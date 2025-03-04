cd ../work/save-vjk-nz;

for d in cco diamond hg1212 nio; do
    grep -r --no-filename --exclude=*py -A 1 "###         c0" ./$d > ../../plot/$d.out
    if [ $? -eq 0 ]; then
        echo "  Created ../../plot/$d.out successfully";
    else
        echo "  Error processing $d";
    fi
done

cd -;

for d in cco diamond hg1212 nio; do
    echo "Plotting $d..."
    python plot.py $d;
done
