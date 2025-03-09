cd ../work/;

method=$(ls -1)
for m in $method; do
    echo "Processing $m..."; cd $m;
    # only keep the even number lines of the output, skip the first row
    grep -r --no-filename --exclude=*py "###" | sed -n 'n;p' | sed 's/^###\s*//' > ../../plot/$m.out;
    cd ..;
done

cd /home/junjiey/work/benchmark-fftisdf/plot;