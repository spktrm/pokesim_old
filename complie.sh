python scripts/download_tokens.py
python scripts/download_data.py
npx prettier ./src -w 
npx prettier ./tests -w 
tsc
for gen in 3 4 5 9
do
    cp -r src/gen${gen}randombattle dist/src/gen${gen}randombattle
done
pkg sim.js -o sim.sim -t node18