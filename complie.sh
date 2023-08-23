python scripts/download_tokens.py
python scripts/download_data.py
npx prettier ./src -w 
npx prettier ./tests -w 
tsc
cp src/*.csv dist/src
pkg sim.js -o sim.sim -t node18