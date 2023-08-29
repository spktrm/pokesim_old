import * as fs from "fs";
import * as path from "path";

const formatid = "gen9randombattle";
const data = fs.readFileSync("./src/data.json");
const { sideConditions, weathers, pseudoWeather, terrain, volatileStatus } =
    JSON.parse(data.toString());

function listToEnumObj(arr: Array<any>): Object {
    return arr.reduce(
        (acc, item) => {
            if (item) {
                acc[item.trim()] = Object.keys(acc).length - 1;
            }
            return acc;
        },
        { "": -1 },
    );
}

const pokemonMapping = listToEnumObj(
    fs
        .readFileSync(path.resolve(__dirname, `${formatid}/moves.csv`))
        .toString()
        .split("\n")
        .map((x) => x.split(",")[0])
        .slice(1)
        .sort(),
);

function readCsvHeaders(fpath: string): Array<string> {
    return fs
        .readFileSync(path.resolve(__dirname, fpath))
        .toString()
        .split("\n")[0]
        .split(",");
}

const itemMapping = listToEnumObj(
    readCsvHeaders(`${formatid}/items.csv`).slice(1),
);

const moveMapping = listToEnumObj(
    readCsvHeaders(`${formatid}/moves.csv`).slice(1),
);

const abilityMapping = listToEnumObj(
    readCsvHeaders(`${formatid}/abilities.csv`).slice(1),
);

const sideConditionsMapping = sideConditions;

const terrainMapping = terrain;

const weatherMapping = weathers;

const volatileStatusMapping = volatileStatus;

const pseudoWeatherMapping = pseudoWeather;

const statusMapping = {
    slp: 0,
    psn: 1,
    brn: 2,
    frz: 3,
    par: 4,
    tox: 5,
};

const boostsMapping = {
    atk: 0,
    def: 1,
    spa: 2,
    spd: 3,
    spe: 4,
    accuracy: 5,
    evasion: 6,
};

export {
    pokemonMapping,
    abilityMapping,
    moveMapping,
    itemMapping,
    sideConditionsMapping,
    terrainMapping,
    weatherMapping,
    volatileStatusMapping,
    pseudoWeatherMapping,
    statusMapping,
    boostsMapping,
};
