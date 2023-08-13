const { Battle } = require("@pkmn/client");
const { Player } = require("./dist/src/player");

const { Generations, ID } = require("@pkmn/data");
const { ModdedDex } = require("@pkmn/dex");
const { Teams } = require("@pkmn/sim");
const { TeamGenerators } = require("@pkmn/randoms");

Teams.setGeneratorFactory(TeamGenerators);

const formatid = "gen9randombattle";
const modid = formatid.slice(0, 4);
const gens = new Generations(new ModdedDex(modid));

const player = new Player(null, 0, 0, new Battle(gens));
const stateSize = Math.floor(player.getState().length / 2);
console.log(`State Bits: `, stateSize);
