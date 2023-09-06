import { stdin, stdout } from "@pkmn/streams";
import { Player } from "./player";
import { Teams } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";
import { ID, ModdedDex } from "@pkmn/dex";
import { Generations } from "@pkmn/data";
import * as readline from "readline";

Teams.setGeneratorFactory(TeamGenerators);

const formatid: string = "gen9randombattle";
const modid = formatid.slice(0, 4);
const gens = new Generations(new ModdedDex(modid as ID));

const player = new Player(null, 0, null, gens, false);
const outStream = stdout();

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: true,
});

rl.on("line", async (chunk) => {
    // console.log(chunk + "\n\n");
    await player.receiveChunk(outStream, chunk);
});
