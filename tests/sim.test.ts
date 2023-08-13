import { test, describe } from "@jest/globals";

import { Game } from "../src/game";
import { Generations, ID } from "@pkmn/data";
import { ModdedDex } from "@pkmn/dex";
import { Teams } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";

Teams.setGeneratorFactory(TeamGenerators);

const formatid: string = "gen9randombattle";
const modid = formatid.slice(0, 4);
const gens = new Generations(new ModdedDex(modid as ID));

describe("test-sim", () => {
    test("test-run", async () => {
        const game = new Game(0, formatid, gens, true);
        for (let i = 0; i < 100; i++) {
            await game.play();
        }
    }, 10);
});
