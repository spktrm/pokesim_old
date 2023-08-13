import { AnyObject, BattleStreams, Teams } from "@pkmn/sim";
import { ModdedDex } from "@pkmn/dex";
import { TeamGenerators } from "@pkmn/randoms";

import { Battle } from "@pkmn/client";
import { Generations, ID } from "@pkmn/data";
import { Uint16State } from "../src/state";
import { expect, test, describe } from "@jest/globals";
import { ObjectReadWriteStream } from "@pkmn/streams";
import { isActionRequired } from "../src/player";

function range(start: number, end?: number, step = 1) {
    if (end === undefined) {
        end = start;
        start = 0;
    }
    const result = [];
    for (; start <= end; start += step) {
        result.push(start);
    }
    return result;
}

function getLegalMask(request: AnyObject, done: boolean) {
    const mask = Array(10);
    if (request === undefined || done) {
        mask.fill(1);
        mask[4] = 0;
        return mask;
    } else {
        mask.fill(0);
    }

    if (request.wait) {
    } else if (request.forceSwitch) {
        const pokemon = request.side.pokemon;
        const switches = range(1, 6).filter(
            (j) =>
                pokemon[j - 1] &&
                // not active
                j > request.forceSwitch.length &&
                // not fainted or fainted and using Revival Blessing
                !!(
                    +!!pokemon[0].reviving ^
                    +!pokemon[j - 1].condition.endsWith(` fnt`)
                )
        );
        for (let i = 0; i < switches.length; i++) {
            const slot = switches[i];
            mask[slot + 3] = 1;
        }
    } else if (request.active) {
        const pokemon = request.side.pokemon;
        const active = request.active[0];
        const possibleMoves = active.moves ?? [];
        const moves = range(1, possibleMoves.length)
            .filter(
                (j) =>
                    // not disabled
                    !possibleMoves[j - 1].disabled
                // NOTE: we don't actually check for whether we have PP or not because the
                // simulator will mark the move as disabled if there is zero PP and there are
                // situations where we actually need to use a move with 0 PP (Gen 1 Wrap).
            )
            .map((j) => ({
                slot: j,
                move: possibleMoves[j - 1].move,
                target: possibleMoves[j - 1].target,
                zMove: false,
            }));

        const canSwitch = range(1, 6).filter(
            (j) =>
                pokemon[j - 1] &&
                // not active
                !pokemon[j - 1].active &&
                // not fainted
                !pokemon[j - 1].condition.endsWith(` fnt`)
        );
        const switches = active.trapped || active.maybeTrapped ? [] : canSwitch;

        for (let i = 0; i < moves.length; i++) {
            const slot = moves[i].slot;
            mask[slot - 1] = 1;
        }

        for (let i = 0; i < switches.length; i++) {
            const slot = switches[i];
            mask[slot + 3] = 1;
        }
    }

    return mask;
}

function numberTo16BitArray(number: number): Array<number> {
    const binaryString = number.toString(2).padStart(16, "0");
    return binaryString.split("").map((bit) => parseInt(bit, 10));
}

async function AssertMask(
    stream: ObjectReadWriteStream<string>,
    room: Battle
): Promise<void> {
    let testMask: number[], corrMask: number[], testP1Side: number[];
    for await (const chunk of stream) {
        for (const line of chunk.split("\n")) {
            room.add(line);
        }
        room.update();
        if (isActionRequired(chunk, room.request)) {
            testMask = Uint16State.getLegalMask(room.request, false);
            corrMask = getLegalMask(room.request, false);

            expect(
                testMask
                // numberTo16BitArray(testMask).slice(-corrMask.length)
            ).toEqual(corrMask);

            stream.write("default");
        }
    }
    testMask = Uint16State.getLegalMask(room.request, true);
    corrMask = getLegalMask(room.request, true);
    expect(
        testMask
        // numberTo16BitArray(testMask).slice(-corrMask.length)
    ).toEqual(corrMask);
}

describe("test-state", () => {
    test("test-mask", async () => {
        Teams.setGeneratorFactory(TeamGenerators);

        const streams = BattleStreams.getPlayerStreams(
            new BattleStreams.BattleStream()
        );
        const formatid = "gen7randombattle";
        const spec = { formatid: formatid, seed: [69, 69, 69, 69] };

        const p1spec = {
            name: "Bot 1",
            team: Teams.pack(Teams.generate(formatid)),
        };
        const p2spec = {
            name: "Bot 2",
            team: Teams.pack(Teams.generate(formatid)),
        };

        const modid = formatid.slice(0, 4) as ID;
        const gens = new Generations(new ModdedDex(modid));
        const p1room = new Battle(gens);
        const p2room = new Battle(gens);

        AssertMask(streams.p1, p1room);
        AssertMask(streams.p2, p2room);

        void streams.omniscient.write(`>start ${JSON.stringify(spec)}
>player p1 ${JSON.stringify(p1spec)}
>player p2 ${JSON.stringify(p2spec)}`);
    });
    test("test-2", async () => {
        const streams = BattleStreams.getPlayerStreams(
            new BattleStreams.BattleStream()
        );
        const formatid = "gen9randombattle";
        const spec = { formatid: formatid, seed: [420, 69, 420, 69] };

        const p1spec = {
            name: "Bot 1",
            team: Teams.pack(Teams.generate(formatid)),
        };
        const p2spec = {
            name: "Bot 2",
            team: Teams.pack(Teams.generate(formatid)),
        };

        const modid = formatid.slice(0, 4) as ID;
        const gens = new Generations(new ModdedDex(modid));
        const p1room = new Battle(gens);
        const p2room = new Battle(gens);

        AssertMask(streams.p1, p1room);
        AssertMask(streams.p2, p2room);

        void streams.omniscient.write(`>start ${JSON.stringify(spec)}
>player p1 ${JSON.stringify(p1spec)}
>player p2 ${JSON.stringify(p2spec)}`);
    });
});
