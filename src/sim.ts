import { Teams } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";
import { ReadStream } from "@pkmn/streams";

import { Generations, ID } from "@pkmn/data";
import { ModdedDex } from "@pkmn/dex";

import { Buffer } from "node:buffer";
import { GameStore } from "./types";
import { Game } from "./game";

Teams.setGeneratorFactory(TeamGenerators);

const formatid: string = "gen9randombattle";
const modid = formatid.slice(0, 4);
const gens = new Generations(new ModdedDex(modid as ID));

export class StdinReadBytes extends ReadStream {
    constructor(options = {}) {
        super(options);
    }
    async nextMessage(byteCount = null) {
        const value = await this.readBuffer(byteCount);
        return { value, done: value === null };
    }
}

export function byteToCommand(actionIndex: number): string {
    let cmd: string;
    if (actionIndex === 255) {
        cmd = "default";
    } else if (actionIndex >= 0 || actionIndex < 4) {
        cmd = `move ${cmd + 1}`;
    } else if (actionIndex >= 4 || actionIndex < 10) {
        cmd = `switch ${cmd + 1}`;
    }
    return cmd;
}

export const n = 1;
export const messageSize = 4;

export function start() {
    var stdin = new StdinReadBytes(process.stdin);

    let games: GameStore = {};

    for (let i = 0; i < n; i++) {
        const game = new Game(i, formatid, gens);
        games[i] = game;
        game.run();
    }

    (async () => {
        let done: any,
            outStream: { write: (arg0: any) => any },
            gameByte: number,
            playerByte: number,
            actionByte: number,
            playerId: string,
            cmd: string,
            value: Buffer;

        while (
            (({ value, done } = await stdin.nextMessage(messageSize)), !done)
        ) {
            [gameByte, playerByte, actionByte] = value;

            switch (playerByte) {
                case 0:
                    playerId = "p1";
                    break;
                case 1:
                    playerId = "p2";
                    break;
            }

            outStream = games[gameByte][playerId].stream;

            cmd = byteToCommand(actionByte);
            await outStream.write(cmd);
        }
    })();
}
