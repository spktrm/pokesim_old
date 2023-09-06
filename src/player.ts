import { ObjectReadWriteStream, WriteStream } from "@pkmn/streams";
import { Battle } from "@pkmn/client";
import { AnyObject } from "@pkmn/sim";
import { Generations } from "@pkmn/data";
import { Game } from "./game";
import { Uint16State } from "./state";

import * as fs from "fs";
import { v4 } from "uuid";
import { getRandomAction } from "./sim";

export function isActionRequired(chunk: string, request: AnyObject): boolean {
    if (request === null) {
        return false;
    }
    if (request.requestType === "wait") {
        return false;
    }
    if (chunk.includes("|turn")) {
        return true;
    }
    if (!chunk.includes("|request")) {
        return !!request.forceSwitch;
    }
    return false;
}

const workerIndex = parseInt(process.argv[2]) || 0;

function randomAction(arr: Array<number>) {
    const nonZeroIndices = arr.reduce((indices, value, index) => {
        if (value !== 0) {
            indices.push(index);
        }
        return indices;
    }, []);
    const randomIndex = Math.floor(Math.random() * nonZeroIndices.length);
    const index = nonZeroIndices[randomIndex];

    if (index >= 0 && index < 4) {
        return `move ${index + 1}`;
    } else {
        return `switch ${index - 4 + 1}`;
    }
}

function isAction(line: string): boolean {
    const splitString = line.split("|");
    const actionType = splitString[1];
    switch (actionType) {
        case "move":
            return true;
        case "switch":
            return true;
        default:
            return false;
    }
}

export class Player {
    game: Game;
    playerIndex: number;
    done: boolean;
    stream: ObjectReadWriteStream<string>;
    log: any[];
    turns: Object;
    debug: boolean;
    room: Battle;
    gens: Generations;
    prevRequest: AnyObject;
    constructor(
        playerStream: ObjectReadWriteStream<string>,
        playerIndex: number,
        game: Game,
        gens: Generations,
        debug: boolean = false
    ) {
        this.game = game;
        this.playerIndex = playerIndex;
        this.done = false;
        this.stream = playerStream;
        this.log = [];
        this.turns = { 0: [] };
        this.debug = debug;
        this.room = new Battle(gens);
        this.prevRequest = null;
    }

    getStream() {
        return (this.game ?? {})?.stream;
    }
    getWinner() {
        let winner: number;
        const stream = this.getStream();
        if (stream === undefined) {
            return -1;
        }
        switch (stream.battle.winner) {
            case undefined:
                winner = -1;
                break;
            case this.room.p1.name:
                winner = 0;
                break;
            case this.room.p2.name:
                winner = 1;
                break;
        }
        return winner;
    }
    getState(): Buffer {
        const turn = Math.max(0, this.room.turn - 1);
        const baseInfo: number[] = [
            workerIndex,
            (this.game ?? {})?.gameIndex ?? 0,
            this.playerIndex,
            this.done ? 1 : 0,
            this.getWinner(),
            turn,
        ];
        const actionLines = this.turns[turn] ?? [];
        const state = new Uint16State(this.playerIndex, this.room);
        const legalMask = state.getLegalMask(this.done);

        let data: any;
        data = [
            ...baseInfo,
            ...state.getRequest(),
            ...state.getMyTeam(),
            ...state.getOppTeam(),
            ...state.getMySideConditions(),
            ...state.getOppSideConditions(),
            ...state.getMyVolatileStatus(),
            ...state.getOppVolatileStatus(),
            ...state.getMyBoosts(),
            ...state.getOppBoosts(),
            ...state.getField(),
            ...state.actionToVector(actionLines[0]),
            ...state.actionToVector(actionLines[1]),
            ...state.actionToVector(actionLines[2]),
            ...state.actionToVector(actionLines[3]),
            ...legalMask,
            2570, // \n\n in ascii hex
        ];
        const arr = new Int16Array(data);
        const buf = Buffer.from(arr.buffer);
        return buf;
    }
    receive(chunk: string): boolean {
        if (chunk.startsWith("|error")) {
            throw Error(chunk);
        }
        const turn = Math.max(0, this.room.turn - 1);
        for (const line of chunk.split("\n")) {
            this.room.add(line);
            this.log.push(line);
            if (isAction(line)) {
                if (this.turns[turn] === undefined) {
                    this.turns[turn] = [];
                }
                this.turns[turn].push(line);
            }
        }
        return isActionRequired(chunk, this.room?.request ?? {});
    }
    async writeState(outStream: WriteStream, state: Buffer) {
        console.log(state);
        // if (this.debug) {
        //     await outStream.write(state);
        //     await this.stream.write(
        //         Math.random() < 0.5
        //             ? getRandomAction(this.room.request)
        //             : "default"
        //     );
        // } else {
        //     await outStream.write(state);
        // }
    }
    async receiveChunk(outStream: WriteStream, value: string) {
        let act: boolean, state: Buffer;

        act = this.receive(value);
        if (act) {
            state = this.getState();
            await this.writeState(outStream, state);
            this.prevRequest = this.room.request;
            this.room.request = null;
        }
    }
    async pipeTo(outStream: WriteStream) {
        let value: string | string[], done: any, state: Buffer;

        const id = v4();

        while ((({ value, done } = await this.stream.next()), !done)) {
            try {
                this.receiveChunk(outStream, value);
            } catch (err) {
                console.error(err + "\n\n");
                fs.writeFileSync(
                    `debug/logs/${id}-${this.game.gameIndex}-${this.playerIndex}.json`,
                    JSON.stringify({
                        log: this.log,
                        inputLog: this.game.stream.battle.inputLog,
                    })
                );
            }
        }
        state = this.getState();
        await outStream.write(state);
        return outStream.writeEnd();
    }
    destroy() {
        this.stream.destroy();
        this.room.destroy();
    }
}
