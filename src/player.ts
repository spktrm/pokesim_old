import { ObjectReadWriteStream, WriteStream } from "@pkmn/streams";
import { Battle } from "@pkmn/client";
import { AnyObject } from "@pkmn/sim";
import { Generations } from "@pkmn/data";
import { Game } from "./game";
import { Uint16State } from "./state";

export function isActionRequired(chunk: string, request: AnyObject): boolean {
    if (request == null) {
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

export class Player {
    game: Game;
    playerIndex: number;
    done: boolean;
    stream: ObjectReadWriteStream<string>;
    log: any[];
    debug: boolean;
    room: Battle;
    gens: Generations;
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
        this.debug = debug;
        this.room = new Battle(gens);
    }
    getStream() {
        return this.game?.stream;
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
        const baseInfo: number[] = [
            workerIndex,
            this.game.gameIndex,
            this.playerIndex,
            this.done ? 1 : 0,
            this.getWinner(),
            this.room.turn,
        ];
        const legalMask = Uint16State.getLegalMask(
            this.room.request,
            this.done
        );
        const arr = new Int16Array([
            ...baseInfo,
            ...Uint16State.getRequest(this.room.request),
            ...Uint16State.getTeam(this.room.p1.team, this.room.p1.active),
            ...Uint16State.getTeam(this.room.p2.team, this.room.p2.active),
            ...Uint16State.getSideConditions(this.room.p1.sideConditions),
            ...Uint16State.getSideConditions(this.room.p2.sideConditions),
            ...Uint16State.getVolatileStatus(this.room.p1.active),
            ...Uint16State.getVolatileStatus(this.room.p2.active),
            ...Uint16State.getBoosts(this.room.p1.active),
            ...Uint16State.getBoosts(this.room.p2.active),
            ...Uint16State.getField(this.room.field),
            ...legalMask,
        ]);
        const buf = Buffer.from(arr.buffer);
        return buf;
    }
    receive(chunk: string): boolean {
        const err = false;
        if (chunk.startsWith("|error")) {
            // console.log(chunk);
            return true;
        }
        for (const line of chunk.split("\n")) {
            this.room.add(line);
        }
        return isActionRequired(chunk, this.room.request);
    }
    async pipeTo(
        outStream: WriteStream,
        options: {
            noEnd?: boolean;
        } = {}
    ) {
        let value: string | string[], done: any, act: boolean, state: Buffer;
        while ((({ value, done } = await this.stream.next()), !done)) {
            act = this.receive(value);
            if (act) {
                state = this.getState();
                if (this.debug) {
                    await outStream.write(state);
                    this.stream.write(
                        randomAction(
                            Uint16State.getLegalMask(
                                this.room.request,
                                this.done
                            )
                        )
                    );
                } else {
                    await outStream.write(state);
                }
            }
        }
        state = this.getState();
        await outStream.write(state);
        if (!options.noEnd) return outStream.writeEnd();
    }
    destroy() {
        this.stream.destroy();
        this.room.destroy();
    }
}
