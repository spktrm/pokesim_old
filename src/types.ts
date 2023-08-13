import { ObjectReadStream, ObjectReadWriteStream } from "@pkmn/streams";
import { Game } from "./game";
import { SideCondition } from "@pkmn/data";

export type PlayerSpec = { name: string; team: string };
export type BattleStreamsType = {
    omniscient: ObjectReadWriteStream<string>;
    spectator: ObjectReadStream<string>;
    p1: ObjectReadWriteStream<string>;
    p2: ObjectReadWriteStream<string>;
    p3: ObjectReadWriteStream<string>;
    p4: ObjectReadWriteStream<string>;
};

export type GameStore = {
    [k: string]: Game;
};

export type SideConditions = {
    [id: string]: {
        name: SideCondition;
        level: number;
        minDuration: number;
        maxDuration: number;
        remove?: boolean;
    };
};
