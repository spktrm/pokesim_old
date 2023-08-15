import { stdout } from "@pkmn/streams";
import { BattleStreams, Teams } from "@pkmn/sim";
import { Generations } from "@pkmn/data";

import { BattleStreamsType, PlayerSpec } from "./types";
import { Player } from "./player";

export class Game {
    stream: BattleStreams.BattleStream;
    streams: BattleStreamsType;
    p1spec: PlayerSpec;
    p2spec: PlayerSpec;
    p1: Player;
    p2: Player;
    gameIndex: number;
    formatid: string;
    gens: Generations;
    debug: boolean;
    constructor(
        gameIndex: number,
        formatid: string,
        gens: Generations,
        debug: boolean = false,
    ) {
        this.gameIndex = gameIndex;
        this.formatid = formatid;
        this.gens = gens;
        this.debug = debug;
    }

    async run() {
        while (true) {
            await this.play();
        }
    }

    play() {
        return new Promise<void>((resolve) => {
            this.stream = new BattleStreams.BattleStream();
            this.streams = BattleStreams.getPlayerStreams(this.stream);
            this.p1spec = {
                name: `Bot${this.gameIndex}1`,
                team: Teams.pack(Teams.generate(this.formatid)),
            };
            this.p2spec = {
                name: `Bot${this.gameIndex}2`,
                team: Teams.pack(Teams.generate(this.formatid)),
            };

            this.p1 = new Player(
                this.streams.p1,
                0,
                this,
                this.gens,
                this.debug,
            );
            this.p2 = new Player(
                this.streams.p2,
                1,
                this,
                this.gens,
                this.debug,
            );

            this.p1.pipeTo(stdout());
            this.p2.pipeTo(stdout());
            (async () => {
                for await (const chunk of this.streams.omniscient) {
                    // if (this.debug) {
                    //     console.log(chunk);
                    // }
                }
                this.p1.done = true;
                this.p2.done = true;
                resolve();
            })();

            this.streams.omniscient.write(`>start ${JSON.stringify({
                formatid: this.formatid,
            })}
>player p1 ${JSON.stringify(this.p1spec)}
>player p2 ${JSON.stringify(this.p2spec)}`);
        });
    }
}
