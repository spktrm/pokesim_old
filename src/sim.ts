import { AnyObject, PRNG, Teams } from "@pkmn/sim";
import { TeamGenerators } from "@pkmn/randoms";
import { ReadStream } from "@pkmn/streams";

import { Generations, ID } from "@pkmn/data";
import { ModdedDex } from "@pkmn/dex";

import { Buffer } from "node:buffer";
import { GameStore } from "./types";
import { Game } from "./game";
import { Player } from "./player";

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

const _PRNG = new PRNG([42, 42, 42, 42]);

function chooseTeamPreview(team: AnyObject[]): string {
    return `default`;
}

function chooseMove(
    active: AnyObject,
    moves: { choice: string; move: AnyObject }[]
): string {
    return _PRNG.sample(moves).choice;
}

function chooseSwitch(
    active: AnyObject | undefined,
    switches: { slot: number; pokemon: AnyObject }[]
): number {
    return _PRNG.sample(switches).slot;
}

// Creates an array of numbers progressing from start up to and including end
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

function getRequest(player: Player): AnyObject {
    return player.prevRequest;
}

function forceSwitchRandomAction(request: AnyObject): string {
    const pokemon = request.side.pokemon;
    const chosen: number[] = [];
    const choices = request.forceSwitch.map(
        (mustSwitch: AnyObject, i: number) => {
            if (!mustSwitch) return `pass`;

            const canSwitch = range(1, 6).filter(
                (j) =>
                    pokemon[j - 1] &&
                    // not active
                    j > request.forceSwitch.length &&
                    // not chosen for a simultaneous switch
                    !chosen.includes(j) &&
                    // not fainted or fainted and using Revival Blessing
                    !!(
                        +!!pokemon[i].reviving ^
                        +!pokemon[j - 1].condition.endsWith(` fnt`)
                    )
            );

            if (!canSwitch.length) return `pass`;
            const target = chooseSwitch(
                request.active,
                canSwitch.map((slot) => ({
                    slot,
                    pokemon: pokemon[slot - 1],
                }))
            );
            chosen.push(target);
            return `switch ${target}`;
        }
    );
    return choices.join(`, `);
}

function activeRandomAction(request: AnyObject): string {
    let [canMegaEvo, canUltraBurst, canZMove, canDynamax, canTerastallize] = [
        true,
        true,
        true,
        true,
        true,
    ];
    const pokemon = request.side.pokemon;
    const chosen: number[] = [];
    const choices = request.active.map((active: AnyObject, i: number) => {
        if (pokemon[i].condition.endsWith(` fnt`) || pokemon[i].commanding)
            return `pass`;

        canMegaEvo = canMegaEvo && active.canMegaEvo;
        canUltraBurst = canUltraBurst && active.canUltraBurst;
        canZMove = canZMove && !!active.canZMove;
        canDynamax = canDynamax && !!active.canDynamax;
        canTerastallize = canTerastallize && !!active.canTerastallize;

        // Determine whether we should change form if we do end up switching
        const change = canMegaEvo || canUltraBurst || canDynamax;
        // If we've already dynamaxed or if we're planning on potentially dynamaxing
        // we need to use the maxMoves instead of our regular moves

        const useMaxMoves =
            (!active.canDynamax && active.maxMoves) || (change && canDynamax);
        const possibleMoves = useMaxMoves
            ? active.maxMoves.maxMoves
            : active.moves;

        let canMove = range(1, possibleMoves.length)
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
        if (canZMove) {
            canMove.push(
                ...range(1, active.canZMove.length)
                    .filter((j) => active.canZMove[j - 1])
                    .map((j) => ({
                        slot: j,
                        move: active.canZMove[j - 1].move,
                        target: active.canZMove[j - 1].target,
                        zMove: true,
                    }))
            );
        }

        // Filter out adjacentAlly moves if we have no allies left, unless they're our
        // only possible move options.
        const hasAlly =
            pokemon.length > 1 && !pokemon[i ^ 1].condition.endsWith(` fnt`);
        const filtered = canMove.filter(
            (m) => m.target !== `adjacentAlly` || hasAlly
        );
        canMove = filtered.length ? filtered : canMove;

        const moves = canMove.map((m) => {
            let move = `move ${m.slot}`;
            // NOTE: We don't generate all possible targeting combinations.
            if (request.active.length > 1) {
                if ([`normal`, `any`, `adjacentFoe`].includes(m.target)) {
                    move += ` ${1 + Math.floor(_PRNG.next() * 2)}`;
                }
                if (m.target === `adjacentAlly`) {
                    move += ` -${(i ^ 1) + 1}`;
                }
                if (m.target === `adjacentAllyOrSelf`) {
                    if (hasAlly) {
                        move += ` -${1 + Math.floor(_PRNG.next() * 2)}`;
                    } else {
                        move += ` -${i + 1}`;
                    }
                }
            }
            if (m.zMove) move += ` zmove`;
            return { choice: move, move: m };
        });

        const canSwitch = range(1, 6).filter(
            (j) =>
                pokemon[j - 1] &&
                // not active
                !pokemon[j - 1].active &&
                // not chosen for a simultaneous switch
                !chosen.includes(j) &&
                // not fainted
                !pokemon[j - 1].condition.endsWith(` fnt`)
        );
        const switches = active.trapped ? [] : canSwitch;

        if (switches.length && (!moves.length || _PRNG.next() > 1)) {
            const target = chooseSwitch(
                active,
                canSwitch.map((slot) => ({ slot, pokemon: pokemon[slot - 1] }))
            );
            chosen.push(target);
            return `switch ${target}`;
        } else if (moves.length) {
            const move = chooseMove(active, moves);
            if (move.endsWith(` zmove`)) {
                canZMove = false;
                return move;
            } else if (change) {
                if (canTerastallize) {
                    canTerastallize = false;
                    return `${move} terastallize`;
                } else if (canDynamax) {
                    canDynamax = false;
                    return `${move} dynamax`;
                } else if (canMegaEvo) {
                    canMegaEvo = false;
                    return `${move} mega`;
                } else {
                    canUltraBurst = false;
                    return `${move} ultra`;
                }
            } else {
                return move;
            }
        } else {
            throw new Error(
                `unable to make choice ${i}. request='${request}',` +
                    ` chosen='${chosen}', (mega=${canMegaEvo}, ultra=${canUltraBurst}, zmove=${canZMove},` +
                    ` dynamax='${canDynamax}', terastallize=${canTerastallize})`
            );
        }
    });
    return choices.join(`, `);
}

export function getRandomAction(request: AnyObject): string {
    let cmd: string;

    if (request.wait) {
        // wait request
        // do nothing
    } else if (request.forceSwitch) {
        cmd = forceSwitchRandomAction(request);
    } else if (request.active) {
        cmd = activeRandomAction(request);
    } else {
        // team preview?
        cmd = chooseTeamPreview(request.side.pokemon);
    }
    return cmd;
}

export function getMaxDamangeAction(request: AnyObject): string {
    let cmd: string;

    if (request.wait) {
        // wait request
        // do nothing
    } else if (request.forceSwitch) {
        cmd = forceSwitchRandomAction(request);
    } else if (request.active) {
    } else {
        // team preview?
        cmd = chooseTeamPreview(request.side.pokemon);
    }
    return cmd;
}

function getHeuristicAction(request: AnyObject): string {
    return "";
}

export function byteToCommand(player: Player, actionIndex: number): string {
    let cmd: string;
    const request = getRequest(player);
    switch (actionIndex) {
        case 65: // default action
            cmd = "default";
            break;
        case 66: // random action
            cmd = getRandomAction(request);
            break;
        case 67: // max dmg action
            cmd = getMaxDamangeAction(request);
            break;
        case 68: // heuristic action
            cmd = getHeuristicAction(request);
            break;
        default:
            if (actionIndex >= 0 && actionIndex < 4) {
                cmd = `move ${actionIndex + 1}`;
            } else if (actionIndex >= 4 && actionIndex < 10) {
                cmd = `switch ${actionIndex + 1 - 4}`;
            }
            break;
    }
    return cmd;
}

export const n = 1;
export const messageSize = 4;

export function start(numGames: number = -1, debug: boolean = false) {
    var stdin = new StdinReadBytes(process.stdin);

    let games: GameStore = {};

    for (let i = 0; i < n; i++) {
        const game = new Game(i, formatid, gens, debug);
        games[i] = game;
        game.run(numGames);
    }

    (async () => {
        let done: any,
            outStream: { write: (arg0: any) => any },
            gameByte: number,
            playerByte: number,
            actionByte: number,
            playerId: string,
            cmd: string,
            value: Buffer,
            player: Player;

        while (
            (({ value, done } = await stdin.nextMessage(messageSize)), !done)
        ) {
            [gameByte, playerByte, actionByte] = value;
            // gameByte -= 97;
            // playerByte -= 97;

            switch (playerByte) {
                case 0:
                    playerId = "p1";
                    break;
                case 1:
                    playerId = "p2";
                    break;
            }

            player = games[gameByte][playerId];
            outStream = player.stream;

            cmd = byteToCommand(player, actionByte);
            await outStream.write(cmd);
        }
    })();
    return games;
}
