import { Field, Pokemon, Side } from "@pkmn/client";
import { AnyObject } from "@pkmn/sim";
import {
    abilityMapping,
    boostsMapping,
    itemMapping,
    moveMapping,
    pokemonMapping,
    pseudoWeatherMapping,
    sideConditionsMapping,
    statusMapping,
    terrainMapping,
    volatileStatusMapping,
    weatherMapping,
} from "./data";
import { SideConditions } from "./types";

function formatKey(key: string): string {
    // Convert to lowercase and remove spaces and non-alphanumeric characters
    return key.toLowerCase().replace(/[\W_]+/g, "");
}

function getMappingValue(
    pokemon: AnyObject,
    mapping: Object,
    key: string
): number {
    let suffix: string;
    if (key === "asone") {
        if (pokemon.baseSpeciesForme === "Calyrex-Shadow") {
            suffix = "spectrier";
        } else if (pokemon.baseSpeciesForme === "Calyrex-Ice") {
            suffix = "glastrier";
        }
        key = key + suffix;
    }
    const value = mapping[key ?? ""];
    if (value === undefined) {
        console.error(`${key} not in ${JSON.stringify(mapping).slice(0, 30)},`);
    }
    return value ?? -1;
}

function getPokemon(pokemon: AnyObject, active: boolean) {
    let moveTokens = [];
    for (let i = 0; i < 4; i++) {
        moveTokens.push(
            getMappingValue(pokemon, moveMapping, pokemon.moves[i])
        );
    }
    return [
        getMappingValue(pokemon, pokemonMapping, formatKey(pokemon.name)),
        getMappingValue(pokemon, itemMapping, pokemon.item),
        getMappingValue(pokemon, abilityMapping, pokemon.ability),
        1000 *
            Math.floor(
                (pokemon.hp ?? 0) / Math.max(pokemon.maxhp ?? 0, 1) || 1
            ),
        active ? 1 : 0,
        pokemon.fainted ? 1 : 0,
        statusMapping[pokemon.status] ?? -1,
        ...moveTokens,
    ];
}

function binaryArrayToNumber(binArray: number[]): number {
    while (binArray.length < 16) {
        binArray.unshift(0);
    }
    let number: number = 0;
    for (let i = 0; i < binArray.length; i++) {
        number = (number << 1) | binArray[i];
    }
    return number;
}

const fillPokemon = getPokemon({ name: "", moves: [] }, false);
const boostsEntries = Object.entries(boostsMapping);

export class Uint16State {
    static getBoosts(actives: Pokemon[]): number[] {
        const boostsVector = Array(actives.length * boostsEntries.length);
        boostsVector.fill(0);
        for (const [activeIndex, activePokemon] of actives.entries()) {
            if (activePokemon !== null) {
                for (const [boost, value] of Object.entries(
                    activePokemon.boosts
                )) {
                    boostsVector[activeIndex + boostsMapping[boost]] = value;
                }
            }
        }
        return boostsVector;
    }

    static getField(field: Field): number[] {
        const pseudoWeatherVector = Array(9);
        pseudoWeatherVector.fill(0);
        for (const [index, [name, pseudoWeather]] of Object.entries(
            field.pseudoWeather
        ).entries()) {
            pseudoWeatherVector[index] = pseudoWeatherMapping[name];
            pseudoWeatherVector[index + 1] = pseudoWeather.minDuration;
            pseudoWeatherVector[index + 2] = pseudoWeather.maxDuration;
        }
        const weatherAndTerrainVector = [
            weatherMapping[field.weatherState.id] ?? -1,
            field.weatherState.minDuration,
            field.weatherState.maxDuration,
            terrainMapping[field.terrainState.id] ?? -1,
            field.terrainState.minDuration,
            field.terrainState.maxDuration,
        ];
        return [...pseudoWeatherVector, ...weatherAndTerrainVector];
    }

    static getVolatileStatus(actives: Pokemon[]): number[] {
        const volatileStatusVector = Array(actives.length * 10);
        volatileStatusVector.fill(0);
        for (const [activeIndex, activePokemon] of actives.entries()) {
            if (activePokemon !== null) {
                for (const [volatileIndex, volatileStatus] of Object.values(
                    activePokemon.volatiles
                ).entries()) {
                    volatileStatusVector[activeIndex + volatileIndex] =
                        (volatileStatusMapping[volatileStatus.id] << 8) |
                        volatileStatus.level;
                }
            }
        }
        return volatileStatusVector;
    }

    static getSideConditions(sideConditions: SideConditions): number[] {
        const sideConditionVector = Array(
            Object.keys(sideConditionsMapping).length
        );
        sideConditionVector.fill(0);
        for (const [name, sideCondition] of Object.entries(sideConditions)) {
            sideConditionVector[sideConditionsMapping[name]] =
                sideCondition.level;
        }
        return sideConditionVector;
    }

    static getTeam(team: Pokemon[], actives: Pokemon[]): number[] {
        let pokemon = [];
        let arr: number[];
        let ident: string;
        const activeIdents = [];
        for (let i = 0; i < actives.length; i++) {
            ident = actives[i]?.ident;
            if (ident !== undefined) {
                activeIdents.push(ident);
            }
        }
        for (let i = 0; i < 6; i++) {
            if (team[i] === undefined) {
                arr = fillPokemon;
            } else {
                arr = this.getPokemon(
                    team[i],
                    activeIdents.includes(team[i].ident)
                );
            }
            pokemon.push(arr);
        }
        return [].concat(...pokemon);
    }

    static getPokemon(pokemon: Pokemon, active: boolean): number[] {
        return getPokemon(pokemon, active);
    }

    static getLegalMask(request: AnyObject, done: boolean): number[] {
        const mask = Array(10);
        if (request === undefined || done) {
            mask.fill(1);
            mask[4] = 0;
        } else {
            mask.fill(0);

            if (request.wait) {
            } else if (request.forceSwitch) {
                const pokemon = request.side.pokemon;
                const forceSwitchLength = request.forceSwitch.length;
                const isReviving = !!pokemon[0].reviving;

                for (let j = 1; j <= 6; j++) {
                    const currentPokemon = pokemon[j - 1];
                    if (
                        currentPokemon &&
                        j > forceSwitchLength &&
                        (isReviving ? 1 : 0) ^
                            (currentPokemon.condition.endsWith(" fnt") ? 0 : 1)
                    ) {
                        mask[j + 3] = 1;
                    }
                }
            } else if (request.active) {
                const pokemon = request.side.pokemon;
                const active = request.active[0];
                const possibleMoves = active.moves ?? [];
                const canSwitch = [];

                for (let j = 1; j <= possibleMoves.length; j++) {
                    const currentMove = possibleMoves[j - 1];
                    if (!currentMove.disabled) {
                        mask[j - 1] = 1;
                    }
                }

                for (let j = 1; j <= 6; j++) {
                    const currentPokemon = pokemon[j - 1];
                    if (
                        currentPokemon &&
                        !currentPokemon.active &&
                        !currentPokemon.condition.endsWith(" fnt")
                    ) {
                        canSwitch.push(j);
                    }
                }

                const switches =
                    active.trapped || active.maybeTrapped ? [] : canSwitch;

                for (let i = 0; i < switches.length; i++) {
                    const slot = switches[i];
                    mask[slot + 3] = 1;
                }
            }
        }

        return mask;
        // return binaryArrayToNumber(mask);
    }
}
