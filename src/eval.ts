import { Player } from "./player";
import { ID, ModdedDex } from "@pkmn/dex";
import { Generations } from "@pkmn/data";

const formatid: string = "gen9randombattle";
const modid = formatid.slice(0, 4);
const gens = new Generations(new ModdedDex(modid as ID));

const player = new Player(null, 0, null, gens, false);

const inputs = [
    ">battle-gen9randombattle-64\n|init|battle\n|title|PokesimBot vs. jtwin\n|j|\u2606PokesimBot\n",
    ">battle-gen9randombattle-64\n|request|",
    ">battle-gen9randombattle-64\n|j|\u2606jtwin",
    '|pm| jtwin| PokesimBot|/nonotify jtwin accepted the challenge, starting &laquo;<a href="/battle-gen9randombattle-64">battle-gen9randombattle-64</a>&raquo;',
    ">battle-gen9randombattle-64\n|t:|1694137722\n|gametype|singles",
    ">battle-gen9randombattle-64\n|player|p1|PokesimBot|102|",
    '>battle-gen9randombattle-64\n|request|{"active":[{"moves":[{"move":"Shadow Ball","id":"shadowball","pp":24,"maxpp":24,"target":"normal","disabled":false},{"move":"Hyper Voice","id":"hypervoice","pp":16,"maxpp":16,"target":"allAdjacentFoes","disabled":false},{"move":"Psychic","id":"psychic","pp":16,"maxpp":16,"target":"normal","disabled":false},{"move":"Calm Mind","id":"calmmind","pp":32,"maxpp":32,"target":"self","disabled":false}],"canTerastallize":"Psychic"}],"side":{"name":"PokesimBot","id":"p1","pokemon":[{"ident":"p1: Indeedee","details":"Indeedee, L88, M","condition":"249/249","active":true,"stats":{"atk":119,"def":147,"spa":235,"spd":217,"spe":217},"moves":["shadowball","hypervoice","psychic","calmmind"],"baseAbility":"psychicsurge","item":"lifeorb","pokeball":"pokeball","ability":"psychicsurge","commanding":false,"reviving":false,"teraType":"Psychic","terastallized":""},{"ident":"p1: Passimian","details":"Passimian, L84, M","condition":"305/305","active":false,"stats":{"atk":250,"def":199,"spa":115,"spd":149,"spe":183},"moves":["closecombat","uturn","knockoff","earthquake"],"baseAbility":"defiant","item":"choicescarf","pokeball":"pokeball","ability":"defiant","commanding":false,"reviving":false,"teraType":"Dark","terastallized":""},{"ident":"p1: Chi-Yu","details":"Chi-Yu, L77","condition":"211/211","active":false,"stats":{"atk":128,"def":168,"spa":252,"spd":229,"spe":199},"moves":["willowisp","darkpulse","nastyplot","fireblast"],"baseAbility":"beadsofruin","item":"heavydutyboots","pokeball":"pokeball","ability":"beadsofruin","commanding":false,"reviving":false,"teraType":"Dark","terastallized":""},{"ident":"p1: Flutter Mane","details":"Flutter Mane, L74","condition":"203/203","active":false,"stats":{"atk":86,"def":124,"spa":243,"spd":243,"spe":243},"moves":["psyshock","shadowball","moonblast","calmmind"],"baseAbility":"protosynthesis","item":"lifeorb","pokeball":"pokeball","ability":"protosynthesis","commanding":false,"reviving":false,"teraType":"Psychic","terastallized":""},{"ident":"p1: Garganacl","details":"Garganacl, L81, F","condition":"295/295","active":false,"stats":{"atk":209,"def":257,"spa":120,"spd":192,"spe":103},"moves":["saltcure","earthquake","protect","recover"],"baseAbility":"purifyingsalt","item":"leftovers","pokeball":"pokeball","ability":"purifyingsalt","commanding":false,"reviving":false,"teraType":"Ghost","terastallized":""},{"ident":"p1: Zangoose","details":"Zangoose, L86, M","condition":"266/266","active":false,"stats":{"atk":247,"def":152,"spa":152,"spd":152,"spe":204},"moves":["nightslash","closecombat","facade","swordsdance"],"baseAbility":"toxicboost","item":"toxicorb","pokeball":"pokeball","ability":"toxicboost","commanding":false,"reviving":false,"teraType":"Normal","terastallized":""}]},"rqid":2}',
    ">battle-gen9randombattle-64\n|player|p2|jtwin|169|\n|teamsize|p1|6\n|teamsize|p2|6\n|gen|9\n|tier|[Gen 9] Random Battle\n|rule|Species Clause: Limit one of each Pok\u00e9mon\n|rule|HP Percentage Mod: HP is shown in percentages\n|rule|Sleep Clause Mod: Limit one foe put to sleep\n|\n|t:|1694137722\n|start\n|switch|p1a: Indeedee|Indeedee, L88, M|249/249\n|switch|p2a: Kingambit|Kingambit, L74, M|100/100\n|-fieldstart|move: Psychic Terrain|[from] ability: Psychic Surge|[of] p1a: Indeedee\n|turn|1",
];

let battle_tag: string = "";

function readLine(chunk: string) {
    let act: boolean, state: Buffer;

    try {
        act = player.receive(chunk);
    } catch (err) {
        process.stderr.write(`${err.toString()}\n`);
    }

    if (act) {
        if (battle_tag === "") {
            battle_tag = chunk.split("\n")[0];
            process.stdout.write(`${battle_tag}\n`);
        }

        state = player.getState();

        process.stdout.write(state);
        process.stdout.write(`${player.room.request.rqid}\n`);

        player.prevRequest = player.room.request;
        player.room.request = null;
    }
}

process.stdin.on("data", (data) => {
    const value = data.toString();

    readLine(value);
});

// for (const chunk of inputs) {
//     readLine(chunk);
// }
