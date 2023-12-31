import re
import json
import requests

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Sequence, Mapping

BASE_URL = "https://raw.githubusercontent.com/pkmn/ps/main/sim/{}.ts"

URLS = [
    BASE_URL.format(fname)
    for fname in [
        "sim/battle-actions",
        "sim/battle-queue",
        "sim/battle-stream",
        "sim/battle",
        "sim/dex-abilities",
        "sim/dex-conditions",
        "sim/dex-data",
        "sim/dex-formats",
        "sim/dex-items",
        "sim/dex-moves",
        "sim/dex-species",
        "sim/dex",
        "sim/exported-global-types",
        "sim/field",
        "sim/global-types",
        "sim/global-variables.d",
        "sim/index",
        "sim/pokemon",
        "sim/prng",
        "sim/side",
        "sim/state",
        "sim/team-validator.",
        "sim/teams",
        "lib/index",
        "lib/streams",
        "lib/utils",
        "data/abilities",
        "data/aliases",
        "data/conditions",
        "data/formats-data",
        "data/index",
        "data/items",
        "data/learnsets",
        "data/legality",
        "data/moves",
        "data/natures",
        "data/pokedex",
        "data/pokemongo",
        "data/rulesets",
        "data/scripts",
        "data/tags",
        "data/typechart",
    ]
]

src = ""


def fetch_url(url):
    req = requests.get(url)
    if req.ok:
        return req.content.decode("utf-8")
    return ""


with ThreadPoolExecutor() as executor:
    for content in tqdm(executor.map(fetch_url, URLS), total=len(URLS)):
        src += content


def to_id(string: str) -> str:
    return "".join(char for char in string.lower() if char.isalnum())


def reduce(arr: Sequence) -> Sequence:
    return list(
        filter(
            lambda x: x,
            list(sorted(set(map(to_id, arr)))),
        )
    )


def enum(arr: Sequence) -> Mapping:
    return {k: v for v, k in enumerate(arr)}


def main():
    volatile_status = set()
    volatile_status.update(re.findall(r"removeVolatile\([\"|\'](.*?)[\"|\']\)", src))
    volatile_status.update(re.findall(r"hasVolatile\([\"|\'](.*?)[\"|\']\)", src))
    volatile_status.update(re.findall(r"volatiles\[[\"|\'](.*?)[\"|\']\]", src))
    volatile_status.update(re.findall(r"volatiles\.(.*?)[\[|\)| ]", src))
    volatile_status.update(re.findall(r"volatileStatus:\s*[\"|\'](.*)[\"|\'],", src))
    volatile_status = reduce(volatile_status)

    weathers = re.findall(r"[\"|\']-weather[\"|\'],\s*[\"|\'](.*)[\"|\'],", src)
    weathers = reduce(weathers)
    weathers = [t.replace("raindance", "rain") for t in weathers]

    side_conditions = re.findall(r"sideCondition:\s*[\"|\'](.*)[\"|\'],", src)
    side_conditions = reduce(side_conditions)

    terrain = re.findall(r"terrain:\s*[\"|\'](.*)[\"|\'],", src)
    terrain = reduce(terrain)
    terrain = [t.replace("terrain", "") for t in terrain]

    pseudoweather = re.findall(r"pseudoWeather\:\s[\"|\'](.*?)[\"|\']", src)
    pseudoweather = reduce(pseudoweather)

    data = {
        "pseudoWeather": enum(pseudoweather),
        "volatileStatus": enum(volatile_status),
        "weathers": enum(weathers),
        "terrain": enum(terrain),
        "sideConditions": enum(side_conditions),
    }

    with open("src/data.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
