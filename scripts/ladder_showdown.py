import json
import time
import torch
import uvloop
import asyncio
import logging
import random
import requests
import websockets
import numpy as np

from pokesim.utils import preprocess
from pokesim.constants import _NUM_HISTORY
from pokesim.structs import EnvStep
from pokesim.model.main import Model

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


class LoginError(Exception):
    pass


class SaveReplayError(Exception):
    pass


class PSWebsocketClient:
    websocket = None
    address = None
    login_uri = None
    username = None
    password = None
    last_message = None
    last_challenge_time = 0

    @classmethod
    async def create(cls, username, password, address):
        self = PSWebsocketClient()
        self.username = username
        self.password = password
        self.address = "ws://{}/showdown/websocket".format(address)
        self.websocket = await websockets.connect(self.address)
        self.login_uri = "https://play.pokemonshowdown.com/action.php"
        return self

    async def join_room(self, room_name):
        message = "/join {}".format(room_name)
        await self.send_message("", [message])

    async def receive_message(self):
        message = await self.websocket.recv()
        return message

    async def send_message(self, room, message_list):
        message = room + "|" + "|".join(message_list)

        await self.websocket.send(message)
        self.last_message = message

    async def get_id_and_challstr(self):
        while True:
            message = await self.receive_message()
            split_message = message.split("|")
            if split_message[1] == "challstr":
                return split_message[2], split_message[3]

    async def login(self):
        client_id, challstr = await self.get_id_and_challstr()
        if self.password:
            response = requests.post(
                self.login_uri,
                data={
                    "act": "login",
                    "name": self.username,
                    "pass": self.password,
                    "challstr": "|".join([client_id, challstr]),
                },
            )

        else:
            response = requests.post(
                self.login_uri,
                data={
                    "act": "getassertion",
                    "userid": self.username,
                    "challstr": "|".join([client_id, challstr]),
                },
            )

        if response.status_code == 200:
            if self.password:
                response_json = json.loads(response.text[1:])
                if not response_json["actionsuccess"]:
                    raise LoginError("Could not log-in")

                assertion = response_json.get("assertion")
            else:
                assertion = response.text

            message = ["/trn " + self.username + ",0," + assertion]

            await self.send_message("", message)
        else:
            raise LoginError("Could not log-in")

    async def update_team(self, battle_format, team):
        if "random" in battle_format:
            message = ["/utm None"]
        else:
            message = ["/utm {}".format(team)]
        await self.send_message("", message)

    async def challenge_user(self, user_to_challenge, battle_format, team):
        if time.time() - self.last_challenge_time < 10:
            await asyncio.sleep(10)
        await self.update_team(battle_format, team)
        message = ["/challenge {},{}".format(user_to_challenge, battle_format)]
        await self.send_message("", message)
        self.last_challenge_time = time.time()

    async def accept_challenge(self, battle_format, team, room_name):
        if room_name is not None:
            await self.join_room(room_name)

        await self.update_team(battle_format, team)
        username = None
        while username is None:
            msg = await self.receive_message()
            split_msg = msg.split("|")
            if (
                len(split_msg) == 9
                and split_msg[1] == "pm"
                and split_msg[3].strip().replace("!", "").replace("‽", "")
                == self.username
                and split_msg[4].startswith("/challenge")
                and split_msg[5] == battle_format
            ):
                username = split_msg[2].strip()

        message = ["/accept " + username]
        await self.send_message("", message)

    async def search_for_match(self, battle_format, team):
        await self.update_team(battle_format, team)
        message = ["/search {}".format(battle_format)]
        await self.send_message("", message)

    async def leave_battle(self, battle_tag, save_replay=False):
        if save_replay:
            await self.save_replay(battle_tag)

        message = ["/leave {}".format(battle_tag)]
        await self.send_message("", message)

        while True:
            msg = await self.receive_message()
            if battle_tag in msg and "deinit" in msg:
                return

    async def save_replay(self, battle_tag):
        message = ["/savereplay"]
        await self.send_message(battle_tag, message)

        while True:
            msg = await self.receive_message()
            if msg.startswith("|queryresponse|savereplay|"):
                obj = json.loads(msg.replace("|queryresponse|savereplay|", ""))
                log = obj["log"]
                identifier = obj["id"]
                post_response = requests.post(
                    "https://play.pokemonshowdown.com/~~showdown/action.php?act=uploadreplay",
                    data={"log": log, "id": identifier},
                )
                if post_response.status_code != 200:
                    raise SaveReplayError(
                        "POST to save replay did not return a 200: {}".format(
                            post_response.content
                        )
                    )
                break


async def start_battle() -> asyncio.subprocess.Process:
    command = "node dist/src/eval.js 2>> err.log"

    process = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    return process


def _get_action(policy: np.ndarray):
    action = random.choices(
        population=list(range(policy.shape[-1])),
        k=1,
        weights=policy.squeeze().tolist(),
    )
    return action[0]


async def read(
    ps_websocket_client: PSWebsocketClient, battle: asyncio.subprocess.Process, model
):
    hist = []
    battle_tag = await battle.stdout.readline()
    battle_tag = battle_tag.decode().strip()[1:]

    while True:
        data = await battle.stdout.readuntil(b"\n\n")

        rqid = await battle.stdout.readline()
        rqid = rqid.decode().strip()

        env_step = EnvStep.from_data(data)

        hist.append(env_step)
        env_step = EnvStep.from_stack(hist[-_NUM_HISTORY:])
        batch = preprocess(env_step.raw_obs)
        batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        mask = torch.from_numpy(env_step.legal.astype(np.bool_))
        model_output = model(**batch, mask=mask)
        policy = model_output.policy.detach().numpy()
        action = _get_action(policy)

        if action <= 3:
            prefix = "move"
            index = action
        else:
            prefix = "switch"
            index = action - 4

        index += 1

        await ps_websocket_client.send_message(
            battle_tag, [repr([round(v, 2) for v in policy.squeeze().tolist()])]
        )
        await ps_websocket_client.send_message(
            battle_tag, [f"/choose {prefix} {index}", rqid]
        )


async def write(
    ps_websocket_client: PSWebsocketClient, battle: asyncio.subprocess.Process
):
    while True:
        msg = await ps_websocket_client.receive_message()
        logger.info(msg)

        battle.stdin.write(msg.encode("utf-8"))
        await asyncio.sleep(0.1)


async def pokemon_battle(
    ps_websocket_client: PSWebsocketClient, battle_format: str, params
) -> str:
    battle = await start_battle()
    await asyncio.gather(
        write(ps_websocket_client, battle),
        read(ps_websocket_client, battle, params),
    )


class ShowdownConfig:
    username: str = "PokesimBot"
    password: str = "PokesimBot"
    websocket_uri: str = "localhost:8000"
    battle_format: str = "gen9randombattle"
    team: str = "null"


async def main():
    ps_websocket_client = await PSWebsocketClient.create(
        ShowdownConfig.username, ShowdownConfig.password, ShowdownConfig.websocket_uri
    )
    await ps_websocket_client.login()
    await asyncio.sleep(1)

    params = Model()
    init = torch.load("ckpts/ckpt-176131.pt")
    params.load_state_dict(init)

    wins = 0
    losses = 0

    while True:
        # await ps_websocket_client.search_for_match(
        #     ShowdownConfig.battle_format, ShowdownConfig.team
        # )
        await ps_websocket_client.challenge_user(
            "jtwin",
            ShowdownConfig.battle_format,
            ShowdownConfig.team,
        )
        winner = await pokemon_battle(
            ps_websocket_client, ShowdownConfig.battle_format, params
        )
        if winner == ShowdownConfig.username:
            wins += 1
        else:
            losses += 1


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())
