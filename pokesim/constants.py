_N_STATE_BITS = 294
_STATE_BYTES = 2 * _N_STATE_BITS
_LINE_FEED = 10
_DEFAULT_ACTION = 255

_NUM_ACTIVE = 1
_NUM_PLAYERS = 2
_POKEMON_SIZE = 11

_TEAM_SIZE = _POKEMON_SIZE * 6 * 3

_SIDE_CON_OFFSET = _TEAM_SIZE
_SIDE_CON_SIZE = 15 * _NUM_PLAYERS

_VOLAILTE_OFFSET = _SIDE_CON_OFFSET + _SIDE_CON_SIZE
_VOLAILTE_SIZE = 10 * _NUM_PLAYERS * _NUM_ACTIVE

_BOOSTS_OFFSET = _VOLAILTE_OFFSET + _VOLAILTE_SIZE
_BOOSTS_SIZE = 7 * _NUM_PLAYERS * _NUM_ACTIVE

_FIELD_OFFSET = _BOOSTS_OFFSET + _BOOSTS_SIZE
