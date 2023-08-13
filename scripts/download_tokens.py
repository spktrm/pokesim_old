import csv
import requests


from collections import defaultdict

# Load JSON data
data = requests.get(
    "https://raw.githubusercontent.com/pkmn/randbats/main/data/gen9randombattle.json"
).json()

# Identifying all unique moves, abilities, and items
unique_moves = set()
unique_abilities = set()
unique_items = set()
for species_data in data.values():
    unique_abilities.update(species_data.get("abilities", []))
    unique_items.update(species_data.get("items", []))
    for role_data in species_data.get("roles", {}).values():
        unique_moves.update(role_data.get("moves", []))


# Function to create one-hot encoding mapping
def one_hot_mapping(values):
    mapping = defaultdict(int)
    for idx, value in enumerate(values):
        mapping[value] = idx
    return mapping


# Function to create one-hot encoded array
def one_hot_array(mapping, values):
    array = [0] * len(mapping)
    for value in values:
        array[mapping[value]] = 1
    return array


# Creating one-hot encoding mapping
moves_mapping = one_hot_mapping(unique_moves)
abilities_mapping = one_hot_mapping(unique_abilities)
items_mapping = one_hot_mapping(unique_items)

# Creating one-hot encoded arrays
species_moves, species_abilities, species_items = [], [], []
for species_name, species_data in data.items():
    abilities_values = species_data.get("abilities", [])
    species_abilities.append(
        (species_name, *one_hot_array(abilities_mapping, abilities_values))
    )
    items_values = species_data.get("items", [])
    species_items.append((species_name, *one_hot_array(items_mapping, items_values)))
    moves_values = set()
    for role_data in species_data.get("roles", {}).values():
        moves_values.update(role_data.get("moves", []))
    species_moves.append((species_name, *one_hot_array(moves_mapping, moves_values)))


# Function to write data to CSV
def write_to_csv(file_path, data, headers):
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)


def to_id(name):
    return "".join(c for c in name if c.isalnum()).lower()


def format_data(data):
    return [(to_id(r[0]),) + r[1:] for r in data]


# Writing to CSV
write_to_csv(
    "src/moves.csv",
    format_data(species_moves),
    ["species"] + list(map(to_id, unique_moves)),
)
write_to_csv(
    "src/abilities.csv",
    format_data(species_abilities),
    ["species"] + list(map(to_id, unique_abilities)),
)
write_to_csv(
    "src/items.csv",
    format_data(species_items),
    ["species"] + list(map(to_id, unique_items)),
)
