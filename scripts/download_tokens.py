import pandas as pd


def to_id(string: str) -> str:
    return "".join(c for c in string if c.isalnum()).lower()


def create_mapping_dataframe(json_data, key):
    unique_elements = set()
    mapping_data = []

    for species, species_data in json_data.iterrows():
        row = {"species": to_id(species)}
        species_roles = species_data.get("roles", {})

        for role_data in species_roles.values():
            unique_elements.update(list(role_data.get(key, [])))
            for element in role_data.get(key, []):
                row[to_id(element)] = 1

        mapping_data.append(row)

    # Create DataFrame with species as index and sorted columns
    columns = ["species"] + sorted(list(set(map(to_id, unique_elements))))
    df = pd.DataFrame(mapping_data, columns=columns)
    df.set_index("species", inplace=True)
    df.fillna(0, inplace=True)
    df = df.astype(int)

    return df


def main():
    formatid = "gen9randombattle"
    url = f"https://raw.githubusercontent.com/pkmn/randbats/main/data/{formatid}.json"
    json_data = pd.read_json(url).transpose()

    # Create DataFrames for moves, items, and abilities
    moves_df = create_mapping_dataframe(json_data, "moves")
    items_df = create_mapping_dataframe(json_data, "items")
    abilities_df = create_mapping_dataframe(json_data, "abilities")

    # Save to CSV files
    moves_df.to_csv("./src/moves.csv")
    items_df.to_csv("./src/items.csv")
    abilities_df.to_csv("./src/abilities.csv")


if __name__ == "__main__":
    main()
