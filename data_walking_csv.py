import json

def export_to_jso(data, filename):
    if not data:
        print("No data to export.")
        return

    filepath = f"{filename}.json"
    print(f"Exporting data to {filepath}...")

    with open(filepath, mode="w") as file:
        json.dump(data, file, indent=4)
