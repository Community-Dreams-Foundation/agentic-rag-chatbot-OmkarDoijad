import os
import json

from src.cli import sanity

if __name__ == "__main__":
    sanity()

os.makedirs("artifacts", exist_ok=True)

output = {
    "status": "ok",
    "message": "Sanity flow executed",
}

with open("artifacts/sanity_output.json", "w") as f:
    json.dump(output, f, indent=2)

print("Sanity output generated.")
