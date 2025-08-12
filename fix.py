import json
from pathlib import Path

# Ruta de la notebook
nb_path = Path("final/modelo_clasifcacion_plantas.ipynb")

# Leer el contenido actual
with nb_path.open(encoding="utf-8") as f:
    nb_data = json.load(f)

# Eliminar solo el bloque metadata.widgets si existe
if "metadata" in nb_data and "widgets" in nb_data["metadata"]:
    print("🔧 Removing broken metadata.widgets...")
    nb_data["metadata"].pop("widgets", None)
else:
    print("✅ No metadata.widgets found, nothing to remove.")

# Guardar de nuevo
with nb_path.open("w", encoding="utf-8") as f:
    json.dump(nb_data, f, ensure_ascii=False, indent=1)

print(f"✅ Fixed notebook saved: {nb_path}")
