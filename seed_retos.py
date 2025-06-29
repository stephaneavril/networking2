#!/usr/bin/env python3
"""
seed_retos.py – Agrega los 3 retos de fotografía de equipo a tu base SQLite.

• Si la tabla `retos` NO tiene la columna `url`, el script la crea con
  `ALTER TABLE`.  ❗ Ahora usa los nombres «Foto RETO 1/2/3» en vez de
  «Foto Equipo …».
• Usa `INSERT OR IGNORE` para evitar duplicados.

Ejemplo de uso (en la carpeta donde está tu database.db):
    python seed_retos.py               # usa database.db por defecto
    python seed_retos.py C:/ruta/a/otro.db
"""

import os
import sqlite3
import sys
from typing import List

# ────────────────────────────────────────────────────────────────────────────────
# Utilidades
# ────────────────────────────────────────────────────────────────────────────────

def column_names(cur: sqlite3.Cursor, table: str) -> List[str]:
    """Devuelve la lista de nombres de columna para `table`."""
    cur.execute(f"PRAGMA table_info({table});")
    return [row[1] for row in cur.fetchall()]

# ────────────────────────────────────────────────────────────────────────────────
# Datos base (nuevos nombres)
# ────────────────────────────────────────────────────────────────────────────────
RETOS = [
    ("Foto RETO 1", "/foto_reto/1", "equipo", 0, 0),
    ("Foto RETO 2", "/foto_reto/2", "equipo", 0, 0),
    ("Foto RETO 3", "/foto_reto/3", "equipo", 0, 0),
]

# ────────────────────────────────────────────────────────────────────────────────
# Script principal
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # 1️⃣ Ruta a la base
    db_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "database.db")
    if not os.path.exists(db_path):
        sys.exit(f"❌ No se encontró la base de datos en: {db_path}")
    print(f"→ Conectando a {db_path}\n")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 2️⃣ Garantiza la existencia de la tabla (estructura mínima)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS retos (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre  TEXT UNIQUE,
            tipo    TEXT,
            puntos  INTEGER DEFAULT 0,
            activo  INTEGER DEFAULT 0
        );
        """
    )

    # 3️⃣ Comprueba si existe la columna `url`; si no, la crea.
    cols = column_names(cur, "retos")
    if "url" not in cols:
        print("⚠️ La columna 'url' no existe. Añadiéndola…")
        cur.execute("ALTER TABLE retos ADD COLUMN url TEXT;")
        conn.commit()
        cols.append("url")
        print("   → Columna añadida.\n")

    # 4️⃣ Inserta los retos: construye la sentencia según columnas presentes
    if "url" in cols:
        cur.executemany(
            """INSERT OR IGNORE INTO retos (nombre, url, tipo, puntos, activo)
                   VALUES (?, ?, ?, ?, ?);""",
            RETOS,
        )
    else:
        cur.executemany(
            """INSERT OR IGNORE INTO retos (nombre, tipo, puntos, activo)
                   VALUES (?, ?, ?, ?);""",
            [(r[0], r[2], r[3], r[4]) for r in RETOS],
        )

    conn.commit()

    # 5️⃣ Reporte
    cur.execute("SELECT nombre, activo FROM retos WHERE nombre LIKE 'Foto RETO %';")
    print("Resultados:\n-----------")
    for nombre, activo in cur.fetchall():
        estado = "🆕 insertado" if activo in (0, 1) else "❔ revisa"
        print(f"{nombre:<15} → {estado}")

    conn.close()
    print("\n✅ Seed completado. Ve a tu admin_panel y actívalos cuando quieras.")


if __name__ == "__main__":
    main()
