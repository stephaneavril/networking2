import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Obtener todos los retos ordenados por nombre y luego por ID (para mantener el primero)
retos = cursor.execute("SELECT id, nombre FROM retos ORDER BY nombre, id").fetchall()

vistos = {}
a_eliminar = []

for id, nombre in retos:
    if nombre in vistos:
        a_eliminar.append(id)  # Duplicado → marcar para eliminar
    else:
        vistos[nombre] = id  # Primera vez que lo vemos → lo conservamos

# Eliminar los duplicados
for id in a_eliminar:
    cursor.execute("DELETE FROM retos WHERE id = ?", (id,))

conn.commit()
conn.close()

print(f"✅ {len(a_eliminar)} retos duplicados eliminados correctamente.")
