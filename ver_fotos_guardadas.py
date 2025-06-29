import sqlite3

conn = sqlite3.connect('database.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("📸 Fotos guardadas en reto_foto:")
for row in cursor.execute("SELECT * FROM reto_foto ORDER BY timestamp DESC"):
    print(f"- {row['nombre']} | archivo: {row['archivo']} | reto_id: {row['reto_id']}")
