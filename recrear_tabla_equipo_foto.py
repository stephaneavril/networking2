import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

# Eliminar tabla si ya existe
cursor.execute("DROP TABLE IF EXISTS reto_equipo_foto")

# Crear la nueva tabla con la columna correcta
cursor.execute('''
CREATE TABLE reto_equipo_foto (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    correo TEXT,
    nombre_participante TEXT,
    equipo INTEGER,
    archivo TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()
conn.close()

print("✅ Tabla 'reto_equipo_foto' eliminada y creada correctamente.")
