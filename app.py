# app.py — Mini TEAMS: Login → Preguntas → Adivina Quién
import os
import json
import random
from functools import wraps
from typing import Tuple

from flask import (
    Flask, render_template, render_template_string, request, session,
    redirect, url_for, flash, jsonify
)
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras

load_dotenv(override=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET", "change-me")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "letmein")

# ─────────────────────────────────────────────────────────────
# Decoradores
# ─────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def _wrap(*args, **kwargs):
        if "jugador_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return _wrap

def admin_required(f):
    @wraps(f)
    def _wrap(*args, **kwargs):
        if session.get("is_admin"):
            return f(*args, **kwargs)
        tok = request.args.get("token") or request.form.get("token")
        if tok and tok == ADMIN_TOKEN:
            session["is_admin"] = True
            flash("Sesión de administrador iniciada.")
            return f(*args, **kwargs)
        return redirect(url_for("admin_login"))
    return _wrap

# ─────────────────────────────────────────────────────────────
# DB helpers (Postgres + retry; sslmode=require)
# ─────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Falta DATABASE_URL")

def _dsn_with_ssl(url: str) -> str:
    if "sslmode=" in url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}sslmode=require"

def db_connect():
    return psycopg2.connect(
        _dsn_with_ssl(DATABASE_URL),
        cursor_factory=psycopg2.extras.RealDictCursor
    )

def execute(sql: str, params: Tuple = ()):
    for _ in (1, 2):
        conn = None
        try:
            conn = db_connect()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(sql, params)
            return
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            if conn: conn.close()
            continue
        finally:
            if conn: conn.close()

def query(sql: str, params: Tuple = ()):
    for _ in (1, 2):
        conn = None
        try:
            conn = db_connect()
            with conn.cursor() as cur:
                cur.execute(sql, params)
                if cur.description:
                    return cur.fetchall()
                return []
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            if conn: conn.close()
            continue
        finally:
            if conn: conn.close()
    return []

# ─────────────────────────────────────────────────────────────
# Esquema + normalización
# ─────────────────────────────────────────────────────────────
DDL = """
CREATE TABLE IF NOT EXISTS jugadores (
  id SERIAL PRIMARY KEY,
  nombre TEXT NOT NULL,
  correo TEXT NOT NULL UNIQUE,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS formulario_respuestas (
  jugador_id INTEGER PRIMARY KEY REFERENCES jugadores(id) ON DELETE CASCADE,
  r2  TEXT,  -- pasión
  r3  TEXT,  -- dato curioso
  r4  TEXT,  -- película favorita
  r6  TEXT,  -- deporte favorito
  r8  TEXT,  -- prenda imprescindible
  r9  TEXT,  -- mejor concierto
  r10 TEXT,  -- libro/arte favorito
  r12 TEXT,  -- mascota
  r13 TEXT,  -- hijos
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- una sola marca de finalización por jugador
CREATE TABLE IF NOT EXISTS adivina_scores (
  jugador_id INTEGER PRIMARY KEY REFERENCES jugadores(id) ON DELETE CASCADE,
  aciertos INTEGER NOT NULL DEFAULT 0,
  rondas   INTEGER NOT NULL DEFAULT 0,
  fallos   INTEGER NOT NULL DEFAULT 0,
  puntos_base  INTEGER NOT NULL DEFAULT 0,
  puntos_bonus INTEGER NOT NULL DEFAULT 0,
  puntos_total INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- control de activación de retos
CREATE TABLE IF NOT EXISTS retos (
  id SERIAL PRIMARY KEY,
  nombre TEXT UNIQUE NOT NULL,
  activo BOOLEAN NOT NULL DEFAULT FALSE
);
"""

def ensure_schema():
    for stmt in [s.strip() for s in DDL.split(";") if s.strip()]:
        execute(stmt + ";")
    # seeds
    execute("INSERT INTO retos (nombre,activo) VALUES ('Adivina Quién', FALSE) ON CONFLICT (nombre) DO NOTHING;")
    for nombre in ('MI6 v1', 'MI6 v2', 'MI6 v3'):
        execute("INSERT INTO retos (nombre,activo) VALUES (%s, FALSE) ON CONFLICT (nombre) DO NOTHING;", (nombre,))

def normalize_schema():
    # adivina_scores columnas por si vienes de otra versión
    execute("ALTER TABLE adivina_scores ADD COLUMN IF NOT EXISTS rondas INTEGER NOT NULL DEFAULT 0;")
    execute("ALTER TABLE adivina_scores ADD COLUMN IF NOT EXISTS fallos INTEGER NOT NULL DEFAULT 0;")
    execute("ALTER TABLE adivina_scores ADD COLUMN IF NOT EXISTS puntos_base  INTEGER NOT NULL DEFAULT 0;")
    execute("ALTER TABLE adivina_scores ADD COLUMN IF NOT EXISTS puntos_bonus INTEGER NOT NULL DEFAULT 0;")
    execute("ALTER TABLE adivina_scores ADD COLUMN IF NOT EXISTS puntos_total INTEGER NOT NULL DEFAULT 0;")
    # jugadores columnas defensivas
    sql = r"""
DO $do$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns WHERE table_name='jugadores' AND column_name='nombre'
  ) THEN
    ALTER TABLE jugadores ADD COLUMN nombre TEXT;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns WHERE table_name='jugadores' AND column_name='correo'
  ) THEN
    ALTER TABLE jugadores ADD COLUMN correo TEXT;
  END IF;
END
$do$;
"""
    execute(sql)

ensure_schema()
normalize_schema()

# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────
def get_jugador_by_correo(correo: str):
    rows = query("SELECT * FROM jugadores WHERE correo=%s", (correo,))
    return rows[0] if rows else None

def upsert_jugador(nombre: str, correo: str):
    row = get_jugador_by_correo(correo)
    if row: return row
    execute("INSERT INTO jugadores (nombre, correo) VALUES (%s,%s)", (nombre, correo))
    return get_jugador_by_correo(correo)

def get_respuestas(jugador_id: int):
    rows = query("SELECT * FROM formulario_respuestas WHERE jugador_id=%s", (jugador_id,))
    return rows[0] if rows else None

def readiness_counts():
    tj = query("SELECT COUNT(*) AS c FROM jugadores")[0]["c"]
    tr = query("SELECT COUNT(*) AS c FROM formulario_respuestas")[0]["c"]
    return int(tj), int(tr), int(max(tj - tr, 0))

def reto_activo(nombre: str) -> bool:
    rows = query("SELECT activo FROM retos WHERE nombre=%s", (nombre,))
    return bool(rows and rows[0]["activo"])

def set_reto_activo(nombre: str, activo: bool):
    execute("""
        INSERT INTO retos (nombre, activo)
        VALUES (%s, %s)
        ON CONFLICT (nombre) DO UPDATE SET activo = EXCLUDED.activo;
    """, (nombre, activo))

# ─────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "HEAD"])
def home():
    if request.method == "HEAD":
        return ("", 200)
    return redirect(url_for("index_page"))

@app.route("/index")
@app.route("/index.html")
@login_required
def index_page():
    me = session["jugador_id"]
    ya_respondio = bool(get_respuestas(me))
    return render_template(
        "index.html",
        nombre=session.get("nombre", ""),
        ya_respondio=ya_respondio,
        adivina_activo=reto_activo("Adivina Quién"),
        show_admin=session.get("is_admin", False)  # <- para ocultar botón
    )

@app.route("/login", methods=["GET", "POST"], endpoint="login")
def login_route():
    if request.method == "GET":
        return render_template("login.html")
    nombre = (request.form.get("nombre") or "").strip()
    correo = (request.form.get("correo") or "").strip().lower()
    if not nombre or not correo or "@" not in correo:
        flash("Nombre y correo válidos son requeridos.")
        return redirect(url_for("login"))
    jugador = upsert_jugador(nombre, correo)
    session["jugador_id"] = jugador["id"]
    session["nombre"] = jugador["nombre"]
    session["correo"] = jugador["correo"]
    if not get_respuestas(jugador["id"]):
        return redirect(url_for("preguntas_post_login"))
    return redirect(url_for("adivina"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# --- Admin login aislado (no ligado a jugador) ---
@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        tok = (request.form.get("token") or "").strip()
        if tok == ADMIN_TOKEN:
            session["is_admin"] = True
            flash("Sesión de administrador iniciada.")
            return redirect(url_for("admin_panel"))
        flash("Token incorrecto.")
    return render_template_string("""
    <html><body style="font-family:Segoe UI;background:#111;color:#eee;padding:30px">
      <h2>🔐 Acceso Administrador</h2>
      <form method="post">
        <input name="token" placeholder="Token de administrador" style="padding:10px;width:300px">
        <button style="padding:10px 16px">Entrar</button>
      </form>
      <p style="margin-top:10px"><a href="{{ url_for('home') }}">Volver</a></p>
    </body></html>
    """)

# --- Preguntas (una tarjeta a la vez lo maneja la plantilla/JS) ---
@app.route("/preguntas_post_login", methods=["GET", "POST"])
@login_required
def preguntas_post_login():
    jugador_id = session["jugador_id"]
    ya_respondio = bool(get_respuestas(jugador_id))

    if request.method == "GET":
        prev = get_respuestas(jugador_id) or {}
        return render_template("preguntas_post_login.html",
                               ya_respondio=ya_respondio, respuestas=prev)

    campos = ["r2","r3","r4","r6","r8","r9","r10","r12","r13"]
    valores = [request.form.get(k,"").strip() for k in campos]

    if ya_respondio:
        execute("""
            UPDATE formulario_respuestas
            SET r2=%s,r3=%s,r4=%s,r6=%s,r8=%s,r9=%s,r10=%s,r12=%s,r13=%s
            WHERE jugador_id=%s
        """, (*valores, jugador_id))
    else:
        execute("""
            INSERT INTO formulario_respuestas
            (jugador_id,r2,r3,r4,r6,r8,r9,r10,r12,r13)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (jugador_id, *valores))

    flash("¡Gracias! Tus respuestas fueron guardadas.")
    return redirect(url_for("adivina"))

# --- Util para juego: 5 categorías y 3 pistas random ---
CAMPOS_JUEGO = ["r2","r3","r4","r6","r9"]  # 5 preguntas/respuestas usadas en el juego

import random  # arriba del archivo si no lo tienes

def _participantes_para_juego(mi_id: int, n: int = 5):
    rows = query("""
      SELECT j.id, j.nombre, r.r2, r.r3, r.r4, r.r6, r.r8, r.r9, r.r10, r.r12, r.r13
      FROM jugadores j
      JOIN formulario_respuestas r ON r.jugador_id=j.id
      WHERE j.id <> %s
    """, (mi_id,))

    # baraja y recorta a 5 (si hay menos, usa los que haya)
    random.shuffle(rows)
    if len(rows) > n:
        rows = rows[:n]

    campos = [
        ("🎶 Pasión", "r2"),
        ("🧠 Dato curioso", "r3"),
        ("🎬 Película favorita", "r4"),
        ("🏀 Deporte favorito", "r6"),
        ("👕 Prenda imprescindible", "r8"),
        ("🎤 Mejor concierto", "r9"),
        ("📚 Libro/Arte favorito", "r10"),
        ("🐾 Mascota", "r12"),
        ("👪 Hijos", "r13"),
    ]

    out = []
    for x in rows:
        disponibles = []
        for label, key in campos:
            val = x.get(key)
            if val:
                disponibles.append({"label": label, "text": val})
        random.shuffle(disponibles)
        pistas = disponibles[:3] if len(disponibles) >= 3 else disponibles
        out.append({"id": x["id"], "nombre": x["nombre"], "pistas": pistas})
    return out


@app.route("/adivina")
@login_required
def adivina():
    if not reto_activo("Adivina Quién"):
        flash("Adivina Quién aún no está activo. Espera a que el administrador lo habilite.")
        return redirect(url_for("index_page"))
    me = session["jugador_id"]

    participantes = session.get("adivina_set")
    if not participantes:
        participantes = _participantes_para_juego(me, n=5)
        session["adivina_set"] = participantes  # congelar la selección de esta partida

    return render_template(
        "adivina.html",
        yo=session.get("nombre",""),
        participantes_json=json.dumps(participantes, ensure_ascii=False)
    )

# Puntaje: +10 acierto, −10 fallo, bonus por llegada: 1º +50, 2º +40, 3º +30, 4º +40, resto +10
@app.route("/adivina_finalizado", methods=["POST"])
@login_required
def adivina_finalizado():
    data = request.get_json(force=True) or {}
    aciertos = int(data.get("aciertos", 0))
    fallos = int(data.get("fallos", 0))
    rondas = int(data.get("rondas", aciertos + fallos))

    puntos_base = aciertos * 10 - fallos * 10

    # posición de llegada (antes de insertar el actual)
    pos = query("SELECT COUNT(*) AS c FROM adivina_scores")[0]["c"] + 1
    if   pos == 1: puntos_bonus = 50
    elif pos == 2: puntos_bonus = 40
    elif pos == 3: puntos_bonus = 30
    elif pos == 4: puntos_bonus = 40  # pedido explícito
    else:          puntos_bonus = 10

    puntos_total = puntos_base + puntos_bonus

    # upsert de la marca de finalización (clave primaria jugador_id)
    execute("""
        INSERT INTO adivina_scores (jugador_id, aciertos, rondas, fallos, puntos_base, puntos_bonus, puntos_total)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (jugador_id) DO UPDATE
        SET aciertos=EXCLUDED.aciertos,
            rondas=EXCLUDED.rondas,
            fallos=EXCLUDED.fallos,
            puntos_base=EXCLUDED.puntos_base,
            puntos_bonus=EXCLUDED.puntos_bonus,
            puntos_total=EXCLUDED.puntos_total
    """, (session["jugador_id"], aciertos, rondas, fallos, puntos_base, puntos_bonus, puntos_total))

# limpiar el set congelado para permitir nueva partida con nuevo random
    session.pop("adivina_set", None)
    return jsonify({"ok": True, "pos": pos, "puntos_base": puntos_base, "puntos_bonus": puntos_bonus, "puntos_total": puntos_total})

# ─────────────────────────────────────────────────────────────
# Admin Panel + activación de reto
# ─────────────────────────────────────────────────────────────
@app.route("/admin_panel", methods=["GET", "POST"])
@admin_required
def admin_panel():
    # POST: toggles desde tu HTML
    if request.method == "POST":
        reto_id = request.form.get("reto_id")
        activo  = request.form.get("activo")
        activar_solo = request.form.get("activar_solo")
        if reto_id is not None and activo is not None:
            execute("UPDATE retos SET activo=%s WHERE id=%s", (bool(int(activo)), int(reto_id)))
            flash("Estado del reto actualizado.")
            return redirect(url_for("admin_panel"))
        if activar_solo is not None:
            rid = int(activar_solo)
            execute("UPDATE retos SET activo=FALSE")
            execute("UPDATE retos SET activo=TRUE WHERE id=%s", (rid,))
            flash("Se activó sólo el reto seleccionado.")
            return redirect(url_for("admin_panel"))

    participantes = query("""
      SELECT j.id, j.nombre, j.correo,
             r.r2, r.r3, r.r4, r.r6, r.r8, r.r9, r.r10, r.r12, r.r13
      FROM jugadores j LEFT JOIN formulario_respuestas r ON r.jugador_id=j.id
      ORDER BY j.nombre
    """)

    resultados = query("""
      SELECT j.nombre AS nombre_jugador,
             s.aciertos,
             s.puntos_total AS puntos_extra
      FROM adivina_scores s
      JOIN jugadores j ON j.id = s.jugador_id
      ORDER BY s.puntos_total DESC, s.created_at ASC
    """)

    retos = [type("Reto", (), r) for r in query("SELECT id, nombre, activo FROM retos ORDER BY id ASC")]

    tj, tr, faltan = readiness_counts()
    adivina_activo = reto_activo("Adivina Quién")

    return render_template(
        "admin_panel.html",
        retos=retos,
        resultados=resultados,
        participantes=participantes,
        equipos={},
        matches_conexion=[],
        total_jugadores=tj,
        total_respuestas=tr,
        faltan=faltan,
        adivina_activo=adivina_activo
    )

@app.route("/admin/activar_adivina", methods=["POST"])
@admin_required
def admin_activar_adivina():
    tj, tr, faltan = readiness_counts()
    if tj < 2:
        flash("Se requieren al menos 2 jugadores para activar Adivina Quién.")
        return redirect(url_for("admin_panel"))
    if faltan > 0:
        flash(f"Aún faltan {faltan} jugadores por llenar preguntas.")
        return redirect(url_for("admin_panel"))
    set_reto_activo("Adivina Quién", True)
    flash("✅ Adivina Quién activado. ¡A jugar!")
    return redirect(url_for("admin_panel"))

@app.route("/admin/forzar_adivina", methods=["POST"])
@admin_required
def admin_forzar_adivina():
    set_reto_activo("Adivina Quién", True)
    flash("⚠️ Adivina Quién fue activado manualmente (aunque falten respuestas).")
    return redirect(url_for("admin_panel"))

@app.route("/admin/desactivar_adivina", methods=["POST"])
@admin_required
def admin_desactivar_adivina():
    set_reto_activo("Adivina Quién", False)
    flash("Adivina Quién desactivado.")
    return redirect(url_for("admin_panel"))

# Aux del panel que tu HTML llama
@app.route("/eliminar_todos_los_jugadores", methods=["POST"])
@admin_required
def eliminar_todos_los_jugadores():
    execute("DELETE FROM adivina_scores")
    execute("DELETE FROM formulario_respuestas")
    execute("DELETE FROM jugadores")
    flash("Se eliminaron todos los jugadores y sus datos.")
    return redirect(url_for("admin_panel"))

@app.route("/reset_reto_equipo_foto", methods=["POST"])
@admin_required
def reset_reto_equipo_foto():
    flash("Reto de equipo (fotos) no está habilitado en esta versión.")
    return redirect(url_for("admin_panel"))

@app.route("/ver_fotos_equipo")
@admin_required
def ver_fotos_equipo():
    return "<h3 style='font-family:Segoe UI;color:#fff'>Módulo de fotos por equipo no habilitado en esta versión.</h3>"

@app.route("/generar_contenido_adivina", methods=["POST"])
@admin_required
def generar_contenido_adivina():
    flash("Adivina Quién usa las respuestas actuales (no requiere pre-carga).")
    return redirect(url_for("admin_panel"))

@app.route("/generar_matches_conexion_alfa", methods=["POST"])
@admin_required
def generar_matches_conexion_alfa():
    flash("Conexión Alfa no está habilitado en esta versión mínima.")
    return redirect(url_for("admin_panel"))

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
