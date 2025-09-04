# app.py — Mini TEAMS: Login → Preguntas → Adivina Quién
import os
import json
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
# Decoradores (deben declararse antes de usarse en @app.route)
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
    return psycopg2.connect(_dsn_with_ssl(DATABASE_URL),
                            cursor_factory=psycopg2.extras.RealDictCursor)

def execute(sql: str, params: Tuple = ()):
    # reintenta 1 vez si la conexión se cayó
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
# Esquema mínimo + normalización suave
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

CREATE TABLE IF NOT EXISTS adivina_scores (
  id SERIAL PRIMARY KEY,
  jugador_id INTEGER NOT NULL REFERENCES jugadores(id) ON DELETE CASCADE,
  aciertos INTEGER NOT NULL,
  rondas INTEGER NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Catálogo de retos para activar/desactivar desde admin
CREATE TABLE IF NOT EXISTS retos (
  id SERIAL PRIMARY KEY,
  nombre TEXT UNIQUE NOT NULL,
  activo BOOLEAN NOT NULL DEFAULT FALSE
);
"""

def ensure_schema():
    for stmt in [s.strip() for s in DDL.split(";") if s.strip()]:
        execute(stmt + ";")
    # Seed del reto “Adivina Quién” (inactivo hasta que admin lo active)
    execute("""
        INSERT INTO retos (nombre, activo)
        VALUES ('Adivina Quién', FALSE)
        ON CONFLICT (nombre) DO NOTHING;
    """)

def normalize_schema():
    # Ajustes suaves si venías de otro esquema
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
    try:
        rows = query("SELECT activo FROM retos WHERE nombre=%s", (nombre,))
        # si no existe el registro, lo consideramos inactivo
        return bool(rows and rows[0]["activo"])
    except Exception:
        return False

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
    if "jugador_id" not in session:
        return redirect(url_for("login"))
    # si ya llenó preguntas, envía al juego; si no, al formulario
    if get_respuestas(session["jugador_id"]):
        return redirect(url_for("adivina"))
    return redirect(url_for("preguntas_post_login"))

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

# --- Admin login minimal ---
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

@app.route("/preguntas_post_login", methods=["GET", "POST"])
@login_required
def preguntas_post_login():
    jugador_id = session["jugador_id"]
    ya_respondio = bool(get_respuestas(jugador_id))

    if request.method == "GET":
        prev = get_respuestas(jugador_id) or {}
        # La plantilla muestra UNA tarjeta a la vez (JS) y máx 10 preguntas
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

def _participantes_para_juego(mi_id: int):
    rows = query("""
      SELECT j.id, j.nombre, r.r2, r.r3, r.r4, r.r6, r.r8, r.r9, r.r10, r.r12, r.r13
      FROM jugadores j
      JOIN formulario_respuestas r ON r.jugador_id=j.id
      WHERE j.id <> %s
      ORDER BY j.nombre
    """, (mi_id,))
    out = []
    for x in rows:
        pistas = [p for p in [x["r2"],x["r3"],x["r4"],x["r6"],x["r8"],x["r9"],x["r10"],x["r12"],x["r13"]] if p]
        out.append({"id": x["id"], "nombre": x["nombre"], "pistas": pistas})
    return out

@app.route("/adivina")
@login_required
def adivina():
    # Solo cuando el reto esté activo
    if not reto_activo("Adivina Quién"):
        flash("Adivina Quién aún no está activo. Espera a que el administrador lo habilite.")
        return redirect(url_for("preguntas_post_login"))

    me = session["jugador_id"]
    participantes = _participantes_para_juego(me)
    return render_template("adivina.html",
                           yo=session.get("nombre",""),
                           participantes_json=json.dumps(participantes, ensure_ascii=False))

@app.route("/adivina_finalizado", methods=["POST"])
@login_required
def adivina_finalizado():
    data = request.get_json(force=True) or {}
    aciertos = int(data.get("aciertos", 0))
    rondas = int(data.get("rondas", 0))
    execute("INSERT INTO adivina_scores (jugador_id, aciertos, rondas) VALUES (%s,%s,%s)",
            (session["jugador_id"], aciertos, rondas))
    return jsonify({"ok": True})

# ─────────────────────────────────────────────────────────────
# Admin Panel + activación de reto
# ─────────────────────────────────────────────────────────────
@app.route("/admin_panel", methods=["GET"])
@admin_required
def admin_panel():
    participantes = query("""
      SELECT j.id, j.nombre, j.correo, r.r2, r.r3, r.r4, r.r6, r.r8, r.r9, r.r10, r.r12, r.r13
      FROM jugadores j LEFT JOIN formulario_respuestas r ON r.jugador_id=j.id
      ORDER BY j.nombre
    """)

    resultados = query("""
      SELECT j.nombre,
             MAX(s.aciertos) AS aciertos,
             MIN(s.rondas)   AS mejores_rondas
      FROM adivina_scores s
      JOIN jugadores j ON j.id = s.jugador_id
      GROUP BY j.nombre
      ORDER BY aciertos DESC, mejores_rondas ASC, j.nombre ASC
    """)

    try:
        retos = query("SELECT id, nombre, activo FROM retos ORDER BY id ASC")
        retos = [type("Reto", (), r) for r in retos]
    except Exception:
        retos = [type("Reto", (), {"id":1,"nombre":"Adivina Quién","activo":False})]

    total_jugadores, total_respuestas, faltan = readiness_counts()
    adivina_activo = reto_activo("Adivina Quién")

    return render_template(
        "admin_panel.html",
        mensajes=[],
        retos=retos,
        resultados=resultados,
        participantes=participantes,
        equipos={},
        matches_conexion=[],  # fuera de alcance en esta mini app
        total_jugadores=total_jugadores,
        total_respuestas=total_respuestas,
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

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
