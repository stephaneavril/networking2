# app.py — Mini TEAMS: Login → Preguntas → Adivina Quién
import os
import json
from functools import wraps
from typing import Tuple

from flask import (
    Flask, render_template, request, session, redirect, url_for, flash, jsonify
)
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras

load_dotenv(override=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET", "change-me")

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
    return psycopg2.connect(_dsn_with_ssl(DATABASE_URL), cursor_factory=psycopg2.extras.RealDictCursor)

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
"""

def ensure_schema():
    for stmt in [s.strip() for s in DDL.split(";") if s.strip()]:
        execute(stmt + ";")

def normalize_schema():
    # Agrega columnas nombre/correo si la tabla existía con otro esquema
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
def login_required(f):
    @wraps(f)
    def _wrap(*args, **kwargs):
        if "jugador_id" not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return _wrap

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

# ─────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────
from flask import request

@app.route("/", methods=["GET", "HEAD"])
def home():
    if request.method == "HEAD":
        return ("", 200)
    if "jugador_id" not in session:
        return redirect("/login")
    # si ya llenó preguntas, envía directo al juego, si no, al formulario
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
        return redirect("/login")
    jugador = upsert_jugador(nombre, correo)
    session["jugador_id"] = jugador["id"]
    session["nombre"] = jugador["nombre"]
    session["correo"] = jugador["correo"]
    # si no ha llenado → preguntas; si sí → juego
    if not get_respuestas(jugador["id"]):
        return redirect(url_for("preguntas_post_login"))
    return redirect(url_for("adivina"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

@app.route("/preguntas_post_login", methods=["GET", "POST"])
@login_required
def preguntas_post_login():
    jugador_id = session["jugador_id"]
    ya = bool(get_respuestas(jugador_id))
    if request.method == "GET":
        return render_template("preguntas_post_login.html", ya_respondio=ya)
    campos = ["r2","r3","r4","r6","r8","r9","r10","r12","r13"]
    valores = [request.form.get(k,"").strip() for k in campos]
    if ya:
        execute("""UPDATE formulario_respuestas 
                   SET r2=%s,r3=%s,r4=%s,r6=%s,r8=%s,r9=%s,r10=%s,r12=%s,r13=%s 
                   WHERE jugador_id=%s""", (*valores, jugador_id))
    else:
        execute("""INSERT INTO formulario_respuestas 
                   (jugador_id,r2,r3,r4,r6,r8,r9,r10,r12,r13) 
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", (jugador_id, *valores))
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
        out.append({
            "id": x["id"],
            "nombre": x["nombre"],
            "pistas": [p for p in [x["r2"],x["r3"],x["r4"],x["r6"],x["r8"],x["r9"],x["r10"],x["r12"],x["r13"]] if p]
        })
    return out

@app.route("/adivina")
@login_required
def adivina():
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
