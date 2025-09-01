# app.py — TEAMS Networking (Adivina Quién, Reto Foto, Conexión Alfa)
# -------------------------------------------------------------------
# Requisitos de entorno:
#   DATABASE_URL=<postgres URI>
#   FLASK_SECRET=<cualquier cadena segura>
#   ADMIN_TOKEN=<clave para entrar a /admin_panel>
#
# Requisitos de carpeta (con permisos de escritura):
#   static/fotos_reto_foto/
#
# Cómo correr:
#   pip install -r requirements.txt
#   python app.py
#   # o en prod: gunicorn app:app

import os
import re
import json
from datetime import datetime
from functools import wraps
from typing import Tuple

from flask import (
    Flask, render_template, render_template_string, request, jsonify, session,
    redirect, url_for, flash, send_from_directory
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# Carga ENV y Flask
# ─────────────────────────────────────────────────────────────
load_dotenv(override=True)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.getenv("FLASK_SECRET", "change_me")

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "letmein")
FOTOS_DIR = os.path.join("static", "fotos_reto_foto")
os.makedirs(FOTOS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# DB: PostgreSQL con pool
# ─────────────────────────────────────────────────────────────
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Falta DATABASE_URL")

pool = SimpleConnectionPool(
    minconn=1, maxconn=12, dsn=DATABASE_URL,
    cursor_factory=psycopg2.extras.RealDictCursor
)

def db_conn():
    conn = pool.getconn()
    conn.autocommit = True
    return conn

def db_return(conn):
    if conn:
        pool.putconn(conn)

def query(sql: str, params: Tuple = ()):
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if cur.description:
                return cur.fetchall()
            return []
    finally:
        db_return(conn)

def execute(sql: str, params: Tuple = ()):
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
    finally:
        db_return(conn)

# ─────────────────────────────────────────────────────────────
# Esquema mínimo (se crea/ajusta al iniciar)
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

-- Resultados del juego Adivina (guardamos por “sesión” de juego)
CREATE TABLE IF NOT EXISTS adivina_scores (
  id SERIAL PRIMARY KEY,
  jugador_id INTEGER NOT NULL REFERENCES jugadores(id) ON DELETE CASCADE,
  aciertos INTEGER NOT NULL,
  puntaje INTEGER NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Reto Foto (una foto por jugador)
CREATE TABLE IF NOT EXISTS reto_foto (
  id SERIAL PRIMARY KEY,
  jugador_id INTEGER NOT NULL UNIQUE REFERENCES jugadores(id) ON DELETE CASCADE,
  filename TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Votos: cada votante puede repartir EXACTAMENTE 3 puntos entre fotos
CREATE TABLE IF NOT EXISTS reto_foto_votos (
  id SERIAL PRIMARY KEY,
  foto_id INTEGER NOT NULL REFERENCES reto_foto(id) ON DELETE CASCADE,
  votante_id INTEGER NOT NULL REFERENCES jugadores(id) ON DELETE CASCADE,
  puntos INTEGER NOT NULL CHECK (puntos BETWEEN 1 AND 3),
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  UNIQUE (foto_id, votante_id)
);

-- Conexión Alfa: emparejamientos 1-a-1 TF-IDF (guardamos bidireccional)
CREATE TABLE IF NOT EXISTS conexion_matches (
  id SERIAL PRIMARY KEY,
  jugador_1 INTEGER NOT NULL REFERENCES jugadores(id) ON DELETE CASCADE,
  jugador_2 INTEGER NOT NULL REFERENCES jugadores(id) ON DELETE CASCADE,
  score REAL NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  UNIQUE (jugador_1),
  UNIQUE (jugador_2),
  CHECK (jugador_1 <> jugador_2)
);

-- Catálogo opcional de retos para poblar /index desde DB (fallback si no)
CREATE TABLE IF NOT EXISTS retos (
  id SERIAL PRIMARY KEY,
  nombre TEXT UNIQUE NOT NULL,
  url TEXT,
  tipo TEXT,
  puntos INTEGER NOT NULL DEFAULT 0,
  activo BOOLEAN NOT NULL DEFAULT TRUE
);
"""
def ensure_schema():
    for stmt in [s.strip() for s in DDL.split(";") if s.strip()]:
        execute(stmt + ";")
    # seeds básicos de retos (idempotentes)
    try:
        execute("""
        INSERT INTO retos (nombre, url, tipo, puntos, activo) VALUES
        ('Adivina Quién', '/adivina', 'individual', 0, TRUE),
        ('Reto Foto', '/reto_foto', 'foto', 0, TRUE),
        ('Ver Fotos y Votar', '/ver_fotos_reto_foto', 'foto', 0, TRUE),
        ('Conexión Alfa', '/conexion_alfa', 'ia', 0, TRUE)
        ON CONFLICT (nombre) DO NOTHING;
        """)
    except Exception:
        pass

ensure_schema()

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
        # ¿ya es admin en sesión?
        if session.get("is_admin"):
            return f(*args, **kwargs)

        # ¿pasó token en la URL o en POST?
        tok = request.args.get("token") or request.form.get("token")
        if tok and tok == ADMIN_TOKEN:
            session["is_admin"] = True
            return f(*args, **kwargs)

        # si no, mándalo al formulario de admin
        return redirect(url_for("admin_login"))
    return _wrap

@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        tok = (request.form.get("token") or "").strip()
        if tok == ADMIN_TOKEN:
            session["is_admin"] = True
            flash("Sesión de administrador iniciada.")
            return redirect(url_for("admin_panel"))
        flash("Token incorrecto.")
    # pequeño form inline para no depender de plantillas
    return render_template_string("""
    <html><body style="font-family:Segoe UI;background:#111;color:#eee;padding:30px">
      <h2>🔐 Acceso Administrador</h2>
      <form method="post">
        <input name="token" placeholder="Token de administrador" style="padding:10px;width:300px">
        <button style="padding:10px 16px">Entrar</button>
      </form>
      <p style="margin-top:10px"><a href="{{ url_for('index_page') }}">Volver al inicio</a></p>
    </body></html>
    """)

# ─────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────
def get_jugador_by_correo(correo: str):
    rows = query("SELECT * FROM jugadores WHERE correo=%s", (correo,))
    return rows[0] if rows else None

def upsert_jugador(nombre: str, correo: str):
    row = get_jugador_by_correo(correo)
    if row:
        return row
    execute("INSERT INTO jugadores (nombre, correo) VALUES (%s,%s)", (nombre, correo))
    return get_jugador_by_correo(correo)

def get_respuestas(jugador_id: int):
    rows = query("SELECT * FROM formulario_respuestas WHERE jugador_id=%s", (jugador_id,))
    return rows[0] if rows else None

# ─────────────────────────────────────────────────────────────
# Index / Home
# ─────────────────────────────────────────────────────────────
def _retos_desde_db_o_fallback():
    try:
        rows = query("SELECT id, nombre, COALESCE(activo, TRUE) AS activo, COALESCE(puntos,0) AS puntos FROM retos ORDER BY id ASC")
        if rows:
            # Adaptador simple para que Jinja use reto.activo
            return [type("Reto", (), {"id": r["id"], "nombre": r["nombre"], "activo": bool(r["activo"]), "puntos": r["puntos"]}) for r in rows]
    except Exception:
        pass
    base = [
        {"id": 1, "nombre": "Adivina Quién",     "activo": True, "puntos": 0},
        {"id": 2, "nombre": "Reto Foto",         "activo": True, "puntos": 0},
        {"id": 3, "nombre": "Ver Fotos y Votar", "activo": True, "puntos": 0},
        {"id": 4, "nombre": "Conexión Alfa",     "activo": True, "puntos": 0},
    ]
    return [type("Reto", (), x) for x in base]

@app.route("/")
def home():
    if "jugador_id" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("index_page"))

@app.route("/index")
@login_required
def index_page():
    retos = _retos_desde_db_o_fallback()
    if "jugador" not in session:
        session["jugador"] = session.get("nombre", "Jugador")
    # Debes tener un archivo index.html en la raíz (donde están tus otras plantillas)
    return render_template("index.html", retos=retos)

# ─────────────────────────────────────────────────────────────
# Login / Logout
# ─────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
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
    session["jugador"] = jugador["nombre"]  # para index.html

    # ya no bloqueamos la entrada al index si falta el perfil
    if not get_respuestas(jugador["id"]):
        flash("Completa tu perfil cuando puedas para mejorar los retos.")

    return redirect(url_for("index_page"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ─────────────────────────────────────────────────────────────
# Formulario post-login (alimenta Adivina + Conexión Alfa)
# ─────────────────────────────────────────────────────────────
@app.route("/preguntas_post_login", methods=["GET", "POST"])
@login_required
def preguntas_post_login():
    jugador_id = session["jugador_id"]
    ya_respondio = bool(get_respuestas(jugador_id))

    if request.method == "GET":
        return render_template("preguntas_post_login.html", ya_respondio=ya_respondio)

    campos = ["r2","r3","r4","r6","r8","r9","r10","r12","r13"]
    valores = [request.form.get(k,"").strip() for k in campos]

    if ya_respondio:
        execute(
            "UPDATE formulario_respuestas SET r2=%s,r3=%s,r4=%s,r6=%s,r8=%s,r9=%s,r10=%s,r12=%s,r13=%s WHERE jugador_id=%s",
            (*valores, jugador_id)
        )
    else:
        execute(
            "INSERT INTO formulario_respuestas (jugador_id,r2,r3,r4,r6,r8,r9,r10,r12,r13) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (jugador_id, *valores)
        )

    flash("¡Gracias! Tus respuestas han sido guardadas.")
    return redirect(url_for("index_page"))

# ─────────────────────────────────────────────────────────────
# Adivina Quién
# ─────────────────────────────────────────────────────────────
def _adivina_participantes():
    rows = query("""
      SELECT j.id, j.nombre,
             r.r2, r.r3, r.r4, r.r6, r.r8, r.r9, r.r10, r.r12, r.r13
      FROM jugadores j
      JOIN formulario_respuestas r ON r.jugador_id=j.id
      ORDER BY j.nombre
    """)
    participantes = []
    for x in rows:
        participantes.append({
            "id": x["id"],
            "nombre_completo": x["nombre"],
            "pasion": x["r2"] or "",
            "dato_curioso": x["r3"] or "",
            "pelicula_favorita": x["r4"] or "",
            "deporte_favorito": x["r6"] or "",
            "prenda_imprescindible": x["r8"] or "",
            "mejor_concierto": x["r9"] or "",
            "libro_favorito": x["r10"] or "",
            "mascota": x["r12"] or "",
            "hijos": x["r13"] or "",
        })
    return participantes

@app.route("/adivina")
@login_required
def adivina():
    participantes = _adivina_participantes()
    match_name = None
    my_id = session["jugador_id"]
    m = query("""SELECT cm.jugador_2, j2.nombre AS nombre_2
                 FROM conexion_matches cm
                 JOIN jugadores j2 ON j2.id=cm.jugador_2
                 WHERE cm.jugador_1=%s""", (my_id,))
    if m:
        match_name = m[0]["nombre_2"]
    return render_template("adivina.html", participantes=participantes, match_name=match_name)

@app.route("/adivina_finalizado", methods=["POST"])
@login_required
def adivina_finalizado():
    data = request.get_json(force=True) or {}
    aciertos = int(data.get("aciertos", 0))
    puntaje  = int(data.get("puntaje", 0))
    execute("INSERT INTO adivina_scores (jugador_id,aciertos,puntaje) VALUES (%s,%s,%s)",
            (session["jugador_id"], aciertos, puntaje))
    my_id = session["jugador_id"]
    m = query("""SELECT j2.nombre AS nombre_2
                 FROM conexion_matches cm JOIN jugadores j2 ON j2.id=cm.jugador_2
                 WHERE cm.jugador_1=%s""", (my_id,))
    return jsonify({"ok": True, "message": "¡Resultados guardados!",
                    "match": (m[0]["nombre_2"] if m else None)})

@app.route("/ranking_adivina")
@login_required
def ranking_adivina():
    rows = query("""
      SELECT j.nombre, MAX(s.puntaje) AS puntaje, MAX(s.aciertos) AS aciertos
      FROM adivina_scores s
      JOIN jugadores j ON j.id=s.jugador_id
      GROUP BY j.nombre
      ORDER BY puntaje DESC, aciertos DESC, j.nombre ASC
    """)
    html = ["<h1 style='font-family:Segoe UI'>🏆 Ranking Adivina Quién</h1><ol>"]
    for r in rows:
        html.append(f"<li><b>{r['nombre']}</b> — {r['puntaje']} pts, {r['aciertos']} aciertos</li>")
    html.append("</ol><p><a href='/adivina'>Volver</a></p>")
    return "\n".join(html)

# ─────────────────────────────────────────────────────────────
# Reto Foto: subir, ver, votar (exactamente 3 puntos por votante)
# ─────────────────────────────────────────────────────────────
ALLOWED_EXT = {"png","jpg","jpeg","webp"}

def _allowed(filename:str)->bool:
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXT

@app.route("/reto_foto", methods=["GET", "POST"])
@login_required
def reto_foto():
    if request.method == "GET":
        mine = query("SELECT * FROM reto_foto WHERE jugador_id=%s", (session["jugador_id"],))
        ya_subio = bool(mine)
        # Reusa tu plantilla de estilo si quieres; o reemplázala por una propia
        return render_template("preguntas_post_login.html", ya_respondio=True)
    # POST: subir foto
    if "foto" not in request.files:
        flash("Sube una imagen.")
        return redirect(url_for("reto_foto"))
    f = request.files["foto"]
    if not f.filename or not _allowed(f.filename):
        flash("Formato no permitido.")
        return redirect(url_for("reto_foto"))

    if query("SELECT 1 FROM reto_foto WHERE jugador_id=%s", (session["jugador_id"],)):
        flash("Ya subiste una foto.")
        return redirect(url_for("ver_fotos"))

    filename = f"{session['jugador_id']}_{secure_filename(f.filename)}"
    path = os.path.join(FOTOS_DIR, filename)
    f.save(path)

    execute("INSERT INTO reto_foto (jugador_id,filename) VALUES (%s,%s)",
            (session["jugador_id"], filename))
    flash("Foto subida. ¡Gracias!")
    return redirect(url_for("ver_fotos"))

@app.route("/ver_fotos")
@login_required
def ver_fotos():
    fotos = query("""
      SELECT rf.id, rf.filename, j.nombre AS dueño
      FROM reto_foto rf JOIN jugadores j ON j.id=rf.jugador_id
      ORDER BY rf.created_at DESC
    """)
    prev = query("SELECT COALESCE(SUM(puntos),0) AS total FROM reto_foto_votos WHERE votante_id=%s",
                 (session["jugador_id"],))
    total_prev = prev[0]["total"] if prev else 0

    html = ["<h1 style='font-family:Segoe UI'>📸 Galería — Reparte EXACTAMENTE 3 puntos</h1>"]
    html.append("<form method='post' action='/votar_fotos'>")
    for f in fotos:
        html.append(
            f"<div style='margin:10px 0;padding:10px;border:1px solid #444;border-radius:8px'>"
            f"<img src='/static/fotos_reto_foto/{f['filename']}' style='max-height:120px'><br>"
            f"<b>{f['dueño']}</b><br>"
            f"Puntos: <select name='foto_{f['id']}'><option value='0'>0</option>"
            f"<option>1</option><option>2</option><option>3</option></select></div>"
        )
    html.append("<button>Guardar votos</button></form>")
    if total_prev > 0:
        html.append(f"<p>Llevas {total_prev}/3 puntos asignados.</p>")
    html.append("<p><a href='/ranking_fotos'>Ver Ranking</a></p>")
    return "\n".join(html)

@app.route("/votar_fotos", methods=["POST"])
@login_required
def votar_fotos():
    pairs = []
    total = 0
    for k, v in request.form.items():
        m = re.match(r"foto_(\d+)$", k)
        if not m: 
            continue
        foto_id = int(m.group(1))
        pts = int(v or 0)
        if pts < 0 or pts > 3:
            return "Valor inválido", 400
        if pts > 0:
            pairs.append((foto_id, pts))
            total += pts

    prev = query("SELECT COALESCE(SUM(puntos),0) AS total FROM reto_foto_votos WHERE votante_id=%s",
                 (session["jugador_id"],))
    total_prev = prev[0]["total"] if prev else 0

    if total + total_prev != 3:
        return f"Debes completar EXACTAMENTE 3 puntos (llevas {total_prev}, propones {total}).", 400

    for foto_id, pts in pairs:
        ya = query("""SELECT 1 FROM reto_foto_votos WHERE foto_id=%s AND votante_id=%s""",
                   (foto_id, session["jugador_id"]))
        if ya:
            return "Ya habías votado alguna de estas fotos; no se permiten cambios.", 400
        execute("""INSERT INTO reto_foto_votos (foto_id,votante_id,puntos) VALUES (%s,%s,%s)""",
                (foto_id, session["jugador_id"], pts))

    flash("¡Votos guardados!")
    return redirect(url_for("ver_fotos"))

# Ranking de fotos
@app.route("/ranking_fotos")
@login_required
def ranking_fotos_alias():
    rows = query("""
        SELECT j.nombre, COALESCE(SUM(v.puntos),0) AS votos
        FROM reto_foto rf
        JOIN jugadores j ON j.id = rf.jugador_id
        LEFT JOIN reto_foto_votos v ON v.foto_id = rf.id
        GROUP BY j.nombre
        ORDER BY votos DESC, j.nombre ASC
    """)
    html = ["<h1 style='font-family:Segoe UI'>🏆 Ranking Fotos</h1><ol>"]
    for r in rows:
        html.append(f"<li><b>{r['nombre']}</b> — {int(r['votos'])} pts</li>")
    html.append("</ol><p><a href='/ver_fotos'>Volver</a></p>")
    return "\n".join(html)

# ─────────────────────────────────────────────────────────────
# Conexión Alfa (TF-IDF local 1-a-1)
# ─────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def _perfil_texto(row) -> str:
    campos = [row.get("r2",""),row.get("r3",""),row.get("r4",""),
              row.get("r6",""),row.get("r8",""),row.get("r9",""),
              row.get("r10",""),row.get("r12",""),row.get("r13","")]
    return " | ".join([c for c in campos if c])

def _build_matches():
    rows = query("""
      SELECT j.id, j.nombre, r.*
      FROM jugadores j JOIN formulario_respuestas r ON r.jugador_id=j.id
      ORDER BY j.id
    """)
    if len(rows) < 2:
        return []

    textos = [_perfil_texto(r) for r in rows]
    vec = TfidfVectorizer(min_df=1, max_features=3000, ngram_range=(1,2))
    X = vec.fit_transform(textos)
    S = cosine_similarity(X)
    np.fill_diagonal(S, -1.0)

    triples = []
    N = len(rows)
    for i in range(N):
        for j in range(i+1, N):
            triples.append((S[i, j], i, j))
    triples.sort(key=lambda t: t[0], reverse=True)

    asignado = set()
    parejas = []
    for score, i, j in triples:
        if i in asignado or j in asignado:
            continue
        asignado.add(i); asignado.add(j)
        parejas.append((rows[i]["id"], rows[j]["id"], float(score)))

    execute("DELETE FROM conexion_matches")
    for a,b,sc in parejas:
        execute("INSERT INTO conexion_matches (jugador_1,jugador_2,score) VALUES (%s,%s,%s)", (a,b,sc))
        execute("INSERT INTO conexion_matches (jugador_1,jugador_2,score) VALUES (%s,%s,%s)", (b,a,sc))
    return parejas

@app.route("/conexion_alfa_mi_perfil")
@login_required
def conexion_alfa_mi_perfil():
    me = session["jugador_id"]
    m = query("""SELECT cm.jugador_2, j2.nombre AS nombre_2, cm.score
                 FROM conexion_matches cm JOIN jugadores j2 ON j2.id=cm.jugador_2
                 WHERE cm.jugador_1=%s""", (me,))
    if not m:
        _build_matches()
        m = query("""SELECT cm.jugador_2, j2.nombre AS nombre_2, cm.score
                     FROM conexion_matches cm JOIN jugadores j2 ON j2.id=cm.jugador_2
                     WHERE cm.jugador_1=%s""", (me,))
    nombre_match = m[0]["nombre_2"] if m else None
    score = m[0]["score"] if m else None

    myr = query("SELECT * FROM formulario_respuestas WHERE jugador_id=%s", (me,))
    mr = myr[0] if myr else {}
    nombre = session.get("nombre")

    html = ["<h1 style='font-family:Segoe UI'>🤝 Conexión Alfa</h1>"]
    if nombre_match:
        html.append(f"<p><b>{nombre}</b>, tu match sugerido es <b>{nombre_match}</b> "
                    f"(similitud {score:.2f}).</p>")
        temas = [("🎶 Pasión", "r2"), ("🧠 Dato curioso","r3"), ("🎬 Película","r4"),
                 ("🏀 Deporte","r6"), ("🎤 Concierto","r9"), ("📖 Libro/arte","r10")]
        html.append("<h3>Posibles temas de conversación (basados en tu perfil):</h3><ul>")
        for label, key in temas:
            val = mr.get(key,"")
            if val:
                html.append(f"<li>{label}: {val}</li>")
        html.append("</ul>")
    else:
        html.append("<p>Todavía no hay suficientes personas para emparejar.</p>")
    html.append("<p><a href='/index'>Volver al inicio</a></p>")
    return "\n".join(html)

# ─────────────────────────────────────────────────────────────
# Admin panel
# ─────────────────────────────────────────────────────────────
@app.route("/admin_panel", methods=["GET"])
@admin_required
def admin_panel():
    mensajes = []
    participantes = query("""
      SELECT j.id, j.nombre, j.correo, r.r2, r.r3, r.r4, r.r6, r.r8, r.r9, r.r10, r.r12, r.r13
      FROM jugadores j LEFT JOIN formulario_respuestas r ON r.jugador_id=j.id
      ORDER BY j.nombre
    """)
    resultados = query("""
      SELECT j.nombre, MAX(s.puntaje) AS puntaje, MAX(s.aciertos) AS aciertos
      FROM adivina_scores s JOIN jugadores j ON j.id=s.jugador_id
      GROUP BY j.nombre
      ORDER BY puntaje DESC, aciertos DESC, j.nombre ASC
    """)
    matches = query("""
      SELECT j1.nombre AS nombre_1, j2.nombre AS nombre_2, cm.score
      FROM conexion_matches cm
      JOIN jugadores j1 ON j1.id=cm.jugador_1
      JOIN jugadores j2 ON j2.id=cm.jugador_2
      WHERE cm.jugador_1 < cm.jugador_2
      ORDER BY cm.score DESC
    """)
    # Si tienes la tabla retos, úsala; si no, mandamos fallback
    try:
        retos = query("SELECT id,nombre,activo,puntos FROM retos ORDER BY id ASC")
        retos = [type("Reto", (), r) for r in retos]
    except Exception:
        retos = [type("Reto", (), x) for x in [
            {"id":1,"nombre":"Adivina Quién","activo":True,"puntos":0},
            {"id":2,"nombre":"Reto Foto","activo":True,"puntos":0},
            {"id":3,"nombre":"Conexión Alfa","activo":True,"puntos":0},
        ]]
    equipos = {}
    return render_template("admin_panel.html",
                           mensajes=mensajes,
                           retos=retos,
                           resultados=resultados,
                           participantes=participantes,
                           equipos=equipos,
                           matches_conexion=matches)

@app.route("/eliminar_todos_los_jugadores", methods=["POST"])
@admin_required
def eliminar_todos_los_jugadores():
    execute("DELETE FROM conexion_matches")
    execute("DELETE FROM reto_foto_votos")
    execute("DELETE FROM reto_foto")
    execute("DELETE FROM adivina_scores")
    execute("DELETE FROM formulario_respuestas")
    execute("DELETE FROM jugadores")
    # Limpia fotos
    for fn in os.listdir(FOTOS_DIR):
        try: os.remove(os.path.join(FOTOS_DIR, fn))
        except Exception: pass
    flash("Todos los jugadores y datos fueron eliminados.")
    return redirect(url_for("admin_panel"))

@app.route("/reset_reto_equipo_foto", methods=["POST"])
@admin_required
def reset_reto_equipo_foto():
    execute("DELETE FROM reto_foto_votos")
    execute("DELETE FROM reto_foto")
    for fn in os.listdir(FOTOS_DIR):
        try: os.remove(os.path.join(FOTOS_DIR, fn))
        except Exception: pass
    flash("Reto foto reiniciado.")
    return redirect(url_for("admin_panel"))

@app.route("/generar_contenido_adivina", methods=["POST"])
@admin_required
def generar_contenido_adivina():
    flash("Adivina Quién usa las respuestas actuales como base de pistas (no requiere pre-carga).")
    return redirect(url_for("admin_panel"))

@app.route("/generar_matches_conexion_alfa", methods=["POST"])
@admin_required
def generar_matches_conexion_alfa():
    parejas = _build_matches()
    flash(f"Se generaron {len(parejas)} parejas (1-a-1) para Conexión Alfa.")
    return redirect(url_for("admin_panel"))

# ─────────────────────────────────────────────────────────────
# Rutas alias para compatibilidad con tu index
# ─────────────────────────────────────────────────────────────
@app.route("/ver_fotos_reto_foto")
@login_required
def ver_fotos_reto_foto_alias():
    return redirect(url_for("ver_fotos"))

@app.route("/conexion_alfa")
@login_required
def conexion_alfa_alias():
    return redirect(url_for("conexion_alfa_mi_perfil"))

@app.route("/foto_reto/<int:any_id>")
@login_required
def foto_reto_alias(any_id):
    # Tus tarjetas "Foto RETO 1/2/3" pueden apuntar aquí; redirigimos al flujo de foto
    return redirect(url_for("reto_foto"))

# archivos estáticos de fotos
@app.route("/static/fotos_reto_foto/<path:filename>")
def fotos_static(filename):
    return send_from_directory(FOTOS_DIR, filename)

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(FOTOS_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
