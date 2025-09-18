# app.py — Mini TEAMS: Login → Preguntas → Adivina Quién + Conexión Alfa + Reto Foto
import os
import json
import random
import re
import math
import time
from functools import wraps
from typing import Tuple
from collections import Counter
from difflib import SequenceMatcher
from threading import Lock
from werkzeug.utils import secure_filename

from flask import (
    Flask, render_template, render_template_string, request, session,
    redirect, url_for, flash, jsonify, send_from_directory
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
                if params:
                    cur.execute(sql, params)
                else:
                    cur.execute(sql)
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
                if params:
                    cur.execute(sql, params)
                else:
                    cur.execute(sql)
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
# DDL base (tablas principales)
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
  jugador_id INTEGER PRIMARY KEY REFERENCES jugadores(id) ON DELETE CASCADE,
  aciertos INTEGER NOT NULL DEFAULT 0,
  rondas   INTEGER NOT NULL DEFAULT 0,
  fallos   INTEGER NOT NULL DEFAULT 0,
  puntos_base  INTEGER NOT NULL DEFAULT 0,
  puntos_bonus INTEGER NOT NULL DEFAULT 0,
  puntos_total INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS retos (
  id SERIAL PRIMARY KEY,
  nombre TEXT UNIQUE NOT NULL,
  activo BOOLEAN NOT NULL DEFAULT FALSE
);
"""

# ─────────────────────────────────────────────────────────────
# Esquema + normalización (migraciones defensivas)
# ─────────────────────────────────────────────────────────────
def ensure_schema():
    for stmt in [s.strip() for s in DDL.split(";") if s.strip()]:
        execute(stmt + ";")
    # seeds
    execute("INSERT INTO retos (nombre,activo) VALUES ('Adivina Quién', FALSE) ON CONFLICT (nombre) DO NOTHING;")
    execute("INSERT INTO retos (nombre,activo) VALUES ('Conexión Alfa', FALSE) ON CONFLICT (nombre) DO NOTHING;")
    for nombre in ('MI6 v1', 'MI6 v2', 'MI6 v3'):
        execute("INSERT INTO retos (nombre,activo) VALUES (%s, FALSE) ON CONFLICT (nombre) DO NOTHING;", (nombre,))

def normalize_schema():
    # ---- Adivina: columnas defensivas ----
    execute("ALTER TABLE adivina_scores ADD COLUMN IF NOT EXISTS rondas INTEGER NOT NULL DEFAULT 0;")
    execute("ALTER TABLE adivina_scores ADD COLUMN IF NOT EXISTS fallos INTEGER NOT NULL DEFAULT 0;")
    execute("ALTER TABLE adivina_scores ADD COLUMN IF NOT EXISTS puntos_base  INTEGER NOT NULL DEFAULT 0;")
    execute("ALTER TABLE adivina_scores ADD COLUMN IF NOT EXISTS puntos_bonus INTEGER NOT NULL DEFAULT 0;")
    execute("ALTER TABLE adivina_scores ADD COLUMN IF NOT EXISTS puntos_total INTEGER NOT NULL DEFAULT 0;")
    # Refuerzo para que ON CONFLICT funcione aunque la tabla viniera sin PK
    execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_adivina_jugador ON adivina_scores(jugador_id)")
    execute(r"""
    DO $$BEGIN
      BEGIN
        ALTER TABLE adivina_scores ADD CONSTRAINT adivina_scores_pk PRIMARY KEY (jugador_id);
      EXCEPTION WHEN duplicate_object THEN PERFORM 1;
      WHEN others THEN PERFORM 1;
      END;
    END$$;
    """)

    # ---- Jugadores: columnas defensivas ----
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

    # ========= Conexión Alfa: respuestas =========
    execute("CREATE TABLE IF NOT EXISTS conexion_alfa_respuestas (jugador_id INTEGER)")
    execute(r"""
DO $$BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='conexion_alfa_respuestas' AND column_name='jugador_id'
  ) THEN
    ALTER TABLE conexion_alfa_respuestas ADD COLUMN jugador_id INTEGER;
  END IF;

  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='conexion_alfa_respuestas' AND column_name='id'
  ) THEN
    UPDATE conexion_alfa_respuestas
       SET jugador_id = id
     WHERE jugador_id IS NULL;
  END IF;
END$$;
""")

    for col in ("r1","r2","r3","r4","r5","r6","r7"):
        execute(f"ALTER TABLE conexion_alfa_respuestas ADD COLUMN IF NOT EXISTS {col} TEXT;")

    execute("ALTER TABLE conexion_alfa_respuestas ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();")
    execute("ALTER TABLE conexion_alfa_respuestas ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();")

    execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_ca_respuestas_jugador_id
        ON conexion_alfa_respuestas(jugador_id)
    """)

    # Migración defensiva: mover PK a jugador_id y relajar NOT NULL en columnas legadas
    execute(r"""
DO $$DECLARE
  pk_name text;
  n_nulls int;
BEGIN
  -- Backfill nombre/correo si existieran columnas legadas
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='conexion_alfa_respuestas' AND column_name='correo') THEN
    EXECUTE $SQL$
      UPDATE conexion_alfa_respuestas car
         SET correo = j.correo
        FROM jugadores j
       WHERE car.jugador_id = j.id
         AND (car.correo IS NULL OR car.correo = '')
    $SQL$;
  END IF;

  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='conexion_alfa_respuestas' AND column_name='nombre') THEN
    EXECUTE $SQL$
      UPDATE conexion_alfa_respuestas car
         SET nombre = j.nombre
        FROM jugadores j
       WHERE car.jugador_id = j.id
         AND (car.nombre IS NULL OR car.nombre = '')
    $SQL$;
  END IF;

  -- Detecta PK previa en 'correo'
  SELECT c.conname INTO pk_name
  FROM pg_constraint c
  JOIN pg_class t ON t.oid = c.conrelid
  JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(c.conkey)
  WHERE c.contype = 'p' AND t.relname = 'conexion_alfa_respuestas' AND a.attname = 'correo'
  LIMIT 1;

  IF pk_name IS NOT NULL THEN
    SELECT COUNT(*) INTO n_nulls
    FROM conexion_alfa_respuestas
    WHERE jugador_id IS NULL;

    IF n_nulls = 0 THEN
      BEGIN
        ALTER TABLE conexion_alfa_respuestas ADD CONSTRAINT conexion_alfa_respuestas_pkey_jid PRIMARY KEY (jugador_id);
      EXCEPTION WHEN duplicate_object THEN
        PERFORM 1;
      END;

      EXECUTE format('ALTER TABLE conexion_alfa_respuestas DROP CONSTRAINT %I', pk_name);

      IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='conexion_alfa_respuestas'
          AND column_name='correo'
          AND is_nullable='NO'
      ) THEN
        ALTER TABLE conexion_alfa_respuestas ALTER COLUMN correo DROP NOT NULL;
      END IF;
    END IF;
  ELSE
    IF EXISTS (
      SELECT 1
      FROM information_schema.columns
      WHERE table_name='conexion_alfa_respuestas'
        AND column_name='correo'
        AND is_nullable='NO'
    ) THEN
      IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(c.conkey)
        WHERE c.contype = 'p' AND t.relname = 'conexion_alfa_respuestas' AND a.attname = 'correo'
      ) THEN
        ALTER TABLE conexion_alfa_respuestas ALTER COLUMN correo DROP NOT NULL;
      END IF;
    END IF;
  END IF;
END$$;
""")

    # Defensa extra: si todavía existen columnas legadas como NOT NULL, relájalas
    execute(r"""
DO $$BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='conexion_alfa_respuestas' AND column_name='correo'
  ) THEN
    BEGIN
      ALTER TABLE conexion_alfa_respuestas ALTER COLUMN correo DROP NOT NULL;
    EXCEPTION WHEN others THEN PERFORM 1; END;
  END IF;

  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='conexion_alfa_respuestas' AND column_name='nombre'
  ) THEN
    BEGIN
      ALTER TABLE conexion_alfa_respuestas ALTER COLUMN nombre DROP NOT NULL;
    EXCEPTION WHEN others THEN PERFORM 1; END;
  END IF;
END$$;
""")

    # ========= Conexión Alfa: matches =========
    execute("""
        CREATE TABLE IF NOT EXISTS conexion_alfa_matches (
            id SERIAL PRIMARY KEY,
            jugador_1_id INTEGER,
            jugador_2_id INTEGER,
            score FLOAT,
            razon_match TEXT,
            evidencia TEXT,
            feedback SMALLINT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    execute(r"""
DO $$BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns WHERE table_name='conexion_alfa_matches' AND column_name='jugador_1_id'
  ) THEN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='conexion_alfa_matches' AND column_name='j1_id') THEN
      ALTER TABLE conexion_alfa_matches RENAME COLUMN j1_id TO jugador_1_id;
    ELSIF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='conexion_alfa_matches' AND column_name='jugador1_id') THEN
      ALTER TABLE conexion_alfa_matches RENAME COLUMN jugador1_id TO jugador_1_id;
    ELSE
      ALTER TABLE conexion_alfa_matches ADD COLUMN jugador_1_id INTEGER;
    END IF;
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns WHERE table_name='conexion_alfa_matches' AND column_name='jugador_2_id'
  ) THEN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='conexion_alfa_matches' AND column_name='j2_id') THEN
      ALTER TABLE conexion_alfa_matches RENAME COLUMN j2_id TO jugador_2_id;
    ELSIF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='conexion_alfa_matches' AND column_name='jugador2_id') THEN
      ALTER TABLE conexion_alfa_matches RENAME COLUMN jugador2_id TO jugador_2_id;
    ELSE
      ALTER TABLE conexion_alfa_matches ADD COLUMN jugador_2_id INTEGER;
    END IF;
  END IF;

  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='conexion_alfa_matches' AND column_name='score') THEN
    ALTER TABLE conexion_alfa_matches ADD COLUMN score FLOAT NOT NULL DEFAULT 0;
  END IF;

  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='conexion_alfa_matches' AND column_name='razon_match') THEN
    ALTER TABLE conexion_alfa_matches ADD COLUMN razon_match TEXT;
  END IF;

  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='conexion_alfa_matches' AND column_name='evidencia') THEN
    ALTER TABLE conexion_alfa_matches ADD COLUMN evidencia TEXT;
  END IF;

  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='conexion_alfa_matches' AND column_name='feedback') THEN
    ALTER TABLE conexion_alfa_matches ADD COLUMN feedback SMALLINT;
  END IF;
END$$;
""")

    execute("CREATE INDEX IF NOT EXISTS idx_ca_j1 ON conexion_alfa_matches(jugador_1_id)")
    execute("CREATE INDEX IF NOT EXISTS idx_ca_j2 ON conexion_alfa_matches(jugador_2_id)")

    # ========= RETO FOTO =========
    execute("""
        CREATE TABLE IF NOT EXISTS photo_challenges (
            id SERIAL PRIMARY KEY,
            titulo TEXT NOT NULL,
            descripcion TEXT,
            activo BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
    """)

    execute("""
        CREATE TABLE IF NOT EXISTS photo_entries (
            id SERIAL PRIMARY KEY,
            challenge_id INTEGER NOT NULL REFERENCES photo_challenges(id) ON DELETE CASCADE,
            jugador_id INTEGER NOT NULL REFERENCES jugadores(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            UNIQUE (challenge_id, jugador_id)
        );
    """)

    execute("""
        CREATE TABLE IF NOT EXISTS photo_votes (
            id SERIAL PRIMARY KEY,
            challenge_id INTEGER NOT NULL REFERENCES photo_challenges(id) ON DELETE CASCADE,
            entry_id INTEGER NOT NULL REFERENCES photo_entries(id) ON DELETE CASCADE,
            jugador_id INTEGER NOT NULL REFERENCES jugadores(id) ON DELETE CASCADE,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
    """)

    # Quita constraints únicas antiguas (challenge_id, jugador_id) si existieran, para permitir 3 votos por reto
    execute(r"""
DO $$DECLARE
  cons RECORD;
  cols TEXT[];
BEGIN
  FOR cons IN
    SELECT c.conname
    FROM pg_constraint c
    JOIN pg_class t ON t.oid = c.conrelid
    JOIN pg_namespace n ON n.oid = t.relnamespace
    WHERE n.nspname = 'public'
      AND t.relname = 'photo_votes'
      AND c.contype = 'u'
  LOOP
    SELECT array_agg(a.attname ORDER BY a.attnum) INTO cols
    FROM pg_attribute a
    JOIN pg_constraint cc ON cc.conrelid = a.attrelid
    WHERE cc.conname = cons.conname
      AND a.attnum = ANY(cc.conkey);

    IF cols = ARRAY['challenge_id','jugador_id'] THEN
      EXECUTE format('ALTER TABLE photo_votes DROP CONSTRAINT %I', cons.conname);
    END IF;
  END LOOP;
END$$;
""")

    # Índices finales para votos/fotos
    execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_photo_votes_one_per_entry
        ON photo_votes(challenge_id, jugador_id, entry_id);
    """)
    execute("CREATE INDEX IF NOT EXISTS idx_photo_entries_ch ON photo_entries(challenge_id);")
    execute("CREATE INDEX IF NOT EXISTS idx_photo_votes_ch ON photo_votes(challenge_id);")

# 👇 Inicialización
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
# Rutas base
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
    session.pop("adivina_set", None)
    me = session["jugador_id"]
    ya_respondio = bool(get_respuestas(me))

    # Conexión Alfa
    _ensure_tablas_conexion_alfa()
    alfa_ya = bool(query("SELECT 1 FROM conexion_alfa_respuestas WHERE jugador_id=%s", (me,)))

    # Adivina Quién
    adivina = reto_activo("Adivina Quién")

    # >>> Reto Foto (activo por photo_challenges o por flag legado 'retos')
    foto_activo = get_active_photo_challenge()
    if not foto_activo:
        foto_activo = ensure_default_photo_challenge_from_legacy_flag()

    mi_foto = None
    if foto_activo:
        mi_foto = query(
            "SELECT id, filename FROM photo_entries WHERE challenge_id=%s AND jugador_id=%s LIMIT 1",
            (foto_activo["id"], me),
        )
        mi_foto = mi_foto[0] if mi_foto else None

    return render_template(
        "index.html",
        nombre=session.get("nombre", ""),
        ya_respondio=ya_respondio,
        adivina_activo=adivina,
        conexion_alfa_activo=reto_activo("Conexión Alfa"),
        alfa_ya=alfa_ya,
        show_admin=session.get("is_admin", False),
        # ← IMPORTANTE: pasar estas 2 variables
        foto_activo=foto_activo,
        mi_foto=mi_foto,
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
    return redirect(url_for("index_page"))

@app.route("/logout")
def logout():
    session.pop("adivina_set", None)
    session.clear()
    return redirect(url_for("login"))

# --- Admin login aislado ---
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

# --- Preguntas post-login ---
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
    return redirect(url_for("index_page"))

# ─────────────────────────────────────────────────────────────
# Juego Adivina Quién
# ─────────────────────────────────────────────────────────────
def _participantes_para_juego(mi_id: int, n: int = 10):
    rows = query("""
        SELECT j.id, j.nombre,
               r.r2, r.r3, r.r4, r.r6, r.r8, r.r9, r.r10, r.r12, r.r13
        FROM jugadores j
        JOIN formulario_respuestas r ON r.jugador_id = j.id
        WHERE j.id <> %s
    """, (mi_id,))

    random.shuffle(rows)
    rows = rows[:n] if len(rows) > n else rows

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
                txt = val.strip() if isinstance(val, str) else str(val)
                if txt:
                    disponibles.append({"label": label, "text": txt})
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

    # Si el set existe pero está vacío (pudo quedar cacheado), lo regeneramos
    participantes = session.get("adivina_set")
    if not participantes:
        participantes = _participantes_para_juego(me, n=10)  # máximo 10
        session["adivina_set"] = participantes

    # Guardas: si sigue sin haber participantes distintos a ti con respuestas, avisa.
    if not participantes:
        flash("Aún no hay suficientes personas con formulario para jugar. Invita a alguien más a completar sus respuestas 🙂")
        return redirect(url_for("index_page"))

    return render_template(
        "adivina.html",
        yo=session.get("nombre",""),
        participantes_json=json.dumps(participantes, ensure_ascii=False)
    )


@app.route("/adivina_finalizado", methods=["POST"])
@login_required
def adivina_finalizado():
    data = request.get_json(force=True) or {}
    aciertos = int(data.get("aciertos", 0))
    fallos   = int(data.get("fallos", 0))
    rondas   = int(data.get("rondas", aciertos + fallos))

    puntos_base  = aciertos * 10 - fallos * 10

    me = session["jugador_id"]

    # ¿Ya tenía registro?
    ya_tenia = bool(query("SELECT 1 FROM adivina_scores WHERE jugador_id=%s LIMIT 1", (me,)))
    # Posición de llegada (sólo cuenta la primera vez que alguien guarda)
    total_prev = query("SELECT COUNT(*) AS c FROM adivina_scores")[0]["c"]
    pos = total_prev + (0 if ya_tenia else 1)

    if   pos == 1: puntos_bonus = 50
    elif pos == 2: puntos_bonus = 40
    elif pos == 3: puntos_bonus = 30
    elif pos == 4: puntos_bonus = 20
    else:          puntos_bonus = 10

    puntos_total = puntos_base + puntos_bonus

    # 1) Intentar UPDATE (devuelve algo si actualizó)
    updated = query("""
        UPDATE adivina_scores
           SET aciertos=%s,
               rondas=%s,
               fallos=%s,
               puntos_base=%s,
               puntos_bonus=%s,
               puntos_total=%s
         WHERE jugador_id=%s
     RETURNING 1
    """, (aciertos, rondas, fallos, puntos_base, puntos_bonus, puntos_total, me))

    # 2) Si no actualizó, hacer INSERT simple
    if not updated:
        execute("""
            INSERT INTO adivina_scores
              (jugador_id, aciertos, rondas, fallos, puntos_base, puntos_bonus, puntos_total)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (me, aciertos, rondas, fallos, puntos_base, puntos_bonus, puntos_total))

    # limpiar el set de juego
    session.pop("adivina_set", None)

    return jsonify({
        "ok": True,
        "pos": pos,
        "puntos_base": puntos_base,
        "puntos_bonus": puntos_bonus,
        "puntos_total": puntos_total
    })

# ─────────────────────────────────────────────────────────────
# Admin Panel + activación de reto (incluye Reto Foto)
# ─────────────────────────────────────────────────────────────
@app.route("/admin_panel", methods=["GET", "POST"])
@admin_required
def admin_panel():
    if request.method == "POST":
        # ====== Retos generales (toggle/solo) ======
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

        # ====== ADMIN: Reto Foto (crear/activar/cerrar) ======
        if request.form.get("crear_reto_foto") == "1":
            titulo = (request.form.get("titulo") or "").strip()
            descripcion = (request.form.get("descripcion") or "").strip()
            if titulo:
                execute("INSERT INTO photo_challenges (titulo, descripcion, activo) VALUES (%s,%s,FALSE)", (titulo, descripcion))
                flash("Reto Foto creado (inactivo).")
            else:
                flash("Escribe un título para el Reto Foto.")
            return redirect(url_for("admin_panel"))

        if request.form.get("activar_reto_foto"):
            rid = int(request.form["activar_reto_foto"])
            execute("UPDATE photo_challenges SET activo=FALSE")
            execute("UPDATE photo_challenges SET activo=TRUE WHERE id=%s", (rid,))
            flash("Reto Foto activado.")
            return redirect(url_for("admin_panel"))

        if request.form.get("cerrar_reto_foto"):
            rid = int(request.form["cerrar_reto_foto"])
            execute("UPDATE photo_challenges SET activo=FALSE WHERE id=%s", (rid,))
            flash("Reto Foto cerrado.")
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

    # Bloque Reto Foto para admin
    foto_challenges = list_photo_challenges()
    active_ch = get_active_photo_challenge()
    foto_entries = get_photo_entries(active_ch["id"]) if active_ch else []
    foto_votes_map = get_votes_map(active_ch["id"]) if active_ch else {}

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
        adivina_activo=adivina_activo,
        # Reto Foto
        foto_challenges=foto_challenges,
        foto_activo=active_ch,
        foto_entries=foto_entries,
        foto_votes_map=foto_votes_map
    )

# ─────────────────────────────────────────────────────────────
# Reto Foto — Constantes y helpers
# ─────────────────────────────────────────────────────────────
ALLOWED_EXTS = {"jpg", "jpeg", "png", "gif"}

def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def list_photo_challenges():
    return query("SELECT * FROM photo_challenges ORDER BY created_at DESC")

def get_active_photo_challenge():
    rows = query("SELECT * FROM photo_challenges WHERE activo=TRUE ORDER BY created_at DESC LIMIT 1")
    return rows[0] if rows else None

def get_photo_entries(ch_id: int):
    return query("""
        SELECT e.id, e.filename, e.jugador_id, j.nombre
        FROM photo_entries e
        JOIN jugadores j ON j.id=e.jugador_id
        WHERE e.challenge_id=%s
        ORDER BY e.created_at ASC
    """, (ch_id,))

def count_my_votes_in_challenge(ch_id: int, jugador_id: int) -> int:
    rows = query("SELECT COUNT(*) AS c FROM photo_votes WHERE challenge_id=%s AND jugador_id=%s", (ch_id, jugador_id))
    return int(rows[0]["c"]) if rows else 0

def already_voted_entry(ch_id: int, entry_id: int, jugador_id: int) -> bool:
    r = query(
        "SELECT 1 FROM photo_votes WHERE challenge_id=%s AND entry_id=%s AND jugador_id=%s LIMIT 1",
        (ch_id, entry_id, jugador_id),
    )
    return bool(r)

def get_votes_map(ch_id: int):
    rows = query(
        "SELECT entry_id, COUNT(*) AS c FROM photo_votes WHERE challenge_id=%s GROUP BY entry_id",
        (ch_id,),
    )
    return {r["entry_id"]: int(r["c"]) for r in rows}

def ensure_default_photo_challenge_from_legacy_flag():
    ch = get_active_photo_challenge()
    if ch:
        return ch
    if reto_activo("Sube tu foto"):
        rows = query("SELECT id FROM photo_challenges ORDER BY created_at DESC LIMIT 1")
        if not rows:
            execute("""
                INSERT INTO photo_challenges (titulo, descripcion, activo)
                VALUES (%s,%s,TRUE)
            """, ("Sube tu foto", "Comparte tu mejor foto con el equipo",))
        else:
            execute("UPDATE photo_challenges SET activo=FALSE")
            execute("""
                UPDATE photo_challenges
                   SET activo=TRUE
                 WHERE id=(SELECT id FROM photo_challenges ORDER BY created_at DESC LIMIT 1)
            """)
        return get_active_photo_challenge()
    return None

# ─────────────────────────────────────────────────────────────
# Reto Foto — helpers de FS
# ─────────────────────────────────────────────────────────────
def _foto_dir_static(ch_id: int) -> str:
    return os.path.join(app.static_folder, "fotos_ch", str(ch_id))

TMP_UPLOAD_ROOT = "/tmp/uploads_fotos_ch"
def _foto_dir_tmp(ch_id: int) -> str:
    return os.path.join(TMP_UPLOAD_ROOT, str(ch_id))

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Reto Foto — Rutas de usuario
# ─────────────────────────────────────────────────────────────
@app.route("/reto_foto", methods=["GET"])
@login_required
def reto_foto_home():
    ch = get_active_photo_challenge()
    if not ch:
        flash("No hay un Reto Foto activo en este momento.")
        return redirect(url_for("index_page"))
    me = session["jugador_id"]
    mi = query(
        "SELECT id, filename FROM photo_entries WHERE challenge_id=%s AND jugador_id=%s LIMIT 1",
        (ch["id"], me),
    )
    return render_template("reto_foto.html", reto=ch, mi_foto=mi[0] if mi else None)

@app.route("/reto_foto_subir", methods=["POST"])
@login_required
def reto_foto_subir():
    ch = get_active_photo_challenge()
    if not ch:
        flash("No hay Reto Foto activo.")
        return redirect(url_for("index_page"))

    me = session["jugador_id"]
    ya = query("SELECT 1 FROM photo_entries WHERE challenge_id=%s AND jugador_id=%s", (ch["id"], me))
    if ya:
        flash("Ya subiste tu foto para este reto.")
        return redirect(url_for("reto_foto_galeria"))

    f = request.files.get("foto")
    if not f or not f.filename:
        flash("Sube una imagen válida (jpg, jpeg, png, gif).")
        return redirect(url_for("reto_foto_home"))

    if "." not in f.filename:
        flash("El archivo no tiene extensión.")
        return redirect(url_for("reto_foto_home"))
    ext = f.filename.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_EXTS:
        flash("Formato no permitido. Usa jpg, jpeg, png o gif.")
        return redirect(url_for("reto_foto_home"))

    raw_name = f"ch{ch['id']}_u{me}_{int(time.time())}.{ext}"
    fname = secure_filename(raw_name)

    # Intento 1: guardar bajo /static/fotos_ch/<id>/
    d_static = _foto_dir_static(ch["id"])
    _ensure_dir(d_static)
    p_static = os.path.join(d_static, fname)
    saved_path = None
    try:
        f.stream.seek(0)
        f.save(p_static)
        if os.path.exists(p_static):
            saved_path = ("static", fname)
    except Exception as e:
        print(f"[RETO_FOTO] Error guardando en static: {e}")

    # Plan B: /tmp/uploads_fotos_ch/<id>/
    if not saved_path:
        d_tmp = _foto_dir_tmp(ch["id"])
        _ensure_dir(d_tmp)
        p_tmp = os.path.join(d_tmp, fname)
        try:
            f.stream.seek(0)
            f.save(p_tmp)
            if os.path.exists(p_tmp):
                saved_path = ("tmp", fname)
        except Exception as e:
            print(f"[RETO_FOTO] Error guardando en /tmp: {e}")

    if not saved_path:
        flash("No se pudo guardar la foto en el servidor.")
        return redirect(url_for("reto_foto_home"))

    # Persistir en BD
    final_filename = fname if saved_path[0] == "static" else f"tmp:{fname}"
    execute(
        "INSERT INTO photo_entries (challenge_id, jugador_id, filename) VALUES (%s,%s,%s)",
        (ch["id"], me, final_filename),
    )
    flash("¡Foto subida!")
    return redirect(url_for("reto_foto_galeria"))

@app.route("/u_foto/<int:ch_id>/<path:filename>")
@login_required
def serve_tmp_upload(ch_id, filename):
    folder = _foto_dir_tmp(ch_id)
    return send_from_directory(folder, filename, as_attachment=False)

@app.route("/reto_foto_galeria", methods=["GET"])
@login_required
def reto_foto_galeria():
    ch = get_active_photo_challenge()
    if not ch:
        flash("No hay Reto Foto activo.")
        return redirect(url_for("index_page"))

    me = session["jugador_id"]
    entries = get_photo_entries(ch["id"])
    votes_map = get_votes_map(ch["id"])
    mis_votos = count_my_votes_in_challenge(ch["id"], me)
    return render_template(
        "reto_foto_galeria.html",
        reto=ch,
        entries=entries,
        votes_map=votes_map,
        mis_votos=mis_votos,
        votos_max=3,
    )

@app.route("/reto_foto_votar", methods=["POST"])
@login_required
def reto_foto_votar():
    ch = get_active_photo_challenge()
    if not ch:
        flash("No hay Reto Foto activo.")
        return redirect(url_for("index_page"))

    me = session["jugador_id"]
    entry_id = int(request.form.get("entry_id") or 0)

    ent = query("SELECT id, jugador_id FROM photo_entries WHERE id=%s AND challenge_id=%s", (entry_id, ch["id"]))
    if not ent:
        flash("Foto no encontrada.")
        return redirect(url_for("reto_foto_galeria"))
    ent = ent[0]

    if ent["jugador_id"] == me:
        flash("No puedes votar tu propia foto.")
        return redirect(url_for("reto_foto_galeria"))

    if already_voted_entry(ch["id"], entry_id, me):
        flash("Ya votaste esta foto.")
        return redirect(url_for("reto_foto_galeria"))

    usados = count_my_votes_in_challenge(ch["id"], me)
    if usados >= 3:
        flash("Ya usaste tus 3 votos en este reto.")
        return redirect(url_for("reto_foto_galeria"))

    execute(
        "INSERT INTO photo_votes (challenge_id, entry_id, jugador_id) VALUES (%s,%s,%s)",
        (ch["id"], entry_id, me),
    )
    flash("Voto registrado ✅")
    return redirect(url_for("reto_foto_galeria"))

@app.route("/reto_foto_quitar_voto", methods=["POST"])
@login_required
def reto_foto_quitar_voto():
    ch = get_active_photo_challenge()
    if not ch:
        flash("No hay Reto Foto activo.")
        return redirect(url_for("index_page"))

    me = session["jugador_id"]
    entry_id = int(request.form.get("entry_id") or 0)
    execute(
        "DELETE FROM photo_votes WHERE challenge_id=%s AND entry_id=%s AND jugador_id=%s",
        (ch["id"], entry_id, me),
    )
    flash("Voto retirado.")
    return redirect(url_for("reto_foto_galeria"))

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
# Conexión Alfa — Tablas + helpers + rutas
# ─────────────────────────────────────────────────────────────
MATCH_BUILD_LOCK = Lock()

def _ensure_tablas_conexion_alfa():
    execute("""
        CREATE TABLE IF NOT EXISTS conexion_alfa_respuestas (
            jugador_id INTEGER PRIMARY KEY,
            r1 TEXT, r2 TEXT, r3 TEXT, r4 TEXT, r5 TEXT, r6 TEXT, r7 TEXT,
            created_at TIMESTAMP DEFAULT NOW(), updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    execute("""
        CREATE TABLE IF NOT EXISTS conexion_alfa_matches (
            id SERIAL PRIMARY KEY,
            jugador_1_id INTEGER NOT NULL,
            jugador_2_id INTEGER NOT NULL,
            score FLOAT NOT NULL,
            razon_match TEXT,
            evidencia TEXT,
            feedback SMALLINT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    execute("CREATE INDEX IF NOT EXISTS idx_ca_j1 ON conexion_alfa_matches(jugador_1_id)")
    execute("CREATE INDEX IF NOT EXISTS idx_ca_j2 ON conexion_alfa_matches(jugador_2_id)")

def _tok(s: str):
    if not s: return []
    s = s.lower()
    s = re.sub(r"[^\w\sáéíóúüñ]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

def _tf(text: str) -> Counter:
    return Counter(_tok(text or ""))

def _cosine_tf(a: Counter, b: Counter) -> float:
    if not a or not b: return 0.0
    keys = set(a.keys()) | set(b.keys())
    dot = sum(a[k]*b[k] for k in keys)
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    if na == 0 or nb == 0: return 0.0
    return dot/(na*nb)

ALFA_CAMPOS_BASE = [
    ("r2","Pasión"), ("r3","Dato curioso"), ("r4","Película"),
    ("r6","Deporte"), ("r8","Prenda"), ("r9","Concierto"),
    ("r10","Libro/Arte"), ("r12","Mascota"), ("r13","Hijos"),
]

ALFA_CAMPOS_EXTRA = [("r1","Cómo te describes"), ("r2","Qué te encanta"),
                     ("r3","Con qué sueñas"), ("r4","Nunca dirías que no"),
                     ("r5","Tiempo libre"), ("r6","Estilo"), ("r7","Qué con tu equipo")]

def _perfil_texto_agregado(row_base: dict, row_extra: dict) -> str:
    partes = []
    for k,label in ALFA_CAMPOS_BASE:
        v = (row_base or {}).get(k)
        if v: partes.append(f"{label}: {v}")
    for k,label in ALFA_CAMPOS_EXTRA:
        v = (row_extra or {}).get(k)
        if v: partes.append(f"{label}: {v}")
    return " | ".join(partes)

def _razones_campos(row_base1,row_extra1,row_base2,row_extra2, top_k=3):
    pares = []
    for k,label in ALFA_CAMPOS_BASE + ALFA_CAMPOS_EXTRA:
        v1 = (row_base1 or {}).get(k) or (row_extra1 or {}).get(k) or ""
        v2 = (row_base2 or {}).get(k) or (row_extra2 or {}).get(k) or ""
        if v1 and v2:
            ratio = SequenceMatcher(None, v1.lower(), v2.lower()).ratio()
            inter = set(_tok(v1)) & set(_tok(v2))
            ratio += min(len(inter),3)*0.05
            pares.append((ratio,label,v1,v2))
    pares.sort(reverse=True, key=lambda x: x[0])
    return pares[:top_k]

def _explicacion_match(row_base1,row_extra1,row_base2,row_extra2, score):
    razones = _razones_campos(row_base1,row_extra1,row_base2,row_extra2, top_k=3)
    if not razones:
        return f"Compatibilidad general (score {score:.2f})."
    lines = [f"• Afinidad en **{label}** → “{v1}” ~ “{v2}”" for _,label,v1,v2 in razones]
    return "Motivo del match:\n" + "\n".join(lines)

def _get_personas_con_perfil():
    base = {r["jugador_id"]: r for r in query("""
        SELECT r.*, j.id AS jugador_id, j.nombre, j.correo
        FROM formulario_respuestas r
        JOIN jugadores j ON j.id=r.jugador_id
    """)}
    extra = {r["jugador_id"]: r for r in query("""
        SELECT * FROM conexion_alfa_respuestas
    """)}
    personas = []
    for pid in base.keys() | extra.keys():
        jb = base.get(pid, {})
        je = extra.get(pid, {})
        if "nombre" not in jb or "correo" not in jb:
            jrow = query("SELECT nombre, correo FROM jugadores WHERE id=%s", (pid,))
            if jrow:
                if "nombre" not in jb: jb["nombre"] = jrow[0]["nombre"]
                if "correo" not in jb: jb["correo"] = jrow[0]["correo"]
        texto = _perfil_texto_agregado(jb, je)
        if texto.strip():
            personas.append({
                "id": pid,
                "nombre": jb.get("nombre"),
                "correo": jb.get("correo"),
                "base": jb,
                "extra": je
            })
    personas.sort(key=lambda x: x["id"])
    return personas

def _limpiar_matches():
    execute("DELETE FROM conexion_alfa_matches")

def _insertar_match_bidireccional(a_id, b_id, score, razon):
    execute("""INSERT INTO conexion_alfa_matches (jugador_1_id, jugador_2_id, score, razon_match)
               VALUES (%s,%s,%s,%s)""", (a_id,b_id,score,razon))
    execute("""INSERT INTO conexion_alfa_matches (jugador_1_id, jugador_2_id, score, razon_match)
               VALUES (%s,%s,%s,%s)""", (b_id,a_id,score,razon))

def _greedy_one_to_one(pers, sim):
    N = len(pers)
    edges = []
    for i in range(N):
        for j in range(i+1,N):
            edges.append((sim[i][j], i, j))
    edges.sort(reverse=True, key=lambda x: x[0])
    used = set(); pairs=[]
    for sc,i,j in edges:
        if i in used or j in used: continue
        used.add(i); used.add(j)
        pairs.append((i,j,sc))
    return pairs

def _auto_match_para(jugador_id: int):
    """Si el jugador no tiene match aún, calcula su mejor pareja y la guarda."""
    _ensure_tablas_conexion_alfa()
    personas = _get_personas_con_perfil()
    if len(personas) < 2:
        return

    pos_por_id = {p["id"]: i for i, p in enumerate(personas)}
    if jugador_id not in pos_por_id:
        return

    # ¿ya tiene match?
    ya = query("SELECT 1 FROM conexion_alfa_matches WHERE jugador_1_id=%s LIMIT 1", (jugador_id,))
    if ya:
        return

    # Vectorizado simple: similitudes sólo contra este jugador
    tfs = [_tf(_perfil_texto_agregado(p["base"], p["extra"])) for p in personas]
    i = pos_por_id[jugador_id]
    mejor_j, mejor_s = None, -1.0
    for j in range(len(personas)):
        if j == i:
            continue
        s = _cosine_tf(tfs[i], tfs[j])
        if s > mejor_s:
            mejor_s, mejor_j = s, j

    if mejor_j is None:
        return

    p1, p2 = personas[i], personas[mejor_j]
    # Evitar duplicados
    dup = query("""
        SELECT 1 FROM conexion_alfa_matches
        WHERE (jugador_1_id=%s AND jugador_2_id=%s)
           OR (jugador_1_id=%s AND jugador_2_id=%s) LIMIT 1
    """, (p1["id"], p2["id"], p2["id"], p1["id"]))
    if dup:
        return

    razon = _explicacion_match(p1["base"], p1["extra"], p2["base"], p2["extra"], mejor_s)
    _insertar_match_bidireccional(p1["id"], p2["id"], float(mejor_s), razon)

@app.route("/generar_matches_conexion_alfa", methods=["POST"])
@admin_required
def generar_matches_conexion_alfa():
    _ensure_tablas_conexion_alfa()
    if not MATCH_BUILD_LOCK.acquire(blocking=False):
        flash("Otro proceso ya está generando matches. Inténtalo en unos segundos.")
        return redirect(url_for("admin_panel"))

    try:
        personas = _get_personas_con_perfil()
        N = len(personas)
        if N < 2:
            flash("Se requieren al menos 2 participantes con perfil.")
            return redirect(url_for("admin_panel"))

        t0 = time.time()
        tfs = [_tf(_perfil_texto_agregado(p["base"], p["extra"])) for p in personas]
        sim = [[0.0]*N for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                s = _cosine_tf(tfs[i], tfs[j])
                sim[i][j]=sim[j][i]=s

        pairs = _greedy_one_to_one(personas, sim)

        _limpiar_matches()
        for i,j,sc in pairs:
            p1, p2 = personas[i], personas[j]
            razon = _explicacion_match(p1["base"],p1["extra"], p2["base"],p2["extra"], sc)
            _insertar_match_bidireccional(p1["id"], p2["id"], float(sc), razon)

        dt = time.time() - t0
        flash(f"Matches generados: {len(pairs)} parejas (participantes: {N}) en {dt:.2f}s.")
        return redirect(url_for("admin_panel"))
    finally:
        MATCH_BUILD_LOCK.release()

# Formulario Conexión Alfa
@app.route("/conexion_alfa_form", methods=["GET", "POST"])
@login_required
def conexion_alfa_form():
    _ensure_tablas_conexion_alfa()
    me = session["jugador_id"]

    if request.method == "POST":
        data = {f"r{k}": (request.form.get(f"r{k}") or "").strip() for k in range(1, 8)}

        # 💡 Parche inmediato: relaja NOT NULL si quedaron columnas legadas
        try:
            execute("ALTER TABLE conexion_alfa_respuestas ALTER COLUMN correo DROP NOT NULL;")
        except Exception:
            pass
        try:
            execute("ALTER TABLE conexion_alfa_respuestas ALTER COLUMN nombre DROP NOT NULL;")
        except Exception:
            pass

        # Guarda/actualiza las respuestas
        execute("""
            INSERT INTO conexion_alfa_respuestas (jugador_id, r1, r2, r3, r4, r5, r6, r7)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (jugador_id) DO UPDATE
              SET r1=EXCLUDED.r1, r2=EXCLUDED.r2, r3=EXCLUDED.r3,
                  r4=EXCLUDED.r4, r5=EXCLUDED.r5, r6=EXCLUDED.r6, r7=EXCLUDED.r7,
                  updated_at=NOW()
        """, (me, data["r1"], data["r2"], data["r3"], data["r4"], data["r5"], data["r6"], data["r7"]))

        # Intenta generar/asegurar un match y llevar al análisis
        try:
            _auto_match_para(me)
        except Exception:
            pass

        return redirect(url_for("conexion_alfa_mi_match"))

    # GET: mostrar si ya existe el formulario
    ya = query("SELECT 1 FROM conexion_alfa_respuestas WHERE jugador_id=%s", (me,))
    return render_template("conexion_alfa.html", ya_existe=bool(ya))

def _perfil_ia_ligero(base_row, extra_row):
    piezas = []
    for k,label in ALFA_CAMPOS_BASE:
        v = (base_row or {}).get(k)
        if v: piezas.append(f"• {label}: {v}.")
    for k,label in ALFA_CAMPOS_EXTRA:
        v = (extra_row or {}).get(k)
        if v: piezas.append(f"• {label}: {v}.")
    return "\n".join(piezas) or "Perfil en construcción."

@app.route("/conexion_alfa_mi_perfil", methods=["GET"])
@login_required
def conexion_alfa_mi_perfil():
    me = session["jugador_id"]
    r_base = query("SELECT * FROM formulario_respuestas WHERE jugador_id=%s",(me,))
    r_extra = query("SELECT * FROM conexion_alfa_respuestas WHERE jugador_id=%s",(me,))
    if not r_base and not r_extra:
        flash("Completa primero tu formulario.")
        return redirect(url_for("conexion_alfa_form"))
    perfil_txt = _perfil_ia_ligero(r_base[0] if r_base else {}, r_extra[0] if r_extra else {})
    return render_template("conexion_alfa_perfil.html", perfil={"perfil_ia": perfil_txt})

@app.route("/conexion_alfa", methods=["GET"])
@login_required
def conexion_alfa_redirect():
    return redirect(url_for("conexion_alfa_mi_perfil"))

@app.route("/conexion_alfa_mi_match", methods=["GET"])
@login_required
def conexion_alfa_mi_match():
    me = session["jugador_id"]
    m = query("""
        SELECT m.id, m.jugador_1_id, m.jugador_2_id, m.score, m.razon_match, m.evidencia, m.feedback,
               j1.nombre AS nombre_1, j2.nombre AS nombre_2
        FROM conexion_alfa_matches m
        JOIN jugadores j1 ON j1.id = m.jugador_1_id
        JOIN jugadores j2 ON j2.id = m.jugador_2_id
        WHERE m.jugador_1_id=%s
        ORDER BY m.score DESC
        LIMIT 1
    """, (me,))
    if not m:
        return render_template("conexion_alfa_mi_match.html", sin_match=True)

    m = m[0]
    b1 = query("SELECT * FROM formulario_respuestas WHERE jugador_id=%s",(m["jugador_1_id"],))
    e1 = query("SELECT * FROM conexion_alfa_respuestas WHERE jugador_id=%s",(m["jugador_1_id"],))
    b2 = query("SELECT * FROM formulario_respuestas WHERE jugador_id=%s",(m["jugador_2_id"],))
    e2 = query("SELECT * FROM conexion_alfa_respuestas WHERE jugador_id=%s",(m["jugador_2_id"],))
    perfil_1 = _perfil_ia_ligero(b1[0] if b1 else {}, e1[0] if e1 else {})
    perfil_2 = _perfil_ia_ligero(b2[0] if b2 else {}, e2[0] if e2 else {})

    match_dict = {
        "id": m["id"],
        "nombre_1": m["nombre_1"],
        "nombre_2": m["nombre_2"],
        "perfil_1": perfil_1,
        "perfil_2": perfil_2,
        "razon_match": m["razon_match"],
        "evidencia": m["evidencia"]
    }
    return render_template("conexion_alfa_mi_match.html",
                           sin_match=False, match=match_dict, feedback_dado=m["feedback"])

@app.route("/subir_foto_match", methods=["POST"])
@login_required
def subir_foto_match():
    me = session["jugador_id"]
    f = request.files.get("foto")
    if not f or not f.filename:
        flash("Selecciona una imagen válida.")
        return redirect(url_for("conexion_alfa_mi_match"))

    os.makedirs(os.path.join(app.static_folder, "evidencias_alfa"), exist_ok=True)
    fname = f"evid_{me}_{int(time.time())}.jpg"
    f.save(os.path.join(app.static_folder, "evidencias_alfa", fname))

    par = query("SELECT jugador_2_id FROM conexion_alfa_matches WHERE jugador_1_id=%s LIMIT 1", (me,))
    if par:
        other = par[0]["jugador_2_id"]
        execute("UPDATE conexion_alfa_matches SET evidencia=%s WHERE (jugador_1_id=%s AND jugador_2_id=%s) OR (jugador_1_id=%s AND jugador_2_id=%s)",
                (fname, me, other, other, me))
    flash("¡Listo! Evidencia subida.")
    return redirect(url_for("conexion_alfa_mi_match"))

@app.route("/feedback_match", methods=["POST"])
@login_required
def feedback_match():
    me = session["jugador_id"]
    match_id = request.form.get("match_id")
    val = request.form.get("feedback")
    if match_id and val in ("0","1"):
        ok = query("SELECT 1 FROM conexion_alfa_matches WHERE id=%s AND jugador_1_id=%s", (match_id, me))
        if ok:
            execute("UPDATE conexion_alfa_matches SET feedback=%s WHERE id=%s", (int(val), match_id))
            flash("¡Gracias por tu feedback!")
    return redirect(url_for("conexion_alfa_mi_match"))

@app.route("/conexion_alfa_subir_foto", methods=["GET", "POST"])
@login_required
def conexion_alfa_subir_foto():
    me = session["jugador_id"]
    m = query("SELECT jugador_2_id FROM conexion_alfa_matches WHERE jugador_1_id=%s LIMIT 1", (me,))
    if not m:
        flash("Aún no tienes match.")
        return redirect(url_for("conexion_alfa_mi_match"))
    if request.method == "POST":
        f = request.files.get("foto")
        if not f or not f.filename:
            flash("Sube una imagen válida.")
            return redirect(url_for("conexion_alfa_subir_foto"))
        os.makedirs(os.path.join(app.static_folder, "evidencias_alfa"), exist_ok=True)
        fname = f"foto_{me}_{int(time.time())}.jpg"
        f.save(os.path.join(app.static_folder, "evidencias_alfa", fname))
        execute("""
            UPDATE conexion_alfa_matches SET evidencia=%s WHERE jugador_1_id=%s
        """, (fname, me))
        flash("Foto subida.")
        return redirect(url_for("conexion_alfa_mi_match"))

    match = query("""
        SELECT j1.nombre AS nombre_1, j1.correo AS correo_1,
               j2.nombre AS nombre_2, j2.correo AS correo_2
        FROM jugadores j1
        JOIN conexion_alfa_matches m ON m.jugador_1_id=j1.id
        JOIN jugadores j2 ON j2.id=m.jugador_2_id
        WHERE j1.id=%s
        LIMIT 1
    """, (me,))
    return render_template("conexion_alfa_subir_foto.html", match=match[0])

@app.route("/conexion_alfa_subir_video", methods=["GET", "POST"])
@login_required
def conexion_alfa_subir_video():
    me = session["jugador_id"]
    m = query("SELECT jugador_2_id FROM conexion_alfa_matches WHERE jugador_1_id=%s LIMIT 1", (me,))
    if not m:
        flash("Aún no tienes match.")
        return redirect(url_for("conexion_alfa_mi_match"))
    if request.method == "POST":
        f = request.files.get("video")
        if not f or not f.filename:
            flash("Sube un video válido.")
            return redirect(url_for("conexion_alfa_subir_video"))
        os.makedirs(os.path.join(app.static_folder, "evidencias_alfa"), exist_ok=True)
        fname = f"video_{me}_{int(time.time())}.mp4"
        f.save(os.path.join(app.static_folder, "evidencias_alfa", fname))
        execute("""
            UPDATE conexion_alfa_matches SET evidencia=%s WHERE jugador_1_id=%s
        """, (fname, me))
        flash("Video subido.")
        return redirect(url_for("conexion_alfa_mi_match"))

    match = query("""
        SELECT j1.nombre AS nombre_1, j1.correo AS correo_1,
               j2.nombre AS nombre_2, j2.correo AS correo_2
        FROM jugadores j1
        JOIN conexion_alfa_matches m ON m.jugador_1_id=j1.id
        JOIN jugadores j2 ON j2.id=m.jugador_2_id
        WHERE j1.id=%s
        LIMIT 1
    """, (me,))
    return render_template("conexion_alfa_subir_video.html", match=match[0])

@app.route("/conexion_alfa_emparejamientos", methods=["GET"])
@admin_required
def conexion_alfa_emparejamientos():
    pares = query("""
        SELECT LEAST(jugador_1_id, jugador_2_id) AS a,
               GREATEST(jugador_1_id, jugador_2_id) AS b
        FROM conexion_alfa_matches
        GROUP BY 1,2
    """)
    matches = []
    for row in pares:
        a,b = row["a"], row["b"]
        m = query("""
            SELECT m1.id, m1.jugador_1_id, j1.nombre AS nombre_1,
                   m1.jugador_2_id, j2.nombre AS nombre_2,
                   m1.score, m1.razon_match, m1.evidencia, m1.feedback
            FROM conexion_alfa_matches m1
            JOIN jugadores j1 ON j1.id=m1.jugador_1_id
            JOIN jugadores j2 ON j2.id=m1.jugador_2_id
            WHERE m1.jugador_1_id=%s AND m1.jugador_2_id=%s
            LIMIT 1
        """, (a,b))
        if not m: continue
        matches.append(m[0])

    fb_vals = [m["feedback"] for m in matches if m["feedback"] is not None]
    if fb_vals:
        tasa = sum(1 for v in fb_vals if v==1)/len(fb_vals)
        accuracy = precision = recall = f1 = round(tasa, 3)
    else:
        accuracy = precision = recall = f1 = None

    return render_template("conexion_alfa_emparejamientos.html",
                           matches=matches, accuracy=accuracy, precision=precision, recall=recall, f1=f1)

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

# --- Fixers admin (one-click) ---
@app.route("/admin/fix_conexion_alfa_hard", methods=["POST"])
@admin_required
def admin_fix_conexion_alfa_hard():
    execute(r"""
    DO $$BEGIN
      -- Asegura tabla base
      CREATE TABLE IF NOT EXISTS conexion_alfa_respuestas (
        jugador_id INTEGER,
        r1 TEXT, r2 TEXT, r3 TEXT, r4 TEXT, r5 TEXT, r6 TEXT, r7 TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
      );

      -- jugador_id y backfill desde id si existía
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='conexion_alfa_respuestas' AND column_name='jugador_id'
      ) THEN
        ALTER TABLE conexion_alfa_respuestas ADD COLUMN jugador_id INTEGER;
      END IF;

      IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='conexion_alfa_respuestas' AND column_name='id'
      ) THEN
        UPDATE conexion_alfa_respuestas
           SET jugador_id = id
         WHERE jugador_id IS NULL;
      END IF;

      -- Relaja NOT NULL legados
      IF EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_name='conexion_alfa_respuestas' AND column_name='correo') THEN
        BEGIN
          ALTER TABLE conexion_alfa_respuestas ALTER COLUMN correo DROP NOT NULL;
        EXCEPTION WHEN others THEN PERFORM 1; END;
      END IF;

      IF EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_name='conexion_alfa_respuestas' AND column_name='nombre') THEN
        BEGIN
          ALTER TABLE conexion_alfa_respuestas ALTER COLUMN nombre DROP NOT NULL;
        EXCEPTION WHEN others THEN PERFORM 1; END;
      END IF;

      -- Quita PK vieja si no es en jugador_id
      PERFORM 1;
      -- Crear PK en jugador_id si no hay nulos
      IF NOT EXISTS (SELECT 1 FROM conexion_alfa_respuestas WHERE jugador_id IS NULL) THEN
        BEGIN
          ALTER TABLE conexion_alfa_respuestas ADD CONSTRAINT ca_res_pk PRIMARY KEY (jugador_id);
        EXCEPTION WHEN duplicate_object THEN PERFORM 1; END;
      END IF;
    END$$;
    """)
    execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_ca_respuestas_jugador_id
        ON conexion_alfa_respuestas(jugador_id)
    """)
    flash("Conexión Alfa saneada: jugador_id como clave y columnas legadas relajadas.")
    return redirect(url_for("admin_panel"))

@app.route("/admin/fix_adivina_scores", methods=["POST"])
@admin_required
def admin_fix_adivina_scores():
    execute(r"""
        WITH ranked AS (
          SELECT ctid, ROW_NUMBER() OVER (
            PARTITION BY jugador_id ORDER BY created_at DESC
          ) AS rn
          FROM adivina_scores
        )
        DELETE FROM adivina_scores a
        USING ranked r
        WHERE a.ctid = r.ctid AND r.rn > 1;
    """)
    execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_adivina_jugador ON adivina_scores(jugador_id)")
    execute(r"""
    DO $$BEGIN
      BEGIN
        ALTER TABLE adivina_scores ADD CONSTRAINT adivina_scores_pk PRIMARY KEY (jugador_id);
      EXCEPTION WHEN duplicate_object THEN PERFORM 1;
      WHEN others THEN PERFORM 1;
      END;
    END$$;
    """)
    flash("Adivina: limpiado duplicados y creado índice/PK en jugador_id.")
    return redirect(url_for("admin_panel"))

@app.route("/admin/debug_reto_foto")
@admin_required
def admin_debug_reto_foto():
    ch = get_active_photo_challenge()
    if not ch:
        return "<pre>No hay reto activo</pre>"
    rows = get_photo_entries(ch["id"])
    lines = []
    for e in rows:
        # Chequea ambas rutas
        p_static = os.path.join(app.static_folder, "fotos_ch", str(ch["id"]), e["filename"].replace("tmp:",""))
        p_tmp = os.path.join(TMP_UPLOAD_ROOT, str(ch["id"]), e["filename"].replace("tmp:",""))
        exists_static = os.path.exists(p_static)
        exists_tmp = os.path.exists(p_tmp)
        lines.append(f"{e['id']:>3} | j:{e['jugador_id']:<4} | {e['nombre']:<20} | {e['filename']} | static={exists_static} | tmp={exists_tmp}")
    return "<pre>" + "\n".join(lines or ["(sin filas)"]) + "</pre>"

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
