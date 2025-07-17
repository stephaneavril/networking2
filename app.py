import os, random, sqlite3, math, json
from datetime import datetime
from functools import wraps
from typing import List, Dict, Sequence, Mapping, Any

from flask import (
    Flask, render_template, request, jsonify, session,
    redirect, flash, url_for
)
from openai import OpenAI, OpenAIError
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# ───────────────────────── CONFIG ─────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("⚠️ Falta OPENAI_API_KEY en variables de entorno")
client = OpenAI()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "clave‑segura")
app.config.update(
    UPLOAD_FOLDER="evidencias",
    UPLOAD_FOLDER_GRUPAL="evidencias_reto_grupal",
    EMBED_MODEL="text-embedding-3-small",
)

# ───────── Admin quick‑access ─────────
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "SupermanNoEsGay")

def admin_ok(req):
    if session.get("is_admin"):
        return True
    token = req.args.get("token") or req.headers.get("X-Admin-Token")
    if token and token == ADMIN_TOKEN:
        session["is_admin"] = True
        return True
    return False

# ───────────────────────── DB (PostgreSQL) ─────────────────────────
import psycopg2, psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
from typing import Sequence, Any

DATABASE_URL = os.getenv("DATABASE_URL")  # ej. export DATABASE_URL="postgresql://…"
if not DATABASE_URL:
    raise RuntimeError("⚠️ Falta la variable DATABASE_URL")

_pool = SimpleConnectionPool(
    1, 10,
    dsn=DATABASE_URL,
    sslmode="require",
    cursor_factory=psycopg2.extras.RealDictCursor,
)

def _qmark_to_psycopg(sql: str) -> str:
    """Convierte placeholders «? » estilo-SQLite → «%s» para psycopg2."""
    # Ojo: no rompe los %s que ya estén escritos
    partes = sql.split("?")
    return "%s".join(partes) if len(partes) > 1 else sql

class DB:
    """Wrapper 100 % compatible con tu viejo `sqlite3.Connection`."""
    def __init__(self, conn):
        self._c = conn

    def execute(self, sql: str, params: Sequence[Any] = ()):
        sql = _qmark_to_psycopg(sql)
        cur = self._c.cursor()                        # RealDictCursor por default
        cur.execute(sql, params)
        return cur           # mantiene `.fetchone()` / `.fetchall()` vivos

    def commit(self):  self._c.commit()
    def cursor(self):  return self._c.cursor()       # para los bloques “with conn.cursor()”
    def close(self):   _pool.putconn(self._c)

def get_db_connection() -> DB:
    return DB(_pool.getconn())

# ─────────── Crear tablas automáticamente si no existen ────────────
SCHEMA_SQL = [
    """
    CREATE TABLE IF NOT EXISTS adivina_participantes (
        id                 SERIAL PRIMARY KEY,
        nombre             TEXT  NOT NULL,
        correo             TEXT  NOT NULL UNIQUE,
        objetivo_2025      TEXT,
        nivel_introversion INTEGER
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS adivina_resultados (
        id              SERIAL PRIMARY KEY,
        nombre_jugador  TEXT     NOT NULL,
        aciertos        INTEGER  NOT NULL,
        puntos_extra    INTEGER  NOT NULL,
        timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS conexion_alfa_respuestas (
        correo              TEXT PRIMARY KEY,
        nombre              TEXT NOT NULL,
        r2 TEXT,  r3 TEXT,  r4 TEXT,
        r6 TEXT,  r8 TEXT,  r9 TEXT,  r10 TEXT,
        r12_mascota TEXT,   r13_hijos TEXT,
        perfil_ia          TEXT,
        objetivo_2025      TEXT,
        nivel_introversion INTEGER DEFAULT 0
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS conexion_alfa_matches (
        id          SERIAL PRIMARY KEY,
        correo_1    TEXT NOT NULL,
        correo_2    TEXT NOT NULL,
        nombre_1    TEXT NOT NULL,
        nombre_2    TEXT NOT NULL,
        perfil_1    TEXT,
        perfil_2    TEXT,
        razon_match TEXT,
        evidencia   TEXT,
        feedback    SMALLINT,
        UNIQUE (correo_1, correo_2)
    );
    """,
    # TODO: añade aquí el resto de tablas (retos, reto_foto, evidencias, etc.)
]

def create_tables_if_needed() -> None:
    """Ejecuta cada sentencia CREATE TABLE IF NOT EXISTS una vez al arranque."""
    conn = get_db_connection()
    try:
        for sql in SCHEMA_SQL:
            conn.execute(sql)
        conn.commit()
        print("✅ Tablas verificadas/creadas")
    finally:
        conn.close()

# Ejecutamos la función al importar la app
create_tables_if_needed()


# ─────────────────────── MODELO IA ───────────────────────
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib, numpy as np
MODEL_PATH = Path(__file__).with_name("vectorizer_ia.pkl")
vectorizer_ia = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else TfidfVectorizer().fit(["placeholder"])

# ─────────────────────── UTILIDADES ──────────────────────

def generar_perfil_ia(nombre: str, *, dato_curioso="", pelicula="", deporte="", prenda="",
                       concierto="", pasion="", libro="", mascota="", hijos="") -> str:
    partes: List[str] = []
    add = lambda emoji, txt: partes.append(f"{emoji} {txt}") if txt else None
    add("🧠", f"{nombre} comparte un dato curioso: “{dato_curioso}”.")
    add("🎬", f"Su película favorita es “{pelicula}”.")
    add("🏀", f"Deporte favorito: “{deporte}”.")
    add("👕", f"No vive sin: “{prenda}”.")
    add("🎤", f"Mejor concierto: “{concierto}”.")
    add("🎶", f"Le apasiona: “{pasion}”.")
    add("📚", f"Libro/arte favorito: “{libro}”.")
    add("🐾", f"Mascota(s): {mascota}.")
    add("👨‍👩‍👧‍👦", f"Hijos: {hijos}.")
    partes.append("✨ ¿Por qué conocerle? Su mezcla de gustos garantiza charlas memorables.")
    return " ".join(partes)

# ---------- Embeddings (batch) ------------------------------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    try:
        resp = client.embeddings.create(
            model=app.config["EMBED_MODEL"], input=[t[:4096] for t in texts]
        )
        return [d.embedding for d in resp.data]
    except Exception as e:
        print("❌ openai error:", e)
        dim = 1536
        return [[0.0] * dim for _ in texts]

# ---------- Emparejamiento Greedy ------------------------------------------

import numpy as np

def build_similarity(vecs: List[List[float]]) -> List[List[float]]:
    M = np.array(vecs, dtype=float)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return (M @ M.T).tolist()


def hacer_matches(datos: Sequence[Mapping]) -> List[Dict]:
    textos = [
        " ".join([
            d.get("r3", ""), d.get("r4", ""), d.get("r6", ""), d.get("r8", ""),
            d.get("r9", ""), d.get("r2", ""), d.get("r10", ""),
            d.get("r12_mascota", ""), d.get("r13_hijos", ""),
        ]).lower()
        for d in datos
    ]
    embeddings = embed_texts(textos)
    S = build_similarity(embeddings)

    n = len(datos)
    usados: set[int] = set()
    pares: list[Dict] = []
    for i in range(n):
        if i in usados:
            continue
        mejor_j, mejor_sim = None, -1.0
        for j in range(n):
            if j == i or j in usados:
                continue
            if S[i][j] > mejor_sim:
                mejor_j, mejor_sim = j, S[i][j]
        if mejor_j is not None:
            usados.update({i, mejor_j})
            pares.append({
                "correo_1": datos[i]["correo"],
                "correo_2": datos[mejor_j]["correo"],
                "nombre_1": datos[i]["nombre"],
                "nombre_2": datos[mejor_j]["nombre"],
                "perfil_1": datos[i]["perfil_ia"],
                "perfil_2": datos[mejor_j]["perfil_ia"],
                "score": round(float(mejor_sim), 2),
            })
    return pares

# ──────────────────── FLASK HELPERS ─────────────────────

@app.before_request
def make_session_permanent():
    session.permanent = True


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "jugador" not in session:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped

# ──────────── GPT helper ─────────────

def explicar_match_gpt(perfil1: str, perfil2: str, score: float) -> str | None:
    prompt = (
        "Eres un coach de networking.\n"
        f"Persona A:\n{perfil1}\n\n"
        f"Persona B:\n{perfil2}\n\n"
        f"Similaridad global: {score:.2f}\n\n"
        "Genera un párrafo de 3‑4 líneas explicando por qué harían una buena conexión "
        "y sugiere un tema concreto para iniciar la conversación."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except OpenAIError as e:
        print("⚠️  GPT error:", e)
        return None

# ───────────────────────── LOGIN ─────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    next_url = request.args.get("next", "/")
    if request.method == "POST":
        jugador = request.form.get("jugador", "").strip()
        correo = request.form.get("correo", "").strip()
        if not jugador or not correo:
            flash("⚠️ Debes indicar nombre y correo")
            return render_template("login.html", next=next_url)
        session.update({"jugador": jugador, "correo": correo})
        flash(f"¡Bienvenido, {jugador}!")
        conn = get_db_connection()
        ya = fetchone(conn, "SELECT 1 FROM conexion_alfa_respuestas WHERE correo = %s", (correo,))
        _pool.putconn(conn)
        if not ya:
            return redirect(url_for("conocete_mejor"))
        return redirect(next_url)
    return render_template("login.html", next=next_url)

# ───────────────────────── HOME ───────────────────────────

@app.route("/")
def index():
    if "jugador" not in session:
        return redirect("/login")
    conn = get_db_connection()
    retos = fetchone(conn, "SELECT json_agg(retos) AS r FROM (SELECT * FROM retos WHERE activo=1) retos")
    _pool.putconn(conn)

    # La DB de QR sigue en SQLite local
    qr_conn = sqlite3.connect("scan_points.db"); qr_conn.row_factory = sqlite3.Row
    ranking_qr = qr_conn.execute("""
        SELECT nombre, SUM(puntos) AS total
        FROM registros GROUP BY nombre ORDER BY total DESC
    """).fetchall(); qr_conn.close()
    return render_template("index.html", retos=retos["r"] if retos else [], ranking_qr=ranking_qr, modo_foto_equipo=False)

# --------------------------- CONÓCETE MEJOR ---------------------------

@app.route("/conocete_mejor", methods=["GET", "POST"])
@login_required
def conocete_mejor():
    if request.method == "GET":
        conn = get_db_connection()
        ya = fetchone(conn, "SELECT 1 FROM conexion_alfa_respuestas WHERE correo=%s", (session["correo"],))
        _pool.putconn(conn)
        return render_template("preguntas_post_login.html", ya_respondio=bool(ya))

    f = request.form.get
    nombre, correo = session["jugador"], session["correo"]
    data = {
        k: f(k, "").strip() for k in ("r2","r3","r4","r6","r8","r9","r10")
    }
    data["r12_mascota"] = f("r12", "").strip()
    data["r13_hijos"] = f("r13", "").strip()

    perfil = generar_perfil_ia(
        nombre,
        dato_curioso=data["r3"], pelicula=data["r4"], deporte=data["r6"],
        prenda=data["r8"], concierto=data["r9"], pasion=data["r2"],
        libro=data["r10"], mascota=data["r12_mascota"], hijos=data["r13_hijos"],
    )

    columnas = (
        "correo, nombre, r2, r3, r4, r6, r8, r9, r10, r12_mascota, r13_hijos, perfil_ia"
    )
    placeholders = ", ".join(["%s"] * 12)
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO conexion_alfa_respuestas ({columnas})
            VALUES ({placeholders})
            ON CONFLICT (correo) DO UPDATE SET
                nombre         = EXCLUDED.nombre,
                r2             = EXCLUDED.r2,
                r3             = EXCLUDED.r3,
                r4             = EXCLUDED.r4,
                r6             = EXCLUDED.r6,
                r8             = EXCLUDED.r8,
                r9             = EXCLUDED.r9,
                r10            = EXCLUDED.r10,
                r12_mascota    = EXCLUDED.r12_mascota,
                r13_hijos      = EXCLUDED.r13_hijos,
                perfil_ia      = EXCLUDED.perfil_ia
            """,
            (
                correo, nombre,
                data["r2"], data["r3"], data["r4"], data["r6"], data["r8"], data["r9"], data["r10"],
                data["r12_mascota"], data["r13_hijos"], perfil,
            ),
        )
    conn.commit(); _pool.putconn(conn)
    flash("✅ Respuestas guardadas. ¡Gracias!")
    return redirect(url_for("index"))

# --------------------- GENERAR MATCHES (ADMIN) ----------------------

@app.route("/generar_matches_conexion_alfa", methods=["POST"])
def generar_matches_conexion_alfa():
    if "jugador" not in session:
        return redirect("/")
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM conexion_alfa_respuestas")
        datos = cur.fetchall()
    if len(datos) < 2:
        flash("❌ Mínimo 2 participantes para generar matches."); _pool.putconn(conn); return redirect("/admin_panel")
    pares = hacer_matches(datos)

    with conn.cursor() as cur:
        cur.execute("SELECT correo_1, correo_2 FROM conexion_alfa_matches")
        ya = {tuple(sorted((r["correo_1"], r["correo_2"]))) for r in cur.fetchall()}

        nuevos = 0
        for p in pares:
            key = tuple(sorted((p["correo_1"], p["correo_2"])));
            if key in ya: continue
            razon = explicar_match_gpt(p["perfil_1"], p["perfil_2"], p["score"]) or f"🤖 Match IA · similitud {p['score']*100:.0f}%"
            cur.execute(
                """
                INSERT INTO conexion_alfa_matches
                      (correo_1, correo_2, nombre_1, nombre_2, perfil_1, perfil_2, razon_match)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    p["correo_1"], p["correo_2"], p["nombre_1"], p["nombre_2"],
                    p["perfil_1"], p["perfil_2"], razon,
                ),
            ); nuevos += 1
    conn.commit(); _pool.putconn(conn)
    flash(f"✅ {nuevos} matches generados con éxito (modelo OpenAI).")
    return redirect("/admin_panel")

# -------------------- RETO ADIVINA – guardar resultados --------------------
@app.route('/adivina_finalizado', methods=['POST'])
def adivina_finalizado():
    if 'jugador' not in session:
        return jsonify({"error": "No autenticado"}), 401

    data      = request.get_json(force=True) or {}
    jugador   = session['jugador']
    puntaje   = int(data.get("puntaje", 0))
    aciertos  = int(data.get("aciertos", 0))

    conn = get_db_connection()

    # 1️⃣  Evitar doble registro
    ya = conn.execute(
        "SELECT 1 FROM adivina_resultados WHERE nombre_jugador = %s",
        (jugador,)
    ).fetchone()
    if ya:
        conn.close()
        return jsonify({"error": "Ya has completado el reto"}), 400

    # 2️⃣  Insertar resultado
    conn.execute(
        """
        INSERT INTO adivina_resultados (nombre_jugador, aciertos, puntos_extra)
        VALUES (%s, %s, %s)
        """,
        (jugador, aciertos, puntaje)
    )
    conn.commit()

    # 3️⃣  Buscar nombre de match *antes* de cerrar la conexión
    match_name = obtener_match_para(session["correo"], conn)
    conn.close()

    return jsonify({
        "message": (
            f"🎉 ¡Reto completado! {jugador} acertó {aciertos} nombre(s) "
            f"y obtuvo {puntaje} pts."
        ),
        "redirect": "/ranking_adivina"
    })

# --- FOTO RETO EQUIPO  (deja sólo ESTA versión) -------------------
@app.route('/foto_reto/<int:reto_no>', methods=['GET', 'POST'])
@login_required
def foto_reto_equipo(reto_no):
    nombre  = session['jugador']
    correo  = session['correo']
    equipo  = session['equipo']

    mensajes = {
        1: "📸",
        2: "📸",
        3: "📸 "
    }
    mensaje = mensajes.get(reto_no, "Sube tu foto")

    conn = get_db_connection()
    ya_existe = conn.execute(
        "SELECT 1 FROM reto_equipo_foto WHERE equipo=? AND reto_no=?",
        (equipo, reto_no)
    ).fetchone()

    if request.method == 'POST' and not ya_existe:
        archivo = request.files.get('foto')
        if not archivo:
            flash("❌ Falta seleccionar la imagen"); return redirect(request.url)

        filename = secure_filename(
            f"{datetime.now():%Y%m%d%H%M%S}_{archivo.filename}")   # <- sin \
        carpeta = os.path.join('static', f'fotos_reto_{reto_no}')
        os.makedirs(carpeta, exist_ok=True)
        archivo.save(os.path.join(carpeta, filename))

        conn.execute("""
            INSERT INTO reto_equipo_foto
            (nombre_participante, correo, equipo, archivo, reto_no)
            VALUES (?,?,?,?,?)""",
            (nombre, correo, equipo, filename, reto_no)
        )
        conn.commit()
        flash("✅ Foto recibida. ¡Gracias!")
        return redirect(url_for('index'))

    conn.close()
    return render_template('foto_reto_equipo.html',
                           mensaje=mensaje, equipo=equipo,
                           reto_no=reto_no, ya_existe=bool(ya_existe))

# ------------------- RANKING ADIVINA --------------------
@app.route("/ranking_adivina")
@login_required
def ranking_adivina():
    try:
        conn = get_db_connection()

        resultados = conn.execute("""
            SELECT nombre_jugador,
                   aciertos,
                   puntos_extra,
                   (aciertos + puntos_extra) AS total,
                   timestamp
            FROM adivina_resultados
            ORDER BY total DESC, timestamp ASC
        """).fetchall()

        # Resultado propio (por si quieres sombrearlo en la tabla)
        mi_resultado = conn.execute(
            "SELECT * FROM adivina_resultados WHERE nombre_jugador = ?",
            (session["jugador"],)
        ).fetchone()

        conn.close()

        # Siempre devolvemos plantilla, incluso si la lista está vacía
        return render_template(
            "ranking_adivina.html",
            resultados=resultados,
            mi_resultado=mi_resultado
        )

    except Exception as e:
        # Log para depuración en Render
        print("❌ ERROR ranking_adivina:", e)
        if "conn" in locals():
            conn.close()
        flash("❌ No se pudo mostrar el ranking. Revisa los logs.")
        # Redirigimos a home para evitar error 500
        return redirect(url_for("index"))

@app.route('/reset_adivina_quien', methods=['POST'])
def reset_adivina_quien():
    conn = get_db_connection()
    conn.execute("DELETE FROM adivina_resultados")
    conn.commit()
    conn.close()
    flash("✅ Ranking de Adivina Quién reiniciado correctamente.")
    return redirect('/admin_panel')

@app.route('/reset_adivina_participantes', methods=['POST'])
def reset_adivina_participantes():
    conn = get_db_connection()
    conn.execute("DELETE FROM adivina_participantes")
    conn.commit()
    conn.close()
    flash("✅ Participantes de Adivina Quién reiniciados correctamente.")
    return redirect('/admin_panel')

@app.route('/generar_contenido_adivina', methods=['POST'])
def generar_contenido_adivina():
    conn = get_db_connection()
    try:
        # 1. Lee a los participantes que llenaron el cuestionario
        respuestas = conn.execute('SELECT * FROM conexion_alfa_respuestas').fetchall()

        if not respuestas:
            flash("❌ Aún no hay participantes que hayan respondido el cuestionario.")
            return redirect('/admin_panel')

        # 2. Borra los datos anteriores para no duplicar
        conn.execute("DELETE FROM adivina_participantes")

        # 3. Inserta los participantes en la tabla del juego
        for r in respuestas:
            conn.execute("""
                INSERT INTO adivina_participantes (
                    nombre_completo, pasion, dato_curioso, pelicula_favorita,
                    deporte_favorito, prenda_imprescindible, mejor_concierto, 
                    mejor_libro, objetivo_2025
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r['nombre'], r['r2'], r['r3'], r['r4'], r['r6'], 
                r['r8'], r['r9'], r['r10'], r['objetivo_2025']
            ))

        conn.commit()

        num_participantes = len(respuestas)
        flash(f"✅ ¡Éxito! Se cargaron {num_participantes} participantes al juego.")

    except Exception as e:
        flash(f"❌ Ocurrió un error al generar el contenido: {e}")
    finally:
        conn.close()

    return redirect('/admin_panel')

@app.route('/respuestas_curiosas')
def respuestas_curiosas():
    conn = get_db_connection()
    respuestas = conn.execute('SELECT * FROM adivina_participantes').fetchall()
    conn.close()

    destacados = []
    for r in respuestas:
        frases = [
            f"🎯 Superpoder: {r['superpoder']}",
            f"🎶 Pasión: {r['pasion']}",
            f"🧠 Dato curioso: {r['dato_curioso']}",
            f"🎬 Película favorita: {r['pelicula_favorita']}",
        f"🎤 Concierto: {r['mejor_concierto']}",
    f"📖 Libro favorito: {r['mejor_libro']}",
    f"🏀 Deporte favorito: {r['deporte_favorito']}",
    f"🎯 Objetivo 2025: {r['objetivo_2025']}"
]
        seleccionadas = random.sample(frases, 3)
        destacados.append({
            "nombre": r["nombre_completo"],
            "frases": seleccionadas
        })

    return render_template("respuestas_curiosas.html", destacados=destacados)

# ── NUEVO helper (lo pones cerca de otras utilidades) ────────────
def obtener_match_para(correo_jugador, conn):
    """
    Devuelve el nombre del match de la tabla conexion_alfa_matches.
    """
    row = conn.execute(
        """
        SELECT CASE
                 WHEN correo_1 = %s THEN nombre_2
                 ELSE nombre_1
               END AS match_name
        FROM   conexion_alfa_matches
        WHERE  correo_1 = %s OR correo_2 = %s
        ORDER  BY id ASC
        LIMIT 1
        """,
        (correo_jugador, correo_jugador, correo_jugador)
    ).fetchone()
    return row["match_name"] if row else None

# -------------------- SUBIR EVIDENCIA INDIVIDUAL --------------------
@app.route('/subir_evidencia', methods=['POST'])
def subir_evidencia():
    if 'jugador' not in session:
        return redirect('/')
    nombre = session['jugador']
    reto_id = request.form.get('reto_id')
    archivo = request.files.get('archivo')
    if not archivo or not reto_id:
        return "❌ Faltan datos", 400
    nombre_archivo = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{archivo.filename}"
    ruta_archivo = os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    archivo.save(ruta_archivo)
    conn = get_db_connection()
    conn.execute("INSERT INTO evidencias (reto_id, nombre_participante, archivo) VALUES (?, ?, ?)",
                 (reto_id, nombre, nombre_archivo))
    conn.commit()
    conn.close()
    return "✅ Evidencia enviada con éxito"

# -------------------- RETO GRUPAL --------------------
@app.route('/reto_grupal')
def reto_grupal():
    if 'jugador' not in session:
        return redirect('/')
    conn = get_db_connection()
    reto = conn.execute("SELECT nombre FROM retos_grupales ORDER BY RANDOM() LIMIT 1").fetchone()
    conn.close()
    return render_template("reto_grupal.html", reto=reto['nombre'])

@app.route('/guardar_reto_grupal', methods=['POST'])
def guardar_reto_grupal():
    reto = request.form.get('reto')
    nombres = request.form.get('nombres')
    if not reto or not nombres:
        return "❌ Faltan datos", 400
    conn = get_db_connection()
    conn.execute("INSERT INTO participaciones_grupales (reto, nombres_participantes) VALUES (?, ?)", (reto, nombres))
    conn.commit()
    conn.close()
    flash("✅ ¡Gracias! Tu participación fue registrada.")
    return redirect('/')

# --------------------  ADMIN PANEL  --------------------
@app.route('/admin_panel', methods=['GET', 'POST'])
def admin_panel():
    if not (session.get("jugador") or admin_ok(request)):
        return redirect(url_for('login', next=request.path))


    conn = get_db_connection()

    # ── 1. Procesar botones ──────────────────────────────
    if request.method == 'POST':
        if 'reto_id' in request.form and 'activo' in request.form:        # ON / OFF
            conn.execute("UPDATE retos SET activo=? WHERE id=?",
                         (int(request.form['activo']), int(request.form['reto_id'])))
            conn.commit()
            flash("✅ Estado de reto actualizado.")
        elif 'activar_solo' in request.form:                              # 🔁 Solo este
            objetivo = int(request.form['activar_solo'])
            conn.execute("UPDATE retos SET activo=0")
            conn.execute("UPDATE retos SET activo=1 WHERE id=?", (objetivo,))
            conn.commit()
            flash("✅ Solo ese reto quedó activo.")

    # ── 2. Datos para la plantilla ───────────────────────
    retos      = conn.execute("SELECT * FROM retos").fetchall()
    resultados = conn.execute("""
        SELECT * FROM adivina_resultados
        ORDER BY puntos_extra DESC, timestamp ASC
    """).fetchall()
    matches    = conn.execute("SELECT * FROM conexion_alfa_matches").fetchall()

    # agrupar fotos: equipo → {reto_no: archivo}
    filas = conn.execute("""
        SELECT equipo, reto_no, archivo
        FROM reto_equipo_foto
        ORDER BY equipo, reto_no
    """).fetchall()
    equipos = {}
    for f in filas:
        equipos.setdefault(f["equipo"], {})[f["reto_no"]] = f["archivo"]

    conn.close()
    return render_template("admin_panel.html",
                           retos=retos,
                           resultados=resultados,
                           matches_conexion=matches,
                           equipos=equipos)

# -------------------- RETOS FOTO Y MI6 --------------------

def get_reto_id(nombre_reto):
    conn = get_db_connection()
    resultado = conn.execute("SELECT id FROM retos WHERE nombre = ?", (nombre_reto,)).fetchone()
    conn.close()
    return resultado["id"] if resultado else None

@app.route('/reto_foto', methods=['GET', 'POST'])
@app.route('/reto_mi6_v1', methods=['GET', 'POST'])
@app.route('/reto_mi6_v2', methods=['GET', 'POST'])
@app.route('/reto_mi6_v3', methods=['GET', 'POST'])
def reto_foto():
    if 'jugador' not in session:
        return redirect('/')

    ruta = request.path.strip("/")

    # Definir información para cada reto
    config = {
        "reto_foto": {
            "nombre_reto": "Reto Foto",
            "mensaje": "Sube una foto original que represente tu creatividad. Esta será votada por los demás participantes."
        },
        "reto_mi6_v1": {
            "nombre_reto": "MI6 v1",
            "titulo_visible": "Integridad FARMAPIEL",
            "mensaje": "📸 Toma una foto que represente cómo haces lo correcto incluso cuando nadie está mirando. Una imagen de integridad, valentía o esfuerzo extra."
        },
        "reto_mi6_v2": {
            "nombre_reto": "MI6 v2",
            "titulo_visible": "Transparencia FARMAPIEL",
            "mensaje": "Sube una foto que muestre apertura, honestidad o confianza. La transparencia se refleja cuando actuamos con claridad y coherencia ante los demás."
        },
        "reto_mi6_v3": {
            "nombre_reto": "MI6 v3",
            "titulo_visible": "Calidad FARMAPIEL",
            "mensaje": " Comparte una foto que represente excelencia, atención al detalle o mejora continua. La calidad se demuestra en cada acción bien hecha."
        }
    }

    datos_reto = config.get(ruta)
    if not datos_reto:
        return "❌ Ruta no válida", 404

    reto_id = get_reto_id(datos_reto["nombre_reto"])
    if reto_id is None:
        return "❌ El reto no existe en la base de datos", 500

    conn = get_db_connection()
    correo = session['correo']
    ya_existe = conn.execute(
        "SELECT * FROM reto_foto WHERE correo = ? AND reto_id = ?",
        (correo, reto_id)
    ).fetchone()

    if request.method == 'POST':
        if ya_existe:
            conn.close()
            return "❌ Ya has subido una foto para este reto."

        archivo = request.files.get('foto')
        if not archivo:
            return "❌ No se proporcionó ninguna imagen."

        nombre = session['jugador']
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{archivo.filename}"
        path = os.path.join('static/fotos_reto_foto', filename)
        os.makedirs('static/fotos_reto_foto', exist_ok=True)
        archivo.save(path)

        conn.execute(
            "INSERT INTO reto_foto (correo, nombre, archivo, reto_id) VALUES (?, ?, ?, ?)",
            (correo, nombre, filename, reto_id)
        )
        conn.commit()
        conn.close()
        flash("✅ Foto subida con éxito. ¡Gracias por participar!")
        return redirect('/')

    conn.close()
    return render_template(
    "reto_foto.html",
    ya_existe=ya_existe,
    mensaje=datos_reto["mensaje"],
    reto_nombre=datos_reto["titulo_visible"] or datos_reto["nombre_reto"]
)
from psycopg2 import IntegrityError

@app.route('/ver_fotos_reto_foto', methods=['GET', 'POST'])
def ver_fotos_reto_foto():
    if 'correo' not in session:
        return redirect('/')

    correo = session['correo']
    conn = get_db_connection()

    # Obtener el reto activo del tipo 'individual' que se llame MI6 o Reto Foto
    reto = conn.execute('''
        SELECT * FROM retos
        WHERE tipo = 'individual' AND activo = 1
        AND (nombre = 'Reto Foto' OR nombre LIKE 'MI6%')
        ORDER BY id ASC
        LIMIT 1
    ''').fetchone()

    if not reto:
        conn.close()
        return "❌ No hay ningún reto de foto activo en este momento."

    reto_id = reto["id"]
    reto_nombre = reto["nombre"]

    # Verificar si el usuario ya votó en este reto
    fotos_ids = [row["id"] for row in conn.execute("SELECT * FROM reto_foto WHERE reto_id = ?", (reto_id,)).fetchall()]
    votos_previos = conn.execute(
        "SELECT COUNT(*) FROM votos_reto_foto WHERE correo_votante = ? AND id_foto IN (%s)" % ",".join("?"*len(fotos_ids)),
        [correo] + fotos_ids
    ).fetchone()[0] if fotos_ids else 0

    # Procesar votos
    if request.method == 'POST' and votos_previos == 0:
        total_puntos = sum(int(v) for v in request.form.values() if v.isdigit())
        if total_puntos > 3:
            conn.close()
            return "❌ Solo puedes asignar hasta 3 puntos en total.", 400

        for key, val in request.form.items():
            if key.startswith("foto_") and val:
                id_foto = int(key.split("_")[1])
                puntos = int(val)
                try:
                    conn.execute(
                        "INSERT INTO votos_reto_foto (correo_votante, id_foto, puntos) VALUES (?, ?, ?)",
                        (correo, id_foto, puntos)
                    )
                except psycopg2.IntegrityError:
                    conn.rollback()      # opcional pero recomendable
                    continue
        conn.commit()
        flash("✅ ¡Tus votos han sido registrados!")
        return redirect('/ver_fotos_reto_foto')

    fotos = conn.execute("SELECT * FROM reto_foto WHERE reto_id = ?", (reto_id,)).fetchall()
    votos = conn.execute(
        "SELECT * FROM votos_reto_foto WHERE correo_votante = ?",
        (correo,)
    ).fetchall()
    votos_dict = {v['id_foto']: v['puntos'] for v in votos}
    conn.close()

    return render_template(
        "ver_fotos_reto_foto.html",
        fotos=fotos,
        votos=votos_dict,
        ya_voto=(votos_previos > 0),
        reto_nombre=reto_nombre
    )
@app.route('/ver_fotos_mi6_v1', methods=['GET', 'POST'])
@app.route('/ver_fotos_mi6_v2', methods=['GET', 'POST'])
@app.route('/ver_fotos_mi6_v3', methods=['GET', 'POST'])
def ver_fotos_mi6():
    if 'correo' not in session:
        return redirect('/')

    ruta = request.path.strip("/")
    nombre_reto = {
        "ver_fotos_mi6_v1": "MI6 v1",
        "ver_fotos_mi6_v2": "MI6 v2",
        "ver_fotos_mi6_v3": "MI6 v3"
    }.get(ruta)

    if not nombre_reto:
        return "❌ Ruta inválida", 404

    conn = get_db_connection()
    reto = conn.execute("SELECT * FROM retos WHERE nombre = ?", (nombre_reto,)).fetchone()

    if not reto:
        conn.close()
        return "❌ El reto no existe en la base de datos", 500

    reto_id = reto["id"]
    correo = session["correo"]
    fotos = conn.execute("SELECT * FROM reto_foto WHERE reto_id = ?", (reto_id,)).fetchall()

    # Revisión de votos
    fotos_ids = [f["id"] for f in fotos]
    votos = conn.execute(
        "SELECT * FROM votos_reto_foto WHERE correo_votante = ? AND id_foto IN (%s)" % ",".join("?" * len(fotos_ids)),
        [correo] + fotos_ids if fotos_ids else [correo]
    ).fetchall() if fotos_ids else []

    votos_previos = len(votos)
    votos_dict = {v['id_foto']: v['puntos'] for v in votos}
  # ✅ ✅ ✅ PEGA AQUÍ ESTE BLOQUE
    if request.method == 'POST' and votos_previos == 0:
        total_puntos = sum(int(v) for v in request.form.values() if v.isdigit())
        if total_puntos > 3:
            conn.close()
            return "❌ Solo puedes asignar hasta 3 puntos en total.", 400

        for key, val in request.form.items():
            if key.startswith("foto_") and val:
                id_foto = int(key.split("_")[1])
                puntos = int(val)
                try:
                    conn.execute(
                        "INSERT INTO votos_reto_foto (correo_votante, id_foto, puntos) VALUES (?, ?, ?)",
                        (correo, id_foto, puntos)
                    )
                except sqlite3.IntegrityError:
                    continue
        conn.commit()
        flash("✅ ¡Tus votos han sido registrados!")
        return redirect(request.path)

    conn.close()

    return render_template(
        "ver_fotos_reto_foto.html",
        fotos=fotos,
        votos=votos_dict,
        ya_voto=(votos_previos > 0),
        reto_nombre=nombre_reto
    )

@app.route('/votar_fotos', methods=['POST'])
def votar_fotos():
    if 'correo' not in session:
        return redirect('/')
    correo_votante = session['correo']

    total_puntos = sum(int(v) for v in request.form.values() if v.isdigit())
    if total_puntos != 3:
        return "❌ Debes asignar exactamente 3 puntos", 400

    conn = get_db_connection()
    for id_foto, puntos in request.form.items():
        if puntos and puntos.isdigit():
            conn.execute(
                """
                INSERT INTO votos_reto_foto (correo_votante, id_foto, puntos)
                VALUES (%s, %s, %s)
                ON CONFLICT (correo_votante, id_foto)
                DO UPDATE SET puntos = EXCLUDED.puntos
                """,
                (correo_votante, int(id_foto), int(puntos))
            )
    conn.commit()
    conn.close()
    return redirect('/')

@app.route('/ranking_fotos')
def ranking_fotos():
    if 'jugador' not in session:
        return redirect('/')

    conn = get_db_connection()

    nombre_reto = request.args.get("reto")

    if nombre_reto:
        reto = conn.execute("SELECT * FROM retos WHERE nombre = ?", (nombre_reto,)).fetchone()
    else:
        reto = conn.execute('''
            SELECT * FROM retos
            WHERE tipo = 'individual' AND activo = 1
            AND (nombre = 'Reto Foto' OR nombre LIKE 'MI6%')
            ORDER BY id ASC
            LIMIT 1
        ''').fetchone()

    if not reto:
        conn.close()
        return "❌ No hay reto de foto activo en este momento."

    reto_id = reto['id']
    reto_nombre = reto['nombre']

    ranking = conn.execute('''
        SELECT nombre, archivo, SUM(puntos) as total_puntos
        FROM votos_reto_foto
        JOIN reto_foto ON votos_reto_foto.id_foto = reto_foto.id
        WHERE reto_foto.reto_id = ?
        GROUP BY id_foto
        ORDER BY total_puntos DESC
    ''', (reto_id,)).fetchall()

    conn.close()
    return render_template("ranking_fotos.html", ranking=ranking, reto_nombre=reto_nombre)

@app.route('/reset_reto_foto', methods=['POST'])
def reset_reto_foto():
    conn = get_db_connection()

    # Obtener IDs de todos los retos de foto (Reto Foto + MI6)
    reto_ids = conn.execute(
        "SELECT id FROM retos WHERE nombre = 'Reto Foto' OR nombre LIKE 'MI6%'"
    ).fetchall()
    ids = [str(r["id"]) for r in reto_ids]

    if ids:
        # Borrar votos solo de esas fotos
        foto_ids = conn.execute(
            f"SELECT id FROM reto_foto WHERE reto_id IN ({','.join(['?'] * len(ids))})",
            ids
        ).fetchall()
        foto_ids_int = [str(f["id"]) for f in foto_ids]

        if foto_ids_int:
            conn.execute(
                f"DELETE FROM votos_reto_foto WHERE id_foto IN ({','.join(['?'] * len(foto_ids_int))})",
                foto_ids_int
            )

        # Borrar fotos
        conn.execute(
            f"DELETE FROM reto_foto WHERE reto_id IN ({','.join(['?'] * len(ids))})",
            ids
        )
        conn.commit()

    conn.close()

    # Eliminar archivos del folder
    carpeta = 'static/fotos_reto_foto'
    if os.path.exists(carpeta):
        for archivo in os.listdir(carpeta):
            ruta = os.path.join(carpeta, archivo)
            if os.path.isfile(ruta):
                os.remove(ruta)

    flash("✅ Reto Foto y fotos MI6 reiniciadas correctamente.")
    return redirect('/admin_panel')

# -------------------- CONEXION ALFA --------------------

@app.route('/conexion_alfa')
def conexion_alfa():
    if 'correo' not in session:
        return redirect('/')
    
    return redirect('/conexion_alfa_match')

@app.route('/conexion_alfa_mi_perfil')
def conexion_alfa_mi_perfil():
    if 'correo' not in session:
        return redirect('/')

    conn = get_db_connection()
    perfil = conn.execute("SELECT * FROM conexion_alfa_respuestas WHERE correo = ?", (session['correo'],)).fetchone()
    conn.close()
    return render_template("conexion_alfa_perfil.html", perfil=perfil)

# ----------------- CONEXIÓN ALFA – Matches -----------------
@app.route('/conexion_alfa_matches', methods=['GET'])
def conexion_alfa_matches():
    if 'correo' not in session:
        return redirect('/')

    correo_usuario = session['correo']
    conn = get_db_connection()

    # 1. Leer todas las respuestas de los participantes
    datos = conn.execute("SELECT * FROM conexion_alfa_respuestas").fetchall()

    textos, correos, nombres, perfiles = [], [], [], []
    for row in datos:
        # Tomamos todas las respuestas r1…r13 (si existen) y las unimos
        respuestas = [row.get(f"r{i}", "") or "" for i in range(1, 14)]
        textos.append(" ".join(respuestas))
        correos.append(row["correo"])
        nombres.append(row["nombre"])
        perfiles.append(row["perfil_ia"])

    # 2. Vectorizar solo **una vez**
    vectores = vectorizer_ia.transform(textos).toarray()
    sim      = cosine_similarity(vectores)

    # 3. Pares ya guardados → evitar duplicados
    ya_guardados = conn.execute(
        "SELECT correo_1, correo_2 FROM conexion_alfa_matches"
    ).fetchall()
    ya_guardados_set = {
        tuple(sorted((r["correo_1"], r["correo_2"]))) for r in ya_guardados
    }

    # 4. Bucle “greedy”: cada persona con su par más similar no usado
    usados = set()
    for i in range(len(correos)):
        if i in usados:
            continue

        mejor_j, mejor_sim = None, -1
        for j in range(i + 1, len(correos)):
            if j in usados:
                continue
            if sim[i, j] > mejor_sim:
                mejor_sim = sim[i, j]
                mejor_j   = j

        if mejor_j is None:
            continue

        correo1, correo2 = correos[i], correos[mejor_j]
        pareja           = tuple(sorted((correo1, correo2)))
        if pareja in ya_guardados_set:
            continue      # ya existe en la base

        # Explicación sencilla (puedes cambiar por `explicar_match_gpt` si lo prefieres)
        razon = f"🤖 Compatibilidad IA {mejor_sim*100:.0f}%"

        conn.execute(
            """
            INSERT INTO conexion_alfa_matches
                  (correo_1, correo_2, nombre_1, nombre_2, perfil_1, perfil_2, razon_match)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                correo1, correo2,
                nombres[i], nombres[mejor_j],
                perfiles[i], perfiles[mejor_j],
                razon,
            ),
        )
        usados.update({i, mejor_j})

    conn.commit()

    # 5. Matches propios para mostrar
    matches = conn.execute(
        """
        SELECT * FROM conexion_alfa_matches
        WHERE correo_1 = ? OR correo_2 = ?
        """,
        (correo_usuario, correo_usuario),
    ).fetchall()

    # 6. Métricas de feedback
    feedbacks = conn.execute(
        "SELECT feedback FROM conexion_alfa_matches WHERE feedback IS NOT NULL"
    ).fetchall()
    total      = len(feedbacks)
    positivos  = sum(f["feedback"] == 1 for f in feedbacks)
    negativos  = sum(f["feedback"] == 0 for f in feedbacks)

    if total:
        accuracy  = round(positivos / total, 2)
        precision = round(positivos / (positivos + negativos), 2) if (positivos + negativos) else 0
        recall    = round(positivos / total, 2)
        f1        = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) else 0
    else:
        accuracy = precision = recall = f1 = None

    conn.close()
    return render_template(
        "conexion_alfa_matches.html",
        matches=matches,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )

@app.route('/confirmar_match', methods=['POST'])
def confirmar_match():
    match_id = request.form.get('match_id')
    respuesta = int(request.form.get('respuesta'))
    conn = get_db_connection()
    conn.execute("UPDATE conexion_alfa_matches SET feedback = ? WHERE id = ?", (respuesta, match_id))
    conn.commit()
    conn.close()
    flash("✅ ¡Gracias por tu respuesta!")
    return redirect('/conexion_alfa_matches')

@app.route('/subir_foto_match', methods=['GET', 'POST'])
def subir_foto_match():
    if 'correo' not in session:
        return redirect('/')

    correo = session['correo']
    conn = get_db_connection()
    match = conn.execute('''
        SELECT * FROM conexion_alfa_matches 
        WHERE (correo_1 = ? OR correo_2 = ?)
        LIMIT 1
    ''', (correo, correo)).fetchone()

    if not match:
        conn.close()
        flash("❌ No tienes un match asignado.")
        return redirect(url_for('conexion_alfa'))

    if match['evidencia']:
         flash("✅ Este equipo ya subió su foto de evidencia.")
         return redirect(url_for('conexion_alfa_match'))

    if request.method == 'POST':
        archivo = request.files.get('foto')
        if archivo and archivo.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            nombre_archivo = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{archivo.filename}"
            carpeta = os.path.join('static', 'evidencias_alfa')
            os.makedirs(carpeta, exist_ok=True)
            ruta = os.path.join(carpeta, nombre_archivo)

            try:
                archivo.save(ruta)
                conn.execute('UPDATE conexion_alfa_matches SET evidencia = ? WHERE id = ?',
                             (nombre_archivo, match['id']))
                conn.commit()
                flash("✅ ¡Excelente! Foto subida exitosamente.")
                return redirect(url_for('conexion_alfa_match'))
            except Exception as e:
                flash(f"❌ Error al guardar la foto: {e}")
        else:
            flash("❌ Formato de archivo no válido. Usa png, jpg o jpeg.")

    conn.close()
    # Se renderiza una plantilla genérica para subir la foto.
    return render_template('conexion_alfa_subir_foto.html', match=match)

@app.route('/conexion_alfa_match')
def conexion_alfa_match():
    if 'correo' not in session:
        return redirect('/')

    correo = session['correo']
    conn = get_db_connection()

    # Buscar match si existe
    match = conn.execute('''
        SELECT * FROM conexion_alfa_matches 
        WHERE correo_1 = ? OR correo_2 = ?
        LIMIT 1
    ''', (correo, correo)).fetchone()

    feedback_dado = match['feedback'] if match and match['feedback'] is not None else None

    conn.close()
    return render_template(
        "conexion_alfa_match.html",
        match=match,
        sin_match=(match is None),
        feedback_dado=feedback_dado
    )

@app.route('/api/conexion_alfa_match', methods=['POST'])
def api_conexion_alfa_match():

    data = request.get_json()
    participantes = data.get("participantes", [])

    if not participantes or len(participantes) < 2:
        return jsonify({"error": "No hay suficientes participantes"}), 400

    textos, correos, nombres, perfiles, respuestas_dict = [], [], [], [], []

    for p in participantes:
        respuestas = [p.get(f"r{i}", "") or "" for i in range(1, 8)]
        textos.append(" ".join(respuestas))
        correos.append(p["correo"])
        nombres.append(p["nombre"])
        perfiles.append(p.get("perfil_ia", ""))
        respuestas_dict.append(p) # Guardamos el diccionario completo

    vectores = vectorizer_ia.transform(textos).toarray()

    usados = set()
    matches = []

    for i in range(len(correos)):
        if i in usados: continue

        mejor_j = None
        mejor_sim = -1

        for j in range(i + 1, len(correos)):
            if j in usados: continue
            sim = cosine_similarity([vectores[i]], [vectores[j]])[0][0]
            if sim > mejor_sim:
                mejor_sim = sim
                mejor_j = j

        if mejor_j is not None:
            p1 = respuestas_dict[i]
            p2 = respuestas_dict[mejor_j]

            # --- Lógica para encontrar temas en común ---
            temas_comunes = []
            if p1.get('r4') and p1['r4'] == p2.get('r4'):
                temas_comunes.append(f"su película favorita en común: '{p1['r4']}'")
            if p1.get('r6') and p1['r6'] == p2.get('r6'):
                temas_comunes.append(f"su gusto por el deporte: '{p1['r6']}'")
            if p1.get('r2') and p1['r2'] == p2.get('r2'):
                temas_comunes.append(f"su pasión por '{p1['r2']}'")

            razon_match = f"Tienen una alta compatibilidad ({round(mejor_sim * 100)}%). "
            if temas_comunes:
                razon_match += "La IA detectó que coinciden en " + " y ".join(temas_comunes) + "."

            temas_sugeridos = [
                f"pueden conversar sobre qué es lo que más les gusta de '{p1.get('r4', 'el cine')}'",
                f"sería un gran tema para romper el hielo hablar de su pasión por '{p1.get('r2', 'sus hobbies')}'",
                f"podrían compartir su opinión sobre el mejor concierto al que han ido, como el de '{p1.get('r9', 'su artista favorito')}'"
            ]
            random.shuffle(temas_sugeridos)

            razon_final = (f"{razon_match}\n\n"
                           f"**Para romper el hielo:**\n"
                           f"La IA sugiere que {temas_sugeridos[0]}.")

            matches.append({
                "correo_1": correos[i], "correo_2": correos[mejor_j],
                "nombre_1": nombres[i], "nombre_2": nombres[mejor_j],
                "perfil_1": perfiles[i], "perfil_2": perfiles[mejor_j],
                "razon": razon_final
            })
            usados.add(i)
            usados.add(mejor_j)

    return jsonify({"matches": matches})

@app.route('/reset_conexion_alfa', methods=['POST'])
def reset_conexion_alfa():
    conn = get_db_connection()

    # Borrar registros de la base de datos
    conn.execute("DELETE FROM conexion_alfa_matches")
    conn.execute("DELETE FROM conexion_alfa_respuestas")
    conn.commit()
    conn.close()

    # Borrar archivos de evidencia de video
    carpeta = 'static/evidencias_alfa'
    if os.path.exists(carpeta):
        for archivo in os.listdir(carpeta):
            ruta = os.path.join(carpeta, archivo)
            if os.path.isfile(ruta):
                os.remove(ruta)

    flash("✅ Conexión Alfa reiniciado correctamente.")
    return redirect('/admin_panel')

@app.route('/forzar_matches_conexion_alfa', methods=['POST'])
def forzar_matches_conexion_alfa():
    import subprocess
    subprocess.call(["python", "generar_matches_conexion_alfa.py"])
    flash("✅ Matches de Conexión Alfa generados correctamente.")
    return redirect('/admin_panel')

@app.route('/reset_datos_participantes', methods=['POST'])
def reset_datos_participantes():
    conn = get_db_connection()
    conn.execute("DELETE FROM conexion_alfa_respuestas")
    conn.execute("DELETE FROM adivina_participantes")
    conn.commit()
    conn.close()
    flash("✅ Datos de participantes reiniciados. Todos podrán volver a llenar el formulario.")
    return redirect('/admin_panel')

@app.route('/eliminar_todos_los_jugadores', methods=['POST'])
def eliminar_todos_los_jugadores():
    conn = get_db_connection()
    conn.execute("DELETE FROM jugadores")
    conn.execute("DELETE FROM conexion_alfa_respuestas")
    conn.execute("DELETE FROM conexion_alfa_matches")
    conn.execute("DELETE FROM adivina_resultados")
    conn.execute("DELETE FROM adivina_participantes")
    conn.commit()
    conn.close()
    session.clear()  # Limpiar sesión activa
    flash("✅ Se eliminaron todos los jugadores, respuestas y sesiones.")
    return redirect('/admin_panel')

@app.route('/feedback_match', methods=['POST'])
def feedback_match():
    if 'correo' not in session:
        return redirect('/')

    feedback = int(request.form.get('feedback'))
    match_id = int(request.form.get('match_id'))

    conn = get_db_connection()
    conn.execute("UPDATE conexion_alfa_matches SET feedback = ? WHERE id = ?", (feedback, match_id))
    conn.commit()
    conn.close()
    
    flash("✅ Gracias por tu feedback sobre la conexión.")
    return redirect("/conexion_alfa_match")

# -------------------- SUBE TU FOTO --------------------
@app.route('/sube_tu_foto', methods=['GET', 'POST'])
def sube_tu_foto():
    if 'jugador' not in session:
        return redirect('/')

    correo = session['correo']
    nombre = session['jugador']
    conn = get_db_connection()

    ya_existe = conn.execute("SELECT * FROM reto_equipo_foto WHERE correo = ?", (correo,)).fetchone()

    if request.method == 'POST':
        equipo = request.form.get('equipo')
        archivo = request.files.get('foto')

        if not equipo or not archivo:
            flash("❌ Faltan datos.")
            return redirect('/sube_tu_foto')

        nombre_archivo = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{archivo.filename}"
        carpeta = os.path.join('static/fotos_equipo')
        os.makedirs(carpeta, exist_ok=True)
        archivo.save(os.path.join(carpeta, nombre_archivo))

        conn.execute(
            "INSERT INTO reto_equipo_foto (nombre_participante, correo, equipo, archivo) VALUES (?, ?, ?, ?)",
            (nombre, correo, equipo, nombre_archivo)
        )
        conn.commit()
        conn.close()

        flash("✅ Foto subida con éxito.")
        return redirect('/')
    
    conn.close()
    return render_template("reto_equipo_foto.html", ya_existe=ya_existe)

@app.route('/reto_equipo_foto', methods=['GET', 'POST'])
def reto_equipo_foto():
    if 'jugador' not in session:
        return redirect('/login')

    nombre = session.get('jugador')
    correo = session.get('correo')

    conn = get_db_connection()
    ya_existe = conn.execute("SELECT * FROM reto_equipo_foto WHERE correo = ?", (correo,)).fetchone()

    if request.method == 'POST':
        equipo = request.form.get('equipo')
        archivo = request.files.get('foto')

        if not equipo or not archivo:
            flash("❌ Faltan datos.")
            return redirect('/reto_equipo_foto')

        nombre_archivo = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{archivo.filename}"
        carpeta = os.path.join('static/fotos_equipo')
        os.makedirs(carpeta, exist_ok=True)
        ruta = os.path.join(carpeta, nombre_archivo)
        archivo.save(ruta)

        conn.execute(
            '''
            INSERT INTO reto_equipo_foto (nombre_participante, correo, equipo, archivo)
            VALUES (?, ?, ?, ?)
            ''',
            (nombre, correo, equipo, nombre_archivo)
        )
        conn.commit()
        conn.close()

        flash("✅ Foto subida exitosamente.")
        return redirect('/')

    conn.close()
    return render_template('reto_equipo_foto.html', ya_existe=ya_existe)

@app.route('/ver_fotos_equipo')
def ver_fotos_equipo():
    conn  = get_db_connection()
    filas = conn.execute("""
        SELECT equipo, reto_no, archivo
        FROM reto_equipo_foto
        ORDER BY equipo, reto_no
    """).fetchall()
    conn.close()

    equipos = {}
    for f in filas:
        equipos.setdefault(f['equipo'], {})[f['reto_no']] = f['archivo']

    return render_template("ver_fotos_equipo.html", equipos=equipos)

@app.route('/reset_reto_equipo_foto', methods=['POST'])
def reset_reto_equipo_foto():
    conn = get_db_connection()
    conn.execute("DELETE FROM reto_equipo_foto")
    conn.commit()
    conn.close()

    # Eliminar archivos físicamente del folder
    carpeta = 'static/fotos_equipo'
    if os.path.exists(carpeta):
        for archivo in os.listdir(carpeta):
            ruta = os.path.join(carpeta, archivo)
            if os.path.isfile(ruta):
                os.remove(ruta)

    flash("✅ Todas las fotos del reto 'Sube tu foto' han sido eliminadas.")
    return redirect('/admin_panel')

# -------------------- RUN --------------------
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER_GRUPAL'], exist_ok=True)
    os.makedirs('static/fotos_reto_foto', exist_ok=True)
    app.run(debug=True)
