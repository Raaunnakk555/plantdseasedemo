import os
import sqlite3

DB_PATH = os.getenv("DB_PATH", "app/db/diseases.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS diseases (
    name TEXT PRIMARY KEY,
    symptoms TEXT,
    treatment TEXT
);
"""

SEED = [
    ("Tomato___Late_blight",
     "Dark lesions on leaves and stems; leaves may wither rapidly in wet weather.",
     "Copper-based fungicide; remove infected leaves; avoid overhead watering."
    ),
    ("Tomato___Leaf_Mold",
     "Olive-green to brown velvety mold on undersides of leaves.",
     "Improve air circulation; apply chlorothalonil; sanitize tools."),
    ("Potato___Early_blight",
     "Brown spots with concentric rings on lower leaves.",
     "Rotate crops; apply mancozeb; remove debris."),
    ("Apple___Black_rot",
     "Circular black lesions on leaves and rotting fruit.",
     "Prune cankers; apply captan; destroy mummified fruit."),
    ("Healthy", "No visible disease symptoms.", "Maintain proper nutrition and watering.")
]

def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    conn = get_conn()
    with conn:
        conn.executescript(SCHEMA_SQL)
        # Seed if empty
        cur = conn.execute("SELECT COUNT(*) FROM diseases")
        (cnt,) = cur.fetchone()
        if cnt == 0:
            conn.executemany("INSERT INTO diseases(name, symptoms, treatment) VALUES (?, ?, ?)", SEED)
    conn.close()

def fetch_disease(name: str):
    conn = get_conn()
    try:
        cur = conn.execute("SELECT name, symptoms, treatment FROM diseases WHERE name = ?", (name,))
        row = cur.fetchone()
        if not row:
            return None
        return {"name": row[0], "symptoms": row[1], "treatment": row[2]}
    finally:
        conn.close()
