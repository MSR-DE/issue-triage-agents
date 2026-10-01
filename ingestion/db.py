import os 
from dotenv import load_dotenv
import psycopg

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing from .env")

def get_db_connection():
    return psycopg.connect(DATABASE_URL)


if __name__ == "__main__":
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM issues")
            print(cur.fetchone())

