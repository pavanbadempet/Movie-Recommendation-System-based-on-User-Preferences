import os
import sys
import logging
from pathlib import Path
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is not set!")
        logger.info("Please set DATABASE_URL, e.g.:")
        logger.info("  Windows (PowerShell): $env:DATABASE_URL=\"postgresql://...\"; python scripts/init_postgres_db.py")
        sys.exit(1)
        
    logger.info("Connecting to database for schema initialization...")
    try:
        engine = create_engine(database_url)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Connection test successful!")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)
        
    migrations_dir = Path(__file__).resolve().parent.parent / "sql" / "migrations"
    migration_files = sorted([f for f in migrations_dir.glob("V*__*.sql")])
    
    if not migration_files:
        logger.error(f"No migration files found in {migrations_dir}")
        sys.exit(1)
        
    logger.info(f"Found {len(migration_files)} migrations to execute.")
    
    try:
        with engine.begin() as conn:
            for filepath in migration_files:
                logger.info(f"Executing migration: {filepath.name}...")
                with open(filepath, 'r', encoding='utf-8') as fh:
                    sql_content = fh.read()
                
                # Split commands by semicolon, ignoring empty elements
                # Note: This is a basic splitter. Since these files are standard CREATE/ALTER statements, splitting is clean.
                # If there are procedural blocks, we should run the whole string. 
                # Let's execute the file content directly. PostgreSQL allows executing multiple statements in one call.
                try:
                    conn.execute(text(sql_content))
                    logger.info(f"Completed migration: {filepath.name} ✓")
                except Exception as file_err:
                    logger.error(f"Error executing migration {filepath.name}: {file_err}")
                    raise file_err
        logger.info("Database schema initialized successfully! All migrations applied ✓")
    except Exception as e:
        logger.error(f"Migration runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
