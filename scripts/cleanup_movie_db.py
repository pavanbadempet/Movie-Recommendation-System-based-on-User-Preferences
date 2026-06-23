import logging
import os
import sys

from sqlalchemy import create_engine, inspect

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    # Try to read DATABASE_URL or ask for it
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is not set!")
        logger.info("Please run the script with DATABASE_URL set, for example:")
        logger.info('  Windows (PowerShell): $env:DATABASE_URL="postgresql://..."; python scripts/cleanup_movie_db.py')
        logger.info('  Linux/Mac: DATABASE_URL="postgresql://..." python scripts/cleanup_movie_db.py')
        sys.exit(1)

    logger.info("Connecting to database...")
    try:
        engine = create_engine(database_url)
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        logger.info(f"Existing tables in database: {existing_tables}")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        logger.error(
            "If Neon project transfer limit is exceeded, this connection will fail until the limit resets or you upgrade."
        )
        sys.exit(1)

    # Healthcare-related tables to clean up
    healthcare_tables = [
        # Clinical & Patient tables
        "clinical_record",
        "fact_clinical_record",
        "dim_patient",
        "dim_doctor",
        "patient",
        "doctor",
        # Appointment tables
        "appointment",
        "dim_appointment",
        "fact_appointment",
        # Billing & Payments
        "billing",
        "dim_billing",
        "fact_billing",
        "payment",
        "transaction",
        # Care events & Operations
        "care_event",
        "fact_care_event",
        "discharge_summary",
        "hospital_stats",
        "dim_facility",
        # Terminology & Pharmacy
        "prescription",
        "medication",
        "dim_pharmacy",
        "medical_record",
        "diagnosis",
        # System & Audit
        "audit_log",
        "incident_report",
        "incident_response",
        "compliance_log",
        "alembic_version",
    ]

    tables_to_drop = [t for t in healthcare_tables if t in existing_tables]

    if not tables_to_drop:
        logger.info("No healthcare-related tables found in the database. Nothing to clean up!")
        return

    logger.info(f"Found {len(tables_to_drop)} healthcare tables to clean up: {tables_to_drop}")

    # Confirm with user
    logger.info("Dropping tables...")
    try:
        with engine.connect() as conn:
            # We need to disable foreign key checks temporarily if there are cross-references
            trans = conn.begin()
            try:
                for table in tables_to_drop:
                    logger.info(f"Dropping table: {table}")
                    conn.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE;')
                trans.commit()
                logger.info("Cleanup completed successfully!")
            except Exception as drop_err:
                trans.rollback()
                raise drop_err
    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
