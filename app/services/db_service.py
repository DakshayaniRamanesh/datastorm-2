"""
DataStorm 2026 - SQLite Abstraction & DB Compiler Service
=========================================================
Manages SQLite database connections and handles the compilation of
Silver transactions, Gold features, and allocations into indexed database tables.

Ensures blazing-fast paginated queries, searches, and aggregations.
"""

import sqlite3
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger("DBService")
DB_PATH = Path(__file__).parent.parent.parent / "data" / "outlet_intelligence.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Parquet file paths
ROOT = Path(__file__).parent.parent.parent
SILVER = ROOT / "pipeline" / "silver"
GOLD = ROOT / "pipeline" / "gold"
OUTPUT = ROOT / "output"

class DBService:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        
    def get_connection(self) -> sqlite3.Connection:
        """Get standard sqlite3 connection with dict-based row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def compile_db(self, force: bool = False) -> None:
        """Compile Silver/Gold parquet files into indexed SQLite tables.

        Saves disk footprint by aggregating transactions to monthly outlet level.
        """
        if self.db_path.exists() and not force:
            logger.info("SQLite database already exists. Skipping compilation.")
            return

        logger.info(f"Compiling SQLite database at: {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 1. Compile Outlets table (from gold_features.parquet)
            gold_path = GOLD / "gold_features.parquet"
            if not gold_path.exists():
                raise FileNotFoundError(f"Missing required source: {gold_path}")
            
            logger.info("Loading gold features parquet...")
            gold_df = pd.read_parquet(gold_path)
            
            # Write outlets
            logger.info("Writing 'outlets' table to SQLite...")
            gold_df.to_sql("outlets", conn, if_exists="replace", index=False)
            
            # 2. Compile Transactions table (from silver/transactions.parquet)
            tx_path = SILVER / "transactions.parquet"
            if not tx_path.exists():
                raise FileNotFoundError(f"Missing required source: {tx_path}")
                
            logger.info("Loading silver transactions parquet...")
            tx_df = pd.read_parquet(tx_path)
            
            # Aggregate transactions to monthly outlet level to make SQLite lightweight and fast
            logger.info("Aggregating transactions to monthly level...")
            tx_agg = (
                tx_df.groupby(["Outlet_ID", "Year", "Month", "Distributor_ID"])
                .agg(
                    monthly_volume=("Volume_Liters", "sum"),
                    total_revenue=("Total_Bill_Value", "sum"),
                    sku_count=("SKU_ID", "nunique"),
                    txn_count=("SKU_ID", "count")
                )
                .reset_index()
            )
            
            logger.info("Writing 'transactions' table to SQLite...")
            tx_agg.to_sql("transactions", conn, if_exists="replace", index=False)

            # 3. Compile allocations table (from budget allocations CSV)
            alloc_path = OUTPUT / "ai_aces_budget_allocations.csv"
            if alloc_path.exists():
                logger.info("Loading budget allocations CSV...")
                alloc_df = pd.read_csv(alloc_path)
                logger.info("Writing 'allocations' table to SQLite...")
                alloc_df.to_sql("allocations", conn, if_exists="replace", index=False)
            else:
                logger.warning(f"Allocations CSV not found at: {alloc_path}. Creating empty allocations table.")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS allocations (
                        Outlet_ID TEXT PRIMARY KEY,
                        Trade_Spend_LKR REAL,
                        Expected_Lift REAL,
                        ROI REAL
                    )
                """)

            # 4. Create indices for high-performance querying
            logger.info("Creating database indices...")
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_outlets_pk ON outlets(Outlet_ID);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_outlets_dist ON outlets(primary_dist);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_outlets_type ON outlets(Outlet_Type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_outlet ON transactions(Outlet_ID);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(Year, Month);")
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_allocations_pk ON allocations(Outlet_ID);")

            conn.commit()
            logger.info("SQLite database compilation complete.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Database compilation failed: {e}")
            raise e
        finally:
            conn.close()

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Run a query and return results as a list of dictionaries."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def execute_scalar(self, query: str, params: tuple = ()) -> Any:
        """Run a query and return the first column of the first row."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return row[0] if row else None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    db_service = DBService()
    db_service.compile_db(force=True)
