import sqlite3
import os
from datetime import datetime
import logging
from typing import List, Dict, Optional, Tuple
import json

from api.config import Config

logger = logging.getLogger(__name__)
config = Config()


class MatchingDB:
    def __init__(self, db_path: str = "name_matching.db"):
        """Initialize the database connection and create tables if they don't exist"""
        self.db_path = config.DB_PATH
        self.init_database()

    def init_database(self):
        """Create the database tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create predictions table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        english_name TEXT NOT NULL,
                        thai_name TEXT NOT NULL,
                        english_name_normalized TEXT,
                        thai_name_normalized TEXT,
                        similarity_score REAL NOT NULL,
                        is_match BOOLEAN NOT NULL,
                        confidence TEXT,
                        threshold_used REAL,
                        timestamp TEXT NOT NULL,
                        request_ip TEXT,
                        processing_time_ms REAL
                    )
                """
                )

                # Create batch_predictions table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS batch_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        batch_id TEXT NOT NULL,
                        total_pairs INTEGER NOT NULL,
                        successful_predictions INTEGER NOT NULL,
                        failed_predictions INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        request_ip TEXT,
                        processing_time_ms REAL
                    )
                """
                )

                # Create indexes for better performance
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
                    ON predictions(timestamp)
                """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_predictions_is_match 
                    ON predictions(is_match)
                """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_batch_predictions_timestamp 
                    ON batch_predictions(timestamp)
                """
                )

                conn.commit()
                logger.info(f"Database initialized successfully at {self.db_path}")

        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def save_prediction(
        self,
        english_name: str,
        thai_name: str,
        english_name_normalized: str,
        thai_name_normalized: str,
        similarity_score: float,
        is_match: bool,
        confidence: str,
        threshold_used: float,
        request_ip: str = None,
        processing_time_ms: float = None,
    ) -> int:
        """Save a single prediction to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO predictions (
                        english_name, thai_name, english_name_normalized, 
                        thai_name_normalized, similarity_score, is_match, 
                        confidence, threshold_used, timestamp, request_ip, 
                        processing_time_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        english_name,
                        thai_name,
                        english_name_normalized,
                        thai_name_normalized,
                        similarity_score,
                        is_match,
                        confidence,
                        threshold_used,
                        datetime.now().isoformat(),
                        request_ip,
                        processing_time_ms,
                    ),
                )

                prediction_id = cursor.lastrowid
                conn.commit()

                logger.info(f"Prediction saved with ID: {prediction_id}")
                return prediction_id

        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            raise

    def save_batch_prediction(
        self,
        batch_id: str,
        total_pairs: int,
        successful_predictions: int,
        failed_predictions: int,
        request_ip: str = None,
        processing_time_ms: float = None,
    ) -> int:
        """Save batch prediction metadata"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO batch_predictions (
                        batch_id, total_pairs, successful_predictions,
                        failed_predictions, timestamp, request_ip,
                        processing_time_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        batch_id,
                        total_pairs,
                        successful_predictions,
                        failed_predictions,
                        datetime.now().isoformat(),
                        request_ip,
                        processing_time_ms,
                    ),
                )

                batch_record_id = cursor.lastrowid
                conn.commit()

                logger.info(f"Batch prediction saved with ID: {batch_record_id}")
                return batch_record_id

        except Exception as e:
            logger.error(f"Error saving batch prediction: {str(e)}")
            raise

    def get_predictions(
        self,
        limit: int = 100,
        offset: int = 0,
        is_match: Optional[bool] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve predictions with optional filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = "SELECT * FROM predictions WHERE 1=1"
                params = []

                if is_match is not None:
                    query += " AND is_match = ?"
                    params.append(is_match)

                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)

                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)

                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                cursor.execute(query, params)
                rows = cursor.fetchall()

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error retrieving predictions: {str(e)}")
            raise

    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Total predictions
                cursor.execute("SELECT COUNT(*) FROM predictions")
                total_predictions = cursor.fetchone()[0]

                # Match statistics
                cursor.execute("SELECT COUNT(*) FROM predictions WHERE is_match = 1")
                total_matches = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM predictions WHERE is_match = 0")
                total_non_matches = cursor.fetchone()[0]

                # Average similarity score
                cursor.execute("SELECT AVG(similarity_score) FROM predictions")
                avg_similarity = cursor.fetchone()[0] or 0

                # Confidence distribution
                cursor.execute(
                    """
                    SELECT confidence, COUNT(*) 
                    FROM predictions 
                    GROUP BY confidence
                """
                )
                confidence_dist = dict(cursor.fetchall())

                # Recent activity (last 24 hours)
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM predictions 
                    WHERE datetime(timestamp) >= datetime('now', '-1 day')
                """
                )
                recent_predictions = cursor.fetchone()[0]

                # Batch statistics
                cursor.execute("SELECT COUNT(*) FROM batch_predictions")
                total_batches = cursor.fetchone()[0]

                return {
                    "total_predictions": total_predictions,
                    "total_matches": total_matches,
                    "total_non_matches": total_non_matches,
                    "match_rate": round(
                        total_matches / max(total_predictions, 1) * 100, 2
                    ),
                    "average_similarity_score": round(avg_similarity, 4),
                    "confidence_distribution": confidence_dist,
                    "recent_predictions_24h": recent_predictions,
                    "total_batches": total_batches,
                }

        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            raise

    def export_predictions_csv(self, filepath: str, limit: int = None) -> bool:
        """Export predictions to CSV file"""
        try:
            import csv

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = "SELECT * FROM predictions ORDER BY timestamp DESC"
                if limit:
                    query += f" LIMIT {limit}"

                cursor.execute(query)
                rows = cursor.fetchall()

                if not rows:
                    logger.warning("No predictions found to export")
                    return False

                with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
                    fieldnames = rows[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writeheader()
                    for row in rows:
                        writer.writerow(dict(row))

                logger.info(f"Exported {len(rows)} predictions to {filepath}")
                return True

        except Exception as e:
            logger.error(f"Error exporting predictions: {str(e)}")
            return False

    def cleanup_old_records(self, days_to_keep: int = 30) -> int:
        """Remove old records to manage database size"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Delete old predictions
                cursor.execute(
                    """
                    DELETE FROM predictions 
                    WHERE datetime(timestamp) < datetime('now', '-{} days')
                """.format(
                        days_to_keep
                    )
                )

                deleted_predictions = cursor.rowcount

                # Delete old batch records
                cursor.execute(
                    """
                    DELETE FROM batch_predictions 
                    WHERE datetime(timestamp) < datetime('now', '-{} days')
                """.format(
                        days_to_keep
                    )
                )

                deleted_batches = cursor.rowcount

                conn.commit()

                # Vacuum to reclaim space
                cursor.execute("VACUUM")

                total_deleted = deleted_predictions + deleted_batches
                logger.info(f"Cleaned up {total_deleted} old records")

                return total_deleted

        except Exception as e:
            logger.error(f"Error cleaning up old records: {str(e)}")
            raise

    def close(self):
        """Close database connection (if needed for cleanup)"""
        # SQLite connections are automatically closed when using context managers
        # This method is here for compatibility if needed
        pass
