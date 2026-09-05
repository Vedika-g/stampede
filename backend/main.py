from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

from backend.database import get_connection


app = FastAPI(
    title="AI Stampede Early Warning System",
    version="2.0"
)


# ============================================================
# DATA MODEL
# ============================================================

class AnalysisData(BaseModel):
    camera_id: str
    people_count: int
    density: float
    movement: float
    risk_score: float
    risk_level: str


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "AI Stampede Early Warning System Backend is running",
        "version": "2.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT 1")
        cursor.fetchone()

        cursor.close()
        connection.close()

        return {
            "status": "OK",
            "database": "Connected"
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "database": "Disconnected",
            "error": str(e)
        }


# ============================================================
# RECEIVE AND STORE AI ANALYSIS
# ============================================================

@app.post("/analysis")
def receive_analysis(data: AnalysisData):

    try:
        connection = get_connection()
        cursor = connection.cursor()

        query = """
            INSERT INTO analysis_results
            (
                camera_id,
                people_count,
                density,
                movement,
                risk_score,
                risk_level
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        values = (
            data.camera_id,
            data.people_count,
            data.density,
            data.movement,
            data.risk_score,
            data.risk_level
        )

        cursor.execute(query, values)
        connection.commit()

        inserted_id = cursor.lastrowid

        cursor.close()
        connection.close()

        return {
            "message": "Analysis saved successfully",
            "id": inserted_id,
            "timestamp": datetime.now(),
            "data": data.model_dump()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )


# ============================================================
# GET LATEST ANALYSIS
# ============================================================

@app.get("/analysis/latest")
def get_latest_analysis():

    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                id,
                camera_id,
                people_count,
                density,
                movement,
                risk_score,
                risk_level,
                timestamp
            FROM analysis_results
            ORDER BY id DESC
            LIMIT 1
        """

        cursor.execute(query)
        result = cursor.fetchone()

        cursor.close()
        connection.close()

        if result is None:
            return {
                "message": "No analysis data found"
            }

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )


# ============================================================
# GET ANALYSIS HISTORY
# ============================================================

@app.get("/analysis/history")
def get_analysis_history(limit: int = 50):

    try:

        if limit < 1:
            limit = 1

        if limit > 500:
            limit = 500

        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        query = f"""
            SELECT
                id,
                camera_id,
                people_count,
                density,
                movement,
                risk_score,
                risk_level,
                timestamp
            FROM analysis_results
            ORDER BY id DESC
            LIMIT {limit}
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        connection.close()

        return {
            "count": len(results),
            "data": results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )


# ============================================================
# GET HIGH-RISK ANALYSIS
# ============================================================

@app.get("/analysis/high-risk")
def get_high_risk():

    try:

        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                id,
                camera_id,
                people_count,
                density,
                movement,
                risk_score,
                risk_level,
                timestamp
            FROM analysis_results
            WHERE risk_level = 'HIGH'
            ORDER BY id DESC
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        connection.close()

        return {
            "count": len(results),
            "data": results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )


# ============================================================
# GET STATISTICS
# ============================================================

@app.get("/analysis/statistics")
def get_statistics():

    try:

        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                COUNT(*) AS total_readings,
                COALESCE(MAX(people_count), 0) AS maximum_people,
                COALESCE(MAX(density), 0) AS maximum_density,
                COALESCE(MAX(movement), 0) AS maximum_movement,
                COALESCE(MAX(risk_score), 0) AS maximum_risk_score,
                SUM(
                    CASE
                        WHEN risk_level = 'HIGH'
                        THEN 1
                        ELSE 0
                    END
                ) AS high_risk_readings,
                SUM(
                    CASE
                        WHEN risk_level = 'MEDIUM'
                        THEN 1
                        ELSE 0
                    END
                ) AS medium_risk_readings,
                SUM(
                    CASE
                        WHEN risk_level = 'LOW'
                        THEN 1
                        ELSE 0
                    END
                ) AS low_risk_readings
            FROM analysis_results
        """

        cursor.execute(query)
        result = cursor.fetchone()

        cursor.close()
        connection.close()

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )


# ============================================================
# GET DATA FOR A SPECIFIC CAMERA
# ============================================================

@app.get("/analysis/camera/{camera_id}")
def get_camera_analysis(camera_id: str):

    try:

        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                id,
                camera_id,
                people_count,
                density,
                movement,
                risk_score,
                risk_level,
                timestamp
            FROM analysis_results
            WHERE camera_id = %s
            ORDER BY id DESC
            LIMIT 100
        """

        cursor.execute(query, (camera_id,))
        results = cursor.fetchall()

        cursor.close()
        connection.close()

        return {
            "camera_id": camera_id,
            "count": len(results),
            "data": results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )