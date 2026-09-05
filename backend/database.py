import mysql.connector


def get_connection():
    return mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="bU-170417",
        database="stampede_dbase",
        connection_timeout=5,
        use_pure=True
    )