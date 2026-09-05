import mysql.connector

print("1. Python started")

print("2. Import successful")

print("3. Trying MySQL connection...")

connection = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password="bU-170417",
    database="stampede_dbase",
    connection_timeout=5,
    use_pure=True
)
print("4. Connection successful!")

connection.close()

print("5. Connection closed")