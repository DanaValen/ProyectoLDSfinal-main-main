import mysql.connector

conexion = mysql.connector.connect(user='root', password='3306',
                                    host='localhost',
                                    database='lds_c',
                                    port='3306')
print(conexion)

