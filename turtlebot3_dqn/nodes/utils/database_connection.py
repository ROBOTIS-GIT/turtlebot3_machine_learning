import psycopg2


formatting_str_dict = {
    "datetime": "%s",
    "float": "%f",
    "double": "%f",
    "int": "%s",
    "str": "%s"
}


def create_table(config):
    conn = psycopg2.connect(user=config["database"]["user"],
                            password=config["database"]["passwd"],
                            host=config["database"]["host"],
                            port=config["database"]["port"],
                            database=config["database"]["database"])
    mycursor = conn.cursor()

    execute_str = "CREATE TABLE " + config["schema_name"] + "." + config["table_name"] + " ("
    for i in range(len(config["key_list"])):
        execute_str += " " + config["key_list"][i] + " " + config["dtype_list"][i]
        if config["key_list"][i] == config["auto_increment"]:
            execute_str += "  AUTO_INCREMENT"
        if config["key_list"][i] == config["primary_key"]:
            execute_str += " PRIMARY KEY"
        if i != len(config["key_list"]) - 1:
            execute_str += ","
    execute_str += ")"
    mycursor.execute(execute_str)
    conn.commit()


def table_exists(config):
    conn = psycopg2.connect(user=config["database"]["user"],
                            password=config["database"]["passwd"],
                            host=config["database"]["host"],
                            port=config["database"]["port"],
                            database=config["database"]["database"])
    mycursor = conn.cursor()
    mycursor.execute(
        "SELECT tablename FROM pg_catalog.pg_tables where schemaname = '{schema_name}' and tablename = '{table_name}';".format(
            table_name=config["table_name"].lower(), schema_name=config["schema_name"]))

    return mycursor.rowcount


def get_existing_tables(config):
    conn = psycopg2.connect(user=config["database"]["user"],
                            password=config["database"]["passwd"],
                            host=config["database"]["host"],
                            port=config["database"]["port"],
                            database=config["database"]["database"])
    mycursor = conn.cursor()
    mycursor.execute(
        "SELECT tablename FROM pg_catalog.pg_tables where schemaname = '{schema_name}';".format(
            schema_name=config["schema_name"]))
    return [c for c in mycursor]


def delete_table(config):
    conn = psycopg2.connect(user=config["database"]["user"],
                            password=config["database"]["passwd"],
                            host=config["database"]["host"],
                            port=config["database"]["port"],
                            database=config["database"]["database"])
    mycursor = conn.cursor()
    mycursor.execute("drop table " + config["schema_name"] + "." + config["table_name"])
    conn.commit()


def insert_to_table(config, data, show_progress=False):
    conn = psycopg2.connect(user=config["database"]["user"],
                            password=config["database"]["passwd"],
                            host=config["database"]["host"],
                            port=config["database"]["port"],
                            database=config["database"]["database"])
    mycursor = conn.cursor()
    if len(data) == data.size:
        N = 1
        data = [data.tolist()]
    else:
        N = len(data)
    for i in range(N):
        dat = data[i]
        sql = "INSERT INTO " + config["schema_name"] + "." + config["table_name"] + " ( "
        for i in range(len(config["key_list"])):
            sql += config["key_list"][i]
            if i != len(config["key_list"]) - 1:
                sql += ", "
        sql += ") values ("
        for i in range(len(config["key_list"])):
            sql += "%s"  # formatting_str_dict[db_config["dtype_list"][i]]
            if i != len(config["key_list"]) - 1:
                sql += ", "
        sql += ")"
        val = tuple(dat)
        mycursor.execute(sql, val)
        conn.commit()


def get_table_values(config):
    conn = psycopg2.connect(user=config["database"]["user"],
                            password=config["database"]["passwd"],
                            host=config["database"]["host"],
                            port=config["database"]["port"],
                            database=config["database"]["database"])
    mycursor = conn.cursor()
    mycursor.execute(
        "SELECT * FROM {schema_name}.{table_name}".format(
            schema_name=config["schema_name"], table_name=config["table_name"]))
    values = mycursor.fetchall()

    return values

def run_statement(config, statement):
    conn = psycopg2.connect(user=config["database"]["user"],
                            password=config["database"]["passwd"],
                            host=config["database"]["host"],
                            port=config["database"]["port"],
                            database=config["database"]["database"])
    mycursor = conn.cursor()
    mycursor.execute(statement)
    values = mycursor.fetchall()
    return values

def get_table_keys(config):
    conn = psycopg2.connect(user=config["database"]["user"],
                            password=config["database"]["passwd"],
                            host=config["database"]["host"],
                            port=config["database"]["port"],
                            database=config["database"]["database"])
    mycursor = conn.cursor()
    mycursor.execute("Select * FROM {schema_name}.{table_name} LIMIT 0".format(schema_name=config["schema_name"],
                                                                               table_name=config["table_name"]))
    return [desc[0] for desc in mycursor.description]
