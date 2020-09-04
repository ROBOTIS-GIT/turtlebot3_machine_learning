import os
import io
import sys
import database_connection

import pandas as pd


class logger:
    def __init__(self, title="log.txt", path="/root/logs/", log="", keys=[], dtypes=[], sep=",", db_config=None, load_full=False):
        self.title = title
        self.path = path
        self.keys = keys
        self.dtypes = dtypes
        self.sep = sep
        self.load_full = load_full
        self.db_config = db_config
        self.header = self.make_header()

        if os.path.isfile(self.path + self.title):
            self.log_file_present = True
            print("log:", self.path + self.title, "is present")
            if self.load_full:
                with open(self.path + self.title, "r") as read_file:
                    self.log = read_file.read()
                    self.header = read_file.read(len(self.header))
                    print("header",self.header,type(self.header))
            else:
                with open(self.path + self.title, "r") as read_file:
                    old_header = read_file.read(len(self.header))
                    if self.header == old_header:
                        self.log = log
                    else:
                        print("keys dont match with old log")
                        # print("old_header:", old_header)
                        # print("set_header:", self.header)
                        sys.exit(0)

        else:
            self.log_file_present = False
            print("log:", self.path + self.title, "not present")
            self.log = self.header + log
            self.load_full = True
            self.save()
            self.log = ""
            self.load_full = False


        # if self.log[:len(self.header)] != self.header:
        #     self.log = self.header+self.log

    def make_header(self):
        header = ""
        for key in self.keys:
            header += str(key) + self.sep
        header = header[:-1] + "\n"
        return header

    def write_line(self, line):
        try:
            assert len(line) == len(self.keys)
            for cell in line:
                self.log += str(cell) + self.sep
            self.log = self.log[:-1] + "\n"
        except:
            print("No proper number of elements:", len(line), len(self.keys))
            pass

    def write(self, lines):
        if type(lines[0]) == list:
            for line in lines:
                self.write_line(line)
        else:
            self.write_line(lines)

    def save(self, save_to_db = False):
        if self.load_full:
            with open(self.path + self.title, "w") as write_file:
                write_file.write(self.log)
            self.log_file_present = True
            if save_to_db:
                self.save_log_to_database()
        else:
            if self.log_file_present:
                with open(self.path + self.title, "a") as write_file:
                    write_file.write(self.log)
            else:
                with open(self.path + self.title, "w") as write_file:
                    write_file.write(self.log)
                self.log_file_present = True
            if save_to_db:
                self.save_log_to_database()
            self.log = ""


    def to_DataFrame(self):
        if self.load_full:
            StringData = io.StringIO(self.log)
        else:
            StringData = io.StringIO(self.header + self.log)
        print(StringData.split(self.sep))
        return pd.read_csv(StringData, sep=self.sep)

    def save_log_to_database(self):
        if not database_connection.table_exists(self.db_config):
            database_connection.create_table(self.db_config)

        df = self.to_DataFrame()
        if len(df) > 0:
            database_connection.insert_to_table(self.db_config, df.values)
        else:
            print("no data")

    def delete_table(self):
            database_connection.delete_table(self.db_config)

    def db_create_table(self):
        if not database_connection.table_exists(self.db_config):
            database_connection.create_table(self.db_config)

# log = logger(keys = ["a","b"],load_full=False)
# lines = [[1,2]]
# log.write(lines)
# print(log.to_DataFrame())
# print(log.to_DataFrame().values)
