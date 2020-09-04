#!/usr/bin/env python
from datetime import datetime

import numpy as np
from logging_script import logger

def setup_logger(title, n_state_vals, action_dim, goal_dim):
    int_keys = ["run_id", "episode", "step"]

    float_keys = ["from_state_" + str(n) for n in range(n_state_vals)]
    float_keys += ["to_state_" + str(n) for n in range(n_state_vals)]
    float_keys += ["action"]# ["action_" + str(n) for n in range(1)]
    float_keys += ["goal_" + str(n) for n in range(goal_dim)]
    float_keys += ["q_vals_" + str(n) for n in range(action_dim)]
    float_keys += ["Reward"]


    string_keys = ["Title"]

    bool_keys = ["Terminal"]


    time_keys = ["Timestamp"]

    keys_ = int_keys + float_keys + string_keys + bool_keys + time_keys

    dtypes_ = ["bigint" for _ in range(len(int_keys))] + \
              ["real" for _ in range(len(float_keys))] + \
              ["varchar" for _ in range(len(string_keys))] + \
              ["boolean" for _ in range(len(bool_keys))] + \
              ["timestamp" for _ in range(len(time_keys))]


    db_config = {
        "database": {
            "host": "dwh.prd.akw",
            "user": "lwidowski",
            "passwd": "$moothOperat0r",
            "database": "sandbox",
            "port": "5432"
        },
        "schema_name": "lwidowski",
        "table_name": "tb_b_" + title,
        "key_list": keys_,
        "dtype_list": dtypes_,
        "primary_key": None,
        "auto_increment": None
    }

    log = logger(title=title + ".log", keys=keys_, dtypes=dtypes_, sep="\t", load_full=False, db_config=db_config)
    log.db_create_table()
    return log, keys_

def make_log_entry(log, title, run_id, episode_number,
                   episode_step, from_state, to_state, goal,
                   action, q_vals,
                   reward, terminal):
    int_vals = [run_id, episode_number, episode_step]

    float_vals = np.asarray(from_state).flatten().tolist()
    float_vals += np.asarray(to_state).flatten().tolist()
    float_vals += [action] #action.flatten().tolist()
    float_vals += np.asarray(goal).flatten().tolist()
    float_vals += np.asarray(q_vals).flatten().tolist()
    float_vals += [reward]

    string_vals = [title]

    bool_vals = [terminal]

    time_vals = [str(datetime.datetime.now())]

    vals = int_vals + float_vals + string_vals + bool_vals + time_vals
    log.write_line(vals)
