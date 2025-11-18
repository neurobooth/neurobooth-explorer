import numpy as np
import neurobooth_terra

import psycopg2
from sshtunnel import SSHTunnelForwarder

import credential_reader


# ==== Read Credentials ==== #
db_args, ssh_args = credential_reader.read_db_secrets()


def get_subjects_from_subject_table():
    subject_qry_command = "SELECT subject_id FROM subject"
    task_qry_command = "SELECT task_id, date_times FROM log_task"
    with SSHTunnelForwarder(**ssh_args) as tunnel:
        with psycopg2.connect(**db_args) as conn:
            subject_id_df = neurobooth_terra.query(conn, subject_qry_command, ['subject_id'])
            task_df = neurobooth_terra.query(conn, task_qry_command, ['task_id', 'session_datetime'])
    
    # building subject_list
    subj_list = [s for s in subject_id_df.subject_id.tolist() if not len(s)<6]

    # building task and session_date list
    task_df.dropna(inplace=True)
    task_df['session_date'] = [i[0].date() for i in task_df.session_datetime]
    task_list = list(np.sort(task_df.task_id.unique()))
    session_date_list = [str(i) for i in np.sort(task_df.session_date.unique())[::-1]]

    # building Primary_Diagnosis list
    clinical_list = ['Select Primary Diagnosis', 'FA', 'Control']

    subj_list.insert(0, 'Select Subject ID')
    session_date_list.insert(0, 'Select Date')
    task_list.insert(0, 'Select Task')

    return subj_list, session_date_list, task_list, clinical_list


if __name__ == '__main__':
    subj_list, session_date_list, task_list, clinical_list = get_subjects_from_subject_table()
    print(subj_list)
    print(session_date_list)
    print(task_list)
    print(clinical_list)