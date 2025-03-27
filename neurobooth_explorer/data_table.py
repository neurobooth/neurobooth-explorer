import numpy as np

import neurobooth_terra

import psycopg2
from sshtunnel import SSHTunnelForwarder

from datetime import date

import credential_reader


# ==== Read Credentials ==== #
ssh_args = credential_reader.get_ssh_args()
db_args = credential_reader.read_db_secrets()


# --- Function to compute age from date of birth --- #
def calculate_age(dob):
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


###### Building Master Data Table ######

sql_query_cmd = """
SELECT subject.subject_id, subject.gender_at_birth, subject.date_of_birth_subject, log_task.date_times, log_task.log_task_id,
       log_task.task_id, log_sensor_file.device_id, log_sensor_file.file_start_time, log_sensor_file.sensor_file_path,
       rc_clinical.neurologist, rc_clinical.primary_diagnosis, rc_clinical.other_primary_diagnosis, rc_clinical.secondary_diagnosis,
       rc_clinical.diagnosis_notes
FROM (((log_task
INNER JOIN log_sensor_file ON log_task.log_task_id = log_sensor_file.log_task_id)
INNER JOIN subject ON log_task.subject_id = subject.subject_id)
INNER JOIN rc_clinical ON subject.subject_id = rc_clinical.subject_id);
"""

def rebuild_master_data_table(sql_query_cmd):
    
    # # --- Querying on control --- #
    # database = secrets['database']['dbname']
    # host = secrets['database']['host']
    # port = 5432
    # with psycopg2.connect(database=database, 
    #                         user=secrets['database']['user'],
    #                         password=secrets['database']['pass'],
    #                         host=host,
    #                         port=port) as conn:
    #     nb_data_df = neurobooth_terra.query(conn,sql_query_cmd, ['subject_id', 'gender_at_birth', 'dob', 'session_datetime', 'task_log', 'tasks', 'file_names'])
    
    def get_prim_diag(x):
        if len(x)==0:
            return None
        x_diag=[]
        for i in x:
            if str(i) in prim_diag_dict.keys():
                x_diag.append(prim_diag_dict[str(i)])
            else:
                x_diag.append(str(i))
        return x_diag

    # queries for fetching primary diagnosis and neurologist response arrays
    prim_diag_qry = """
    SELECT rc_data_dictionary.response_array FROM rc_data_dictionary
    WHERE rc_data_dictionary.field_name = 'primary_diagnosis'
    """

    neurologist_qry = """
    SELECT rc_data_dictionary.response_array FROM rc_data_dictionary
    WHERE rc_data_dictionary.field_name = 'neurologist'
    """

    # query for bars scores
    bars_qry='''
    SELECT subject_id, end_time_ataxia_pd_scales, bars_total, bars_gait, bars_heel_shin_left, bars_heel_shin_right, bars_finger_nose_left,
    bars_finger_nose_right, bars_speech, bars_oculomotor from rc_ataxia_pd_scales
    '''

    # --- Querying Neurobooth Terra Database --- #
    with SSHTunnelForwarder(**ssh_args) as tunnel:
        with psycopg2.connect(**db_args) as conn:
            nb_data_df = neurobooth_terra.query(conn,sql_query_cmd, ['subject_id', 'gender_at_birth', 'dob', 'session_datetime', 'session_id',
                                                                    'tasks', 'device_id', 'task_datetime', 'file_names', 'neurologist',
                                                                    'primary_diagnosis', 'other_primary_diagnosis', 'secondary_diagnosis',
                                                                    'diagnosis_notes'])
            
            prim_diag_df = neurobooth_terra.query(conn, prim_diag_qry, ['prim_diag_dict'])

            neurologist_df = neurobooth_terra.query(conn, neurologist_qry, ['neurologist_dict'])

            bars_df = neurobooth_terra.query(conn, bars_qry, ['subject_id', 'end_time_ataxia_pd_scales',
                                                            'bars_total_score', 'bars_gait',
                                                            'bars_heel_shin_left', 'bars_heel_shin_right',
                                                            'bars_finger_nose_left', 'bars_finger_nose_right',
                                                            'bars_speech', 'bars_oculomotor'])
    # ------------------------------------------ #

    prim_diag_dict = prim_diag_df.prim_diag_dict[0]
    neurologist_dict = neurologist_df.neurologist_dict[0]
    drop_na_cols = ['subject_id', 'gender_at_birth', 'dob', 'session_datetime', 'session_id',
                    'tasks', 'device_id', 'task_datetime', 'file_names']
    nb_data_df.dropna(inplace=True, subset=drop_na_cols)
    nb_data_df['age'] = nb_data_df.dob.apply(calculate_age)
    nb_data_df['primary_diagnosis'] = nb_data_df['primary_diagnosis'].apply(get_prim_diag)
    nb_data_df['neurologist'] = nb_data_df['neurologist'].apply(lambda x: neurologist_dict[str(int(x))] if not np.isnan(x) and str(int(x)) in neurologist_dict.keys() else None)
    nb_data_df['session_date'] = [i[0].date() for i in nb_data_df.session_datetime]
    nb_data_df['gender'] = ['M' if int(i)==1 else 'F' for i in nb_data_df.gender_at_birth]
    col_reorder = ['subject_id', 'gender', 'dob', 'age', 'neurologist', 'primary_diagnosis', 'other_primary_diagnosis', 'secondary_diagnosis', 'diagnosis_notes',
                'session_id', 'session_date', 'session_datetime', 'tasks', 'device_id', 'task_datetime', 'file_names']
    nb_data_df = nb_data_df[col_reorder]
    # creating hdf5 file column
    hdf5_arr = [np.nan] * len(nb_data_df)
    for ix, fil_arr in enumerate(nb_data_df.file_names.tolist()):
        for fil in fil_arr:
            if fil[-5:] == '.hdf5':
                hdf5_arr[ix] = fil
    nb_data_df['hdf5_files'] = hdf5_arr
    # nb_data_df.dropna(inplace=True)

    # --- Generating dropdown lists --- #
    sub_id_list = [str(j) for j in np.sort([int(i) for i in nb_data_df.subject_id.unique()])[::-1]]
    session_date_list = [str(i) for i in np.sort(nb_data_df.session_date.unique())[::-1]]
    task_list = nb_data_df.tasks.unique()
    # clinical_list = ['Ataxia-Telangiectasia','Spino Cerebellar Ataxia', 'Parkinsonism']
    diagnoses_list = [diag for diag_list in nb_data_df.primary_diagnosis.tolist() for diag in diag_list]
    clinical_list = []
    for i in  list(prim_diag_dict.values()): # this loop is so that list elements are in same order as db response array
        if i in diagnoses_list:
            clinical_list.append(i)

    return nb_data_df, bars_df, sub_id_list, session_date_list, task_list, clinical_list


if __name__ == '__main__':

    import time

    t0 = time.time()
    nb_data_df, bars_df, sub_id_list, session_date_list, task_list, clinical_list = rebuild_master_data_table(sql_query_cmd)
    t1 = time.time()
    print(f'Took {t1-t0:.3f} seconds to rebuild data table')
    print(nb_data_df.shape)
    print(bars_df.shape)

