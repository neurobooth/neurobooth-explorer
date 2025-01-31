from typing import Optional, List, Union, Literal
import os
import yaml
import socket
from pydantic import BaseModel, AnyUrl, PositiveInt, DirectoryPath, Field
from pydantic.networks import IPvAnyAddress


class databaseArgs(BaseModel):
    db_name: str
    db_user: str
    password: str
    host: Union[str, IPvAnyAddress, AnyUrl]
    # this allows for values such as localhost or 127.0.0.1
    # or 192.168.xxx.xxx or <server_name>.nmr.mgh.harvard.edu


class dataflowArgs(BaseModel):
    reserve_threshold_bytes: PositiveInt
    suitable_volumes: List[DirectoryPath]
    delete_threshold: float = Field(ge=0, le=1)


def get_config_file_location(environment_variable_name: str) -> os.PathLike:
    '''Reads an environment variable and returns location of corresponding config files'''
    config_file_location = os.environ.get(environment_variable_name)
    if config_file_location is None:
        raise Exception(f'got None when retreiving {environment_variable_name} environment variable')
    return config_file_location


def validate_config_fpath(config_fpath: os.PathLike) -> None:
    '''validates that any path-to-file exists,
        meant to ensure config file exists'''
    if not os.path.exists(config_fpath):
        raise Exception(f'config file at {config_fpath} does not exist')


def get_config_file_path(config_file_name: str, config_type: Literal['TERRA', 'EXPLORER', 'USERPASS']) -> os.PathLike:
    '''Returns validated config file path defined by environment variable
    '''

    # Dictionary to match confir_type string literal to
    # the corresponding environment variable name
    environ_dict = dict()
    environ_dict['TERRA'] = 'TERRA_CONFIG_LOC'
    environ_dict['EXPLORER'] = 'EXPLORER_CONFIG_LOC'
    environ_dict['USERPASS'] = 'EXPLORER_USER_PASS'

    env_var_name = environ_dict[config_type]
    try:
        config_file_location = get_config_file_location(env_var_name)
    except Exception as e:
        raise Exception(f'Could not get config file location from environment variable {env_var_name}')

    config_fpath = os.path.join(config_file_location, config_file_name)
    validate_config_fpath(config_fpath)
    return config_fpath


def load_yaml_file_into_dict(yaml_file_path):
    with open(yaml_file_path) as yaml_file:
        param_dict = yaml.load(yaml_file, yaml.FullLoader)
        return param_dict


def read_db_secrets(config_fpath: Optional[str] = None):
    '''
    Returns a dictionary of database credentials with keys:
    'database' for the name of the postgres database
    'user' for the pg username
    'password' for the pg user password
    'host' for database host

    The credential file is assumed to be at a location which
    is defined in the TERRA_CONFIG_LOC environment variable

    '''

    # First get config file path
    if config_fpath is None:
        config_file_name = '.db.secrets.yml'
        config_fpath = get_config_file_path(config_file_name, 'EXPLORER')

    # Then load the config file
    db_config_dict = load_yaml_file_into_dict(config_fpath)

    # Next get the name of server on which the script is being run
    try:
        hostname = socket.gethostname().split('.')[0]
    except Exception as e:
        raise Exception('Something went wrong in trying to get hostname')

    if hostname not in db_config_dict.keys():
        text = f'Hostname {hostname} does not match any key in config file {config_fpath}'
        raise RuntimeError (text)
    
    # Now extract db args specific to host
    host_specific_db_args = db_config_dict[hostname]

    # Then apply pydantic validation to db args values
    db_args = databaseArgs(**host_specific_db_args)

    credentials = {'database': db_args.db_name,
                   'user': db_args.db_user,
                   'password': db_args.password,
                   'host': db_args.host}
    return credentials


def read_dataflow_configs(config_fpath: Optional[str] = None):
    '''
    Returns a dictionary of dataflow parameters with keys:
    'reserve_threshold_bytes' 
    'suitable_volumes' 
    'delete_threshold'
    See config yaml for context on these keys
    '''

    if config_fpath is None:
        config_file_name = 'dataflow_config.yml'
        config_fpath = get_config_file_path(config_file_name, 'TERRA')

    dataflow_config_dict = load_yaml_file_into_dict(config_fpath)
    dataflow_args = dataflowArgs(**dataflow_config_dict)
    # this validates dataflow config values

    dataflow_configs = {'reserve_threshold_bytes': dataflow_args.reserve_threshold_bytes,
                        'suitable_volumes': dataflow_args.suitable_volumes,
                        'delete_threshold': dataflow_args.delete_threshold}
    
    # check that all volumes in suitable volumes are actually suitable
    for volume in dataflow_configs['suitable_volumes']:
        if not os.path.exists(volume):
            raise Exception(f'volume path {volume} does not exist')

    return dataflow_configs


def get_ssh_args():
    '''
    SSH arguments don't change, hence are hardcoded here and
    returned as is
    '''
    ssh_args = dict(
            ssh_address_or_host='neurodoor.nmr.mgh.harvard.edu',
            ssh_pkey='/space/neurobooth/1/applications/config/id_rsa', # this is user sp1022's id_rsa
            remote_bind_address=('192.168.100.1', 5432),
            local_bind_address=('localhost', 6543)
    )
    return ssh_args


def get_user_pass_pairs():
    '''
    Read username-password pairs from yml file
    '''

    config_file_name = '.explorer_user-passwords.yml'
    config_fpath = get_config_file_path(config_file_name, 'USERPASS')
    user_pass_dict = load_yaml_file_into_dict(config_fpath)

    return user_pass_dict



if __name__ == '__main__':
    '''Run this script standalone to test config reading or config value validation
       Pass config file paths as command line arguments'''
    import sys
    
    db_args = read_db_secrets(config_fpath=sys.argv[1])
    dataflow_args = read_dataflow_configs(config_fpath=sys.argv[2])
    ssh_args = get_ssh_args()
    user_pass_pairs = get_user_pass_pairs()

    print()
    for ky in db_args.keys():
        print(ky, db_args[ky])
    print()
    for ky in dataflow_args.keys():
        print(ky, dataflow_args[ky])
    print()
    for ky in ssh_args.keys():
        print(ky, ssh_args[ky])
    print()
    for ky in user_pass_pairs.keys():
        print(ky, user_pass_pairs[ky])

