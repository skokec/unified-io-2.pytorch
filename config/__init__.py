import argparse
import ast
import importlib
import copy
import sys

def get_cmd_args():
    class ParseConfigCMDArgs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for value in values:
                value_eq = value.split('=')
                key, value = value_eq[0], value_eq[1:]
                getattr(namespace, self.dest)[key] = "=".join(value)

    # get any config values from CMD arg that override the config file ones
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('-c', '--configs', nargs='*', action=ParseConfigCMDArgs, default=dict())

    cmd_args = parser.parse_args()

    return cmd_args

def import_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    return config

def get_config_args(merge_from_cmd_args=True):
    # parse command line arguments
    cmd_args = get_cmd_args()

    print(f'Loading config from {cmd_args.config_path}')  
    config = import_config(cmd_args.config_path)
    
    args = config.get_args()

    ########################################################
    # Merge from CMD args if any
    if merge_from_cmd_args:
        args = merge_args_from_config(args, cmd_args)

    ########################################################
    # updated any string with format substitution based on other other arg values
    return replace_args(args)

def merge_args_from_config(args, cmd_args):
    def set_config_val_recursive(config, k, v):
            k0 = k[0]
            if isinstance(config, list):
                k0 = int(k0)
            if isinstance(k, list) and len(k) > 1:
                config[k0] = set_config_val_recursive(config[k0], k[1:], v)
            else:
                config[k0] = v
            return config

    for k,v in cmd_args.configs.items():
        try:
            v = ast.literal_eval(v)
        except:
            print('WARNING: cfg %s=%s interpreting as string' % (k,v))
        args = set_config_val_recursive(args, k.split("."), v)
        print("Overriding config with cmd %s=%s" % (k,v))

    return args
    
def replace_args(_args, full_args=None):
    if full_args is None:
        full_args = copy.deepcopy(_args)

    if isinstance(_args, str):
        _args = _args.format(args=full_args)
    elif isinstance(_args, dict):
        _args = {k: replace_args(a, full_args) for k,a in _args.items()}
    elif isinstance(_args, list) or isinstance(_args, tuple):
        _args = [replace_args(a,full_args) for a in _args]
    return _args
