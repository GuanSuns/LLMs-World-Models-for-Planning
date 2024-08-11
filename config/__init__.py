import os


def get_openai_key():
    import json

    dir_name = os.path.dirname(os.path.abspath(__file__))
    config_fname = os.path.join(dir_name, 'my_api_config.json')
    if not os.path.isfile(config_fname):
        config_fname = os.path.join(dir_name, 'api_config.json')

    with open(config_fname) as f:
        d = json.load(f)

    api_key = d['openai']['api_key']
    if api_key.strip() == 'put your OpenAI api key here':
        raise ValueError('Please put your OpenAI api key in config/api_config.json')
    return api_key
