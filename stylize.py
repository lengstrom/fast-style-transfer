import vimeo
import os

stylize_script = 'bash ./stylize.sh {} {} {}'

client_id = os.environ.get('CLIENT_ID')
client_secret = os.environ.get('CLIENT_SECRET')
token = os.environ.get('TOKEN')
input_path = os.environ.get('INPUT_PATH')
output_path = os.environ.get('OUTPUT_PATH')
model_path = os.environ.get('MODEL_PATH')

os.system(stylize_script.format(input_path, output_path, model_path))

if all(env_var in os.environ for env_var in ['CLIENT_ID', 'CLIENT_SECRET', 'TOKEN']):
    v = vimeo.VimeoClient(
        token=token,
        key=client_id,
        secret=client_secret
    )
    v.upload(output_file, data={'privacy': {'view':'nobody'}})
