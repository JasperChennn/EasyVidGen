import os
import json

cakeify_data_path = 'tmp/Open-VFX_Cake-ify/train/Cake-ify'
cakeify_data_out = 'examples/cakeify/data.json'

data = []
for file_name in os.listdir(cakeify_data_path):
    if file_name.endswith('.mp4'):
        data.append({
            'video_path': file_name,
            'caption': "Cut it open like you're cutting a cake."
        })

with open(cakeify_data_out, 'w') as f:
    json.dump(data, f, indent=4)