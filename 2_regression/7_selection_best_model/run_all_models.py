import os

for filename in os.listdir(r'./'):
    if filename.endswith('.py') and not filename.endswith('run_all_models.py'):
        exec(open(filename).read())