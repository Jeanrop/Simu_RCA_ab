import argparse
import json

Path = r'config.json'

with open(Path) as json_file:
        config = json.load(json_file)

def example_function(para1, paara2):
    t = 1
    print(f'param1: {para1}')
    print(f'param2: {paara2}')

example_function(**config)

config['a','b']

