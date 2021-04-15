import argparse

parser = argparse.ArgumentParser('Global flags parser')
parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
DEBUG = parser.parse_known_args()[0].debug
