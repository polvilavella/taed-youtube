# -*- coding: utf-8 -*-

""" Script that creates a virtual environment for the execution of our project. """

import os
import subprocess
import venv
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


ROOT_PATH = os.path.dirname(__file__)


def create_venv(args):
    """Function to create a virtual environment with venv"""

    venv_path = os.path.join(ROOT_PATH, 'venv')
    req_path = os.path.join(ROOT_PATH, 'requirements.txt')
    src_path = os.path.join(ROOT_PATH, 'src')

    # Creating venv
    if args['verbose']:
        print(f'Creating virtual environment in {venv_path}')
    venv.create(venv_path, system_site_packages=False, clear=False, symlinks=False, with_pip=True)

    # Installing requirements packages
    if args['verbose']:
        print(f'Installing packages defined in the requirements file at {req_path}')
    subprocess.call([f'{venv_path}/bin/pip', 'install', '-r', f'{req_path}'])

    # Installing src
    if args['verbose']:
        print(f'Installing src package in the virtual environment {venv_path}')
    subprocess.call([f'{venv_path}/bin/pip', 'install', f'{src_path}'])


def main(args):
    """ Main code: create venv. """
    create_venv(args)


if __name__ == "__main__":

    DESCRIPTION = """"""
    parser = ArgumentParser(description=DESCRIPTION,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Shows configuration process on screen")

    main(vars(parser.parse_args()))
