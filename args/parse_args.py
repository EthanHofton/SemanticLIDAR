import argparse
import os
import yaml
from args.train_parser import add_train_parser
from args.visualize_parser import add_visualize_parser
from args.make_dataset_parser import add_make_dataset_parser

def parse_args():
    parser = argparse.ArgumentParser(prog='SemanticKITTI solution',
                                     description='Training, Testing and util tools for SemanticKITTI solution')
    subparsers = parser.add_subparsers(title='subcommands',
                                       description='The subcommands of the program',
                                       help='additional help',
                                       required=True)
    # add visualize command parser
    add_visualize_parser(subparsers)
    # add train command parser
    add_train_parser(subparsers)
    # add make dataset command parser
    add_make_dataset_parser(subparsers)

    return parser.parse_args(), parser.print_help

