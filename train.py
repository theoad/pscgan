from runner import runner
import argparse
import utils.argparse_utils as argparse_utils

if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = argparse_utils.add_base_args(parent=parent_parser, test_flag=False)
    parent_parser = argparse_utils.add_train_args(parent_parser)
    parent_parser = argparse_utils.add_optional_args(parent_parser)
    args = parent_parser.parse_args()
    runner(parent_parser, args, test_flag=False)
