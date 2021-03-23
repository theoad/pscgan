import os
import argparse
import utils.drive_utils as drive_utils


def main(params) -> None:
    service = drive_utils.build_service("ro")
    if params.init:
        return
    if params.all:
        drive_utils.download_all(service, params.out_dir, verbose=not params.silent)
        return
    print(str(params.method))
    drive_utils.download_checkpoint(service, str(params.dataset), params.noise_std,
                                    str(params.method), params.out_dir, verbose=not params.silent)


if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = drive_utils.add_download_args(parent=parent_parser)

    args = parent_parser.parse_args()
    if args.method == drive_utils.PossibleMethods.lag and args.dataset != drive_utils.PossibleDatasets.ffhq:
        parent_parser.error("the LAG method is supported for the FFHQ dataset only.\nAborting")
    if args.init and (os.path.exists('token.json') or os.path.exists('token.pickle')):
        parent_parser.error("token.{pickle, json} detected. No need for additional init.\nAborting")
    main(args)
