import argparse
from .sources import DEFAULT_DATA, fetch_all
from .run import run


def main():
    parser=argparse.ArgumentParser(description='Public-data EMD4 evidence workflow (separate from manuscript outputs).')
    parser.add_argument('command', choices=('fetch','run','complete','fetch-complete'), nargs='?', default='complete')
    parser.add_argument('--data-dir',type=str,default=str(DEFAULT_DATA))
    parser.add_argument('--outdir',type=str,default=None)
    parser.add_argument('--offline',action='store_true',help='Require checksum-verified cached inputs; no network.')
    parser.add_argument('--trials',type=int,default=25,help='Synthetic trials per true mechanism (default 25).')
    parser.add_argument('--starts',type=int,default=3,help='Optimisation starting points per fit (default 3).')
    args=parser.parse_args()
    if args.command=='fetch-complete':
        from .extension_sources import fetch_extensions
        fetch_extensions(args.data_dir,offline=args.offline)
    elif args.command=='complete':
        from .complete import run_complete
        run_complete(data_dir=args.data_dir,outdir=args.outdir,offline=args.offline,
                     trials=args.trials,starts=args.starts)
    elif args.command=='fetch':
        fetch_all(args.data_dir,offline=args.offline)
    else:
        run(data_dir=args.data_dir,outdir=args.outdir,offline=args.offline,
            trials=args.trials,starts=args.starts)

if __name__=='__main__':main()
