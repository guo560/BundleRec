import pytorch_lightning as pl
from argparse import ArgumentParser

from pytorch_lightning import Trainer

from model.BundleBST import BundleBST


def main(args):
    pl.seed_everything(args.seed)
    model = BundleBST(args)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.max_epochs)
    model.to('cuda')
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--data_path', default=r'./data/bundle_time/', type=str)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(max_epochs=5)
    args = parser.parse_args()
    main(args)
