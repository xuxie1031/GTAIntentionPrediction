from torch.utils.data import DataLoader
from trajectories import TrajectoryDataset, seq_collate

def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
    )

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        collate_fn=seq_collate
    )

    return dset, loader