from torch.utils.data import DataLoader
from trajectories import TrajectoryDataset, seq_collate

def data_loader(args, path, min_agent=1):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        frame_skip=args.frame_skip,
        num_feature=args.dset_feature,
        min_agent=min_agent
    )

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        drop_last=True,
        collate_fn=seq_collate
    )

    return dset, loader
