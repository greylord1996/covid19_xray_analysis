from src.classification.torch_dataset import CovidDataset
import torch


def prepare_train_valid_dataloader(
        df, fold, target_cols,
        train_transform, valid_transform,
        train_batch_size, valid_batch_size,
        num_workers
):
    df_train_this = df[df['fold'] != fold]
    df_valid_this = df[df['fold'] == fold]

    dataset_train = CovidDataset(df_train_this, target_cols, 'train', transform=train_transform)
    dataset_valid = CovidDataset(df_valid_this, target_cols, 'train', transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=train_batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=valid_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, valid_loader
