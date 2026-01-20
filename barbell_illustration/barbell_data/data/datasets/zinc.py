import os.path as osp

from torch_geometric.datasets import ZINC


class ZINCProcessed(ZINC):
    """the purpose of this class is to rewrite where to save the preprocessed dataset"""
    @property
    def processed_dir(self) -> str:
        name = 'subset' if self.subset else 'full'
        if self.pre_transform is not None:
            metadata = f"_{self.pre_transform.__self__}" if self.pre_transform is not None else ""
            name += metadata
        return osp.join(self.root, name, 'processed')


def load_zinc_dataset(root, subset=True, pre_transform=None, transform=None):
    raw_dir = osp.join(root, 'ZINC')
    print(raw_dir)

    train_data = ZINCProcessed(raw_dir, subset=subset, split='train',
                               pre_transform=pre_transform, transform=transform)
    val_data = ZINCProcessed(raw_dir, subset=subset, split='val',
                             pre_transform=pre_transform, transform=transform)
    test_data = ZINCProcessed(raw_dir, subset=subset, split='test',
                              pre_transform=pre_transform, transform=transform)

    if subset:
        assert len(train_data) == 10000
        assert len(val_data) == 1000
        assert len(test_data) == 1000
    else:
        assert len(train_data) == 220011
        assert len(val_data) == 24445
        assert len(test_data) == 5000

    return train_data, val_data, test_data
