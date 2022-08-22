

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()


    if dataset_type in ['mhcnn', 'denoising']:
        from data.dataset_mhcnn import MyDataset as D
    elif dataset_type in ['mhcnn_single', 'denoising']:
        from data.dataset_mhcnn_single import MyDataset as D
    elif dataset_type in ['mhcnn_norot', 'denoising']:
        from data.dataset_mhcnn_norot import MyDataset as D
    elif dataset_type in ['mhcnn_rot90270', 'denoising']:
        from data.dataset_mhcnn_rot90270 import MyDataset as D
    elif dataset_type in ['mhcnn_rot180270', 'denoising']:
        from data.dataset_mhcnn_rot90270 import MyDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
