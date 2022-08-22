
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']      # one input: L

    if model == 'plain':
        from models.model_plain import ModelPlain as M
    elif model == 'plain_mhcnn':  # two inputs: L, C
        from models.model_plain_mhcnn import ModelPlain as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)
    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
