import torch
import torch.nn as nn
from networks.mh_mirnet import MH_MIRNet

mhcnn_path = '../pretrained_models/mhcnn.pth'
mirnet_path = '../pretrained_models/mirnet.pth'


mhcnn_weights = torch.load(mhcnn_path,map_location='cpu')['state_dict'],
mirnet_weights = torch.load(mirnet_path,map_location='cpu')['state_dict']
mh_mirnet = MH_MIRNet()
mh_mirnet = nn.DataParallel(mh_mirnet)

# for name,layer in mh_mirnet._modules.items():
#     for k,v in layer._modules.items():
#         if k == 'mirnet':
#             v = nn.DataParallel(v)
#             v.load_state_dict(mirnet_weights)
temp = mh_mirnet.state_dict()
for k,v in mh_mirnet.named_parameters():
    # for k1, v1 in mhcnn_weights[0].items():
    #     if k == k1:
    #         temp[k] = torch.Tensor(v1)
            # print(k)
    if 'mirnet' in k:
        for k2, v2 in mirnet_weights.items():
            if k2[6:] == k[13:]:
                temp[k] = torch.Tensor(v2)
temp.update(temp)
mh_mirnet.load_state_dict(temp)
print(mh_mirnet)
torch.save(mh_mirnet.state_dict(),'mh_mirnet.pth')
# print(mh_mirnet)
