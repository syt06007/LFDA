import argparse
from utils import *
from LFDA import Net

import random
seed = 777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=9, help="angular resolution")
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--testset_dir', type=str, default='./dataset/validation/')
    parser.add_argument('--crop', type=bool, default=True)
    parser.add_argument('--patchsize', type=int, default=32)
    parser.add_argument('--minibatch_test', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='log/Model_LFDA.pth')
    parser.add_argument('--save_path', type=str, default='/Results')

    return parser.parse_args()

'''
Note: 1) We crop LFs into overlapping patches to save the CUDA memory during inference. 
      2) When cropping is performed, the inference time will be longer than the one reported in our paper.
'''

def test(cfg):
    save_path = cfg.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    scene_list = os.listdir(cfg.testset_dir)
    angRes = cfg.angRes

    net = Net(cfg)
    net.to(cfg.device)
    model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
    net.load_state_dict(model['state_dict'])

    for scenes in scene_list:
        print('Working on scene: ' + scenes + '...')
        temp = imageio.imread(cfg.testset_dir + scenes + '/input_Cam000.png')
        lf = np.zeros(shape=(9, 9, temp.shape[0], temp.shape[1], 3), dtype=int)
        for i in range(81):
            temp = imageio.imread(cfg.testset_dir + scenes + '/input_Cam0%.2d.png' % i)
            lf[i // 9, i - 9 * (i // 9), :, :, :] = temp
        lf_gray = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)
        angBegin = (9 - angRes) // 2
        lf_angCrop = lf_gray[angBegin:  angBegin + angRes, angBegin: angBegin + angRes, :, :]

        if cfg.crop == False:
            data = rearrange(lf_angCrop, 'u v h w -> (u h) (v w)')
            data = ToTensor()(data.copy())
            with torch.no_grad():
                disp = net(data.unsqueeze(0).to(cfg.device))
            disp = np.float32(disp[0,0,:,:].data.cpu())

        else:
            patchsize = cfg.patchsize
            stride = patchsize // 2
            data = torch.from_numpy(lf_angCrop)
            sub_lfs = LFdivide(data.unsqueeze(2), patchsize, stride)
            n1, n2, u, v, c, h, w = sub_lfs.shape
            sub_lfs = rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')
            mini_batch = cfg.minibatch_test
            num_inference = (n1 * n2) // mini_batch
            with torch.no_grad():
                out_disp = []
                for idx_inference in range(num_inference):
                    current_lfs = sub_lfs[idx_inference * mini_batch: (idx_inference + 1) * mini_batch, :, :, :, :, :]
                    input_data = rearrange(current_lfs, 'b u v c h w -> b c (u h) (v w)')

                    out = net(input_data.to(cfg.device))
                    out_disp.append(out[0])

                if (n1 * n2) % mini_batch:
                    current_lfs = sub_lfs[(idx_inference + 1) * mini_batch:, :, :, :, :, :]
                    input_data = rearrange(current_lfs, 'b u v c h w -> b c (u h) (v w)')
                    out = net(input_data.to(cfg.device))
                    out_disp.append(out[0])

            out_disps = torch.cat(out_disp, dim=0)
            out_disps = rearrange(out_disps, '(n1 n2) c h w -> n1 n2 c h w', n1=n1, n2=n2)
            disp = LFintegrate(out_disps, patchsize, patchsize // 2)
            disp = disp[0: data.shape[2], 0: data.shape[3]]
            disp = np.float32(disp.data.cpu())

        print('Finished! \n')
        write_pfm(disp, cfg.save_path + '%s.pfm' % (scenes))

    return


if __name__ == '__main__':
    cfg = parse_args()
    test(cfg)
