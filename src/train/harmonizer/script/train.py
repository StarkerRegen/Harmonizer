import os
import sys
import collections

sys.path.append('..')
import torchtask

import proxy

config = collections.OrderedDict(
    [
        ('exp_id', os.path.basename(__file__).split(".")[0]),

        ('trainer', 'harmonizer_trainer'),

        # arguments - Task Proxy
        ('short_ep', False),

        # arguments - exp
        # ('resume', './result/train/2022-11-06_16-53-13/ckpt/checkpoint_60.ckpt'),
        ('validation', False),

        ('out_path', '../../../../autodl-tmp/result'),

        ('visualize', False),

        # ('visualize', True),

        ('debug', False),

        ('val_freq', 1),
        ('log_freq', 100),
        ('visual_freq', 100),
        ('checkpoint_freq', 1),

        # arguments - dataset / dataloader
        ('im_size', 256),
        ('num_workers', 4),
        ('ignore_additional', False),

        ('trainset', {
            # 'harmonizer_iharmony4': [
            #     './dataset/iHarmony4/HAdobe5k/train',
            #     './dataset/iHarmony4/HCOCO/train',
            #     './dataset/iHarmony4/Hday2night/train',
            #     './dataset/iHarmony4/HFlickr/train',
            # ]
            'hyoutube': [
                '../../../../autodl-tmp/train_list.txt'
            ]
        }),
        ('additionalset', {
            # 'original_iharmony4': [
            #     './dataset/iHarmony4/HAdobe5k/train',
            #     './dataset/iHarmony4/HCOCO/train',
            #     './dataset/iHarmony4/Hday2night/train',
            #     './dataset/iHarmony4/HFlickr/train',
            # ],
            'hyoutube': [
                '../../../../autodl-tmp/train_list.txt'
            ]
        }),
        ('valset', {
            # 'original_iharmony4': [
            #     './dataset/iHarmony4/HAdobe5k/test',
            #     './dataset/iHarmony4/HCOCO/test',
            #     './dataset/iHarmony4/Hday2night/test',
            #     './dataset/iHarmony4/HFlickr/test',
            # ]
            'hyoutube': [
                '../../../../autodl-tmp/test_list.txt'
            ]
        }),

        # arguments - task specific components
        ('models', {'model': 'harmonizer'}),
        ('optimizers', {'model': 'adam'}),
        ('lrers', {'model': 'multisteplr'}),
        ('criterions', {'model': 'harmonizer_loss'}),

        # arguments - task specific optimizer / lr scheduler
        ('lr', 0.0001),

        ('milestones', [25, 50]),
        ('gamma', 0.1),

        # arguments - training details
        ('epochs', 60),
        ('batch_size', 16),
        ('additional_batch_size', 8),
    ]
)

if __name__ == '__main__':
    torchtask.run_script(config, proxy, proxy.HarmonizerProxy)
