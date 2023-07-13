import torch

class Ambigramflip(torch.nn.Module):
    def __init__(self, ambigram_mode):
        super().__init__()
        self.ambigram_mode = {
            'rot_180': self.flip_180,
            'rot_p90': self.flip_pos_90,
            'rot_n90': self.flip_neg_90,
            'lr': self.flip_lr_mirror,
            'ud': self.flip_ud_mirror,
            'identity': self.identity,
        }
        self.ambigram_flip = self.ambigram_mode[ambigram_mode]

    def forward(self, tensor, reverse=False):
        return self.ambigram_flip(tensor, reverse)

    ## ambigram-list
    # rotate 180 degree
    def flip_180(self, img, reverse=False):
        return torch.flip(img, [2, 3])

    # rotate +90 degree
    def flip_pos_90(self, img, reverse=False):
        if reverse:
            return torch.rot90(img, k=3, dims=(2, 3))
        else:    
            return torch.rot90(img, k=1, dims=(2, 3))

    # rotate -90 degree
    def flip_neg_90(self, img, reverse=False):
        if reverse:
            return torch.rot90(img, k=1, dims=(2, 3))
        else:
            return torch.rot90(img, k=3, dims=(2, 3))

    # flip left-right mirror
    def flip_lr_mirror(self, img, reverse=False):
        return torch.flip(img, (3, ))

    # flip up-down mirror
    def flip_ud_mirror(self, img, reverse=False):
        return torch.flip(img, (2, ))
    
    # identity
    def identity(self, img, reverse=False):
        return img