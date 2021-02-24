import torch
import torch.nn as nn
import numpy as np
import sys
from sdf import SDF
import pdb

class SDFLoss(nn.Module):

    def __init__(self, right_faces, left_faces, grid_size=32, robustifier=None):
        super(SDFLoss, self).__init__()
        self.sdf = SDF()
        self.register_buffer('right_face', torch.tensor(right_faces.astype(np.int32)))
        self.register_buffer('left_face', torch.tensor(left_faces.astype(np.int32)))
        self.grid_size = grid_size
        self.robustifier = robustifier
        # print("robustifier", self.robustifier)

    @torch.no_grad()
    def get_bounding_boxes(self, vertices):
        bs = vertices.shape[0]
        boxes = torch.zeros(bs, 2, 2, 3, device=vertices.device)
        boxes[:, :, 0, :] = vertices.min(dim=2)[0]
        boxes[:, :, 1, :] = vertices.max(dim=2)[0]
        return boxes

    def forward(self, vertices, scale_factor=0.2):
        bs = vertices.shape[0]
        num_hand = 2
        boxes = self.get_bounding_boxes(vertices) # (bs, 2, 2, 3)
        loss = torch.tensor(0., device=vertices.device)

        # re-scale the input vertices
        boxes_center = boxes.mean(dim=2).unsqueeze(dim=2) # (bs, 2, 1, 3)
        boxes_scale = (1+scale_factor) * 0.5*(boxes[:,:,1] - boxes[:,:,0]).max(dim=-1)[0][:, :, None,None] # (bs, 2, 1, 1)

        with torch.no_grad():
            vertices_centered = vertices - boxes_center
            vertices_centered_scaled = vertices_centered / boxes_scale
            assert(vertices_centered_scaled.min() >= -1)
            assert(vertices_centered_scaled.max() <= 1)
            right_verts = vertices_centered_scaled[:, 0].contiguous()
            left_verts = vertices_centered_scaled[:, 1].contiguous()
            right_phi = self.sdf(self.right_face, right_verts, self.grid_size)
            left_phi = self.sdf(self.left_face, left_verts, self.grid_size)
            assert(right_phi.min() >= 0) # (bs, 32, 32, 32)
            assert(left_phi.min() >= 0) # (bs, 32, 32, 32)
        
        # concat left & right phi
        # be aware of the order, input vertices the order is right, left
        phi = [right_phi, left_phi]
        losses = list()

        for i in [0, 1]:
            # print('vertices', vertices.size())
            vertices_local = (vertices[:, i:i+1] - boxes_center[:, 1-i].unsqueeze(dim=1)) / boxes_scale[:, i].unsqueeze(dim=1)
            # print('vertices_local', vertices_local.size())
            vertices_grid = vertices_local.view(bs,-1,1,1,3)
            # print('vertices_grid', vertices_grid.size())
            # Sample from the phi grid
            phi_val = nn.functional.grid_sample(
                phi[1-i].unsqueeze(dim=1), vertices_grid, align_corners=True).view(bs, -1)
            # print(phi[1-i].unsqueeze(dim=1).size())
            # print(vertices_grid.size())
            # print(phi_val.size())
            # sys.exit(0)
            cur_loss = phi_val

            # robustifier, cur_loss = cur_loss^2 / (cur_loss^2 + robust^2)
            if self.robustifier:
                frac = (cur_loss / self.robustifier) ** 2
                cur_loss = frac / (frac + 1)
            cur_loss = cur_loss / num_hand ** 2
            losses.append(cur_loss)
        loss = (losses[0] + losses[1])
        loss = loss.sum(dim=1)
        return loss
