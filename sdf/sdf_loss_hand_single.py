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
        num_people = vertices.shape[0]
        boxes = torch.zeros(num_people, 2, 3, device=vertices.device)
        for i in range(num_people):
            boxes[i, 0, :] = vertices[i].min(dim=0)[0]
            boxes[i, 1, :] = vertices[i].max(dim=0)[0]
        return boxes

    def forward(self, vertices, scale_factor=0.2):
        num_hand = vertices.shape[0]
        boxes = self.get_bounding_boxes(vertices)
        loss = torch.tensor(0., device=vertices.device)

        # re-scale the input vertices
        boxes_center = boxes.mean(dim=1).unsqueeze(dim=1) # (2, 1, 3)
        boxes_scale = (1+scale_factor) * 0.5*(boxes[:,1] - boxes[:,0]).max(dim=-1)[0][:,None,None] # (2, 1, 1)

        with torch.no_grad():
            vertices_centered = vertices - boxes_center
            vertices_centered_scaled = vertices_centered / boxes_scale
            assert(vertices_centered_scaled.min() >= -1)
            assert(vertices_centered_scaled.max() <= 1)
            right_phi = self.sdf(self.right_face, vertices_centered_scaled[0:1], self.grid_size)
            left_phi = self.sdf(self.left_face, vertices_centered_scaled[1:2], self.grid_size)
            assert(right_phi.min() >= 0)
            assert(left_phi.min() >= 0)
        
        # concat left & right phi
        # be aware of the order, input vertices the order is right, left
        phi = torch.cat([right_phi, left_phi], dim=0)

        for i in [0, 1]:
            vertices_local = (vertices[i:i+1] - boxes_center[1-i].unsqueeze(dim=0)) / boxes_scale[i].unsqueeze(dim=0)
            vertices_grid = vertices_local.view(1,-1,1,1,3)
            # Sample from the phi grid
            phi_val = nn.functional.grid_sample(
                phi[1-i][None, None], vertices_grid, align_corners=True).view(1, -1)
            # print(phi.size(), vertices_local.size(), vertices_grid.size(), phi_val.size())
            # sys.exit(0)
            cur_loss = phi_val
            # robustifier, cur_loss = cur_loss^2 / (cur_loss^2 + robust^2)
            if self.robustifier:
                frac = (cur_loss / self.robustifier) ** 2
                cur_loss = frac / (frac + 1)
                # print("here", cur_loss)
            loss += cur_loss.sum() / num_hand ** 2
        return loss
