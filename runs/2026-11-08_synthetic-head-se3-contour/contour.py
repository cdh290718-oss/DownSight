"""Visible exterior contours, with full-mesh depth testing and frozen matches.

Segmentation edges are only used where the person meets background. Clothing
labels near detected arms are excluded; no cut/semantic border is invented.
Matches refresh between joint optimizer steps, not between parameter blocks.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numba import njit
from common import positive_points


@njit(cache=False)
def raster_faces(vertices, faces, uv, rays, max_edge):
    h, w = rays.shape[:2]
    depth = np.full((h, w), np.inf, np.float32)
    face_map = np.full((h, w), -1, np.int32)
    bary = np.zeros((h, w, 3), np.float32)
    for i in range(len(faces)):
        a, b, c = vertices[faces[i, 0]], vertices[faces[i, 1]], vertices[faces[i, 2]]
        if min(a[2], b[2], c[2]) <= 1e-5:
            continue
        p, q, r = uv[faces[i, 0]], uv[faces[i, 1]], uv[faces[i, 2]]
        if not (np.isfinite(p).all() and np.isfinite(q).all() and np.isfinite(r).all()):
            continue
        if max(((p-q)**2).sum(), ((p-r)**2).sum(), ((q-r)**2).sum()) > max_edge**2:
            continue
        x0 = max(0, int(np.floor(min(p[0], q[0], r[0]))) - 1)
        x1 = min(w-1, int(np.ceil(max(p[0], q[0], r[0]))) + 1)
        y0 = max(0, int(np.floor(min(p[1], q[1], r[1]))) - 1)
        y1 = min(h-1, int(np.ceil(max(p[1], q[1], r[1]))) + 1)
        e0, e1 = b-a, c-a
        n = np.cross(e0, e1)
        ndot = (n*a).sum()
        d00, d01, d11 = (e0*e0).sum(), (e0*e1).sum(), (e1*e1).sum()
        denom = d00*d11-d01*d01
        if abs(denom) < 1e-16:
            continue
        for y in range(y0, y1+1):
            for x in range(x0, x1+1):
                ray = rays[y, x]
                nd = (n*ray).sum()
                if abs(nd) < 1e-12 or not np.isfinite(nd):
                    continue
                t = ndot/nd
                if t <= 0 or t >= depth[y, x]:
                    continue
                d = t*ray-a
                d20, d21 = (d*e0).sum(), (d*e1).sum()
                u = (d11*d20-d01*d21)/denom
                v = (d00*d21-d01*d20)/denom
                if min(u, v, 1-u-v) < -1e-5:
                    continue
                face_map[y, x] = i
                depth[y, x] = t
                bary[y, x, 0], bary[y, x, 1], bary[y, x, 2] = 1-u-v, u, v
    return face_map, bary


def subsample(points, maximum):
    return np.linspace(0, len(points)-1, min(maximum, len(points))).astype(int)


def make_targets(labels, probabilities, lookup, cfg):
    h, w = labels.shape
    arm = np.zeros_like(labels, np.uint8)
    radius = cfg['garment_arm_exclusion_radius_px']
    for side in ['left', 'right']:
        for a, b in [('shoulder', 'elbow'), ('elbow', 'wrist')]:
            qa, qb = lookup.get(side+'_'+a), lookup.get(side+'_'+b)
            if qa is None or qb is None or min(qa['score'], qb['score']) < .3:
                continue
            pa, pb = tuple(np.rint([qa['x'], qa['y']]).astype(int)), tuple(np.rint([qb['x'], qb['y']]).astype(int))
            cv2.line(arm, pa, pb, 1, 2*radius)
            cv2.circle(arm, pa, radius, 1, -1)
            cv2.circle(arm, pb, radius, 1, -1)
    bg_neighbor = cv2.dilate((labels == 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    yy, xx = np.indices((h, w))
    b = cfg['contour_border_px']
    interior = (xx >= b) & (xx < w-b) & (yy >= b) & (yy < h-b)
    result = {}
    for group, ids in [('torso', cfg['torso_classes']), ('legs', cfg['leg_classes'])]:
        membership = np.isin(labels, ids)
        confident = probabilities[ids].sum(0) >= cfg['contour_min_probability']
        eligible = membership & confident & interior
        if group == 'torso':
            eligible &= arm == 0
        boundary = eligible & bg_neighbor
        y, x = np.where(boundary)
        points = np.column_stack([x, y]).astype(np.float32)
        active = int(eligible.sum()) >= cfg['contour_min_pixels'] and len(points) >= cfg['contour_min_boundary_points']
        selected = points[subsample(points, cfg['contour_max_points'])] if len(points) else points
        result[group] = dict(active=bool(active), classes=ids, class_pixels=int(membership.sum()),
                             eligible_pixels=int(eligible.sum()), boundary_points=int(len(points)),
                             target=selected, reason='supported' if active else 'insufficient semantic area or exterior contour',
                             arm_exclusion=arm if group == 'torso' else np.zeros_like(arm))
    return result


class Contours:
    def __init__(self, model, camera, targets, cfg):
        self.model, self.camera, self.targets, self.cfg = model, camera, targets, cfg
        self.faces = np.asarray(model.face, np.int32)
        dominant = model.layer['neutral'].lbs_weights.detach().numpy().argmax(1)
        part = np.zeros(len(dominant), np.int32)
        part[np.isin(dominant, [0, 3, 6, 9, 13, 14])] = 1
        part[np.isin(dominant, [1, 2, 4, 5, 7, 8])] = 2
        # Require a majority of triangle vertices to belong to a part.
        self.face_part = np.zeros(len(self.faces), np.int32)
        for p in [1, 2]:
            self.face_part[(part[self.faces] == p).sum(1) >= 2] = p
        self.scale = cfg['raster_scale']
        self.width, self.height = int(1280*self.scale), int(720*self.scale)
        y, x = np.indices((self.height, self.width), dtype=float)
        pixels = np.stack([x.ravel(), y.ravel()], 1)/self.scale
        self.rays = camera.camera2world_ray(pixels).reshape(self.height, self.width, 3).astype(np.float32)
        self.matches = {}
        self.last_diagnostics = {}

    def refresh(self, mesh):
        vertices = mesh.detach().numpy() if isinstance(mesh, torch.Tensor) else np.asarray(mesh)
        uv = self.camera.world2camera(vertices.astype(float))*self.scale
        face_map, bary_map = raster_faces(vertices.astype(np.float32), self.faces, uv.astype(np.float32), self.rays, 180.*self.scale)
        visible = (face_map >= 0).astype(np.uint8)
        edge = (visible > 0) & (cv2.erode(visible, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) == 0)
        yy, xx = np.indices(visible.shape)
        border = max(1, int(np.ceil(self.cfg['contour_border_px']*self.scale)))
        edge &= (xx >= border) & (xx < self.width-border) & (yy >= border) & (yy < self.height-border)
        self.matches = {}
        diag = {}
        for name, number in [('torso', 1), ('legs', 2)]:
            target = self.targets[name]
            if not target['active']:
                diag[name] = dict(active=False, reason=target['reason'])
                continue
            take = edge & (self.face_part[np.maximum(face_map, 0)] == number)
            y, x = np.where(take)
            original_xy = np.column_stack([x, y])/self.scale
            if len(x):
                good = target['arm_exclusion'][original_xy[:, 1].astype(int), original_xy[:, 0].astype(int)] == 0
                y, x = y[good], x[good]
            if len(x) == 0:
                diag[name] = dict(active=True, valid=False, reason='no model exterior samples')
                continue
            indices = subsample(x, self.cfg['contour_max_points'])
            y, x = y[indices], x[indices]
            vertex_ids = self.faces[face_map[y, x]]
            bary = bary_map[y, x]
            xyz = (vertices[vertex_ids] * bary[:, :, None]).sum(1)
            pred = self.camera.world2camera(xyz.astype(float)).astype(np.float32)
            goals = target['target']
            distances = np.linalg.norm(pred[:, None] - goals[None], axis=2)
            model_nearest = distances.argmin(1)
            target_nearest = distances.argmin(0)
            self.matches[name] = dict(ids=torch.tensor(vertex_ids, dtype=torch.long), bary=torch.tensor(bary),
                                      model_goals=torch.tensor(goals[model_nearest]), goals=torch.tensor(goals),
                                      target_nearest=torch.tensor(target_nearest, dtype=torch.long))
            diag[name] = dict(active=True, valid=True, model_samples=len(pred), target_samples=len(goals),
                              model_to_target_mean_px=float(distances.min(1).mean()),
                              target_to_model_mean_px=float(distances.min(0).mean()))
        self.last_diagnostics = diag
        return diag

    def loss(self, mesh):
        total = mesh.sum()*0
        terms = {}
        for name, target in self.targets.items():
            if not target['active']:
                continue
            if name not in self.matches:
                # Candidate is marked infeasible by valid(); no fake zero fit.
                total = total + .1
                terms[name] = .1
                continue
            m = self.matches[name]
            xyz = (mesh[m['ids']] * m['bary'][:, :, None]).sum(1)
            uv = self.camera.world2camera_pytorch(positive_points(xyz, self.cfg['minimum_observed_joint_z_m']))
            a = (uv-m['model_goals']).norm(dim=1)/self.cfg['contour_residual_scale_px']
            b = (uv[m['target_nearest']]-m['goals']).norm(dim=1)/self.cfg['contour_residual_scale_px']
            value = .5*(F.smooth_l1_loss(a, torch.zeros_like(a), beta=self.cfg['contour_huber_beta']) +
                        F.smooth_l1_loss(b, torch.zeros_like(b), beta=self.cfg['contour_huber_beta']))
            value = self.cfg['contour_weight_per_group']*value
            total = total+value
            terms[name] = float(value.detach())
        return total, terms

    def valid(self):
        return all(not target['active'] or name in self.matches for name, target in self.targets.items())
