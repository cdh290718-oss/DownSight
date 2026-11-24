"""Model-relative elbow joint coordinates, in a right-handed neutral-template frame."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

LOW = np.array([-15., -20., -105.])
HIGH = np.array([160., 20., 105.])
CHAINS = [(16,18,20), (17,19,21)]


def unit(v):
    return v / np.linalg.norm(v)


class Elbow:
    def __init__(self, joints, side):
        s,e,w = CHAINS[side]
        self.upper = unit(joints[e]-joints[s])
        self.distal = unit(joints[w]-joints[e])
        lateral = joints[16]-joints[17]
        up = joints[12]-joints[0]
        self.front = unit(np.cross(lateral,up))
        hinge = unit(np.cross(self.distal,self.front))
        self.axes = np.stack([self.distal,np.cross(hinge,self.distal),hinge],axis=1)
        self.sign = np.array([1.,1.,1.]) if side==0 else np.array([1.,-1.,-1.])
        assert np.allclose(self.axes.T@self.axes,np.eye(3)) and np.linalg.det(self.axes)>.99999

    def decompose(self, rotation):
        return Rot.from_matrix(self.axes.T@rotation@self.axes).as_euler('ZYX',degrees=True)*self.sign

    def compose(self, angles):
        return self.axes@Rot.from_euler('ZYX',angles*self.sign,degrees=True).as_matrix()@self.axes.T

    def check(self):
        # Pure flexion sends the wrist anteriorly; pure twist leaves its ray fixed.
        assert np.dot(self.compose([90,0,0])@self.distal,self.front)>.85
        assert np.allclose(self.compose([0,0,80])@self.distal,self.distal)
        examples=np.array([[0,0,0],[90,0,0],[-10,0,0],[0,16,0],[0,0,80],[130,-18,-90]])
        assert np.allclose(self.decompose(self.compose(examples)),examples)
        return dict(axes=self.axes.tolist(), upper=self.upper.tolist(), anterior=self.front.tolist(),
                    signed_tests_deg=examples.tolist())


def legal(a):
    return np.all((a>=LOW-1e-4)&(a<=HIGH+1e-4),axis=-1)
