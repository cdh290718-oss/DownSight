"""Check shoulder relaxation, finite gradients and normalization invariance."""
import torch
from fit import shoulder_excess, weighted_mean, positive_projection_points

def main():
    x = torch.tensor([[0., 0.], [18., 24.], [36., 48.]], requires_grad=True)
    result = shoulder_excess(x, 30.)
    torch.testing.assert_close(result, torch.tensor([[0., 0.], [0., 0.], [18., 24.]]))
    result.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.equal(x.grad[0], torch.zeros(2))
    e, w = torch.tensor([1., 4.]), torch.tensor([.3, .8])
    torch.testing.assert_close(weighted_mean(e, w), weighted_mean(e.repeat(2), w.repeat(2)))
    p = torch.tensor([[1., 2., -.1], [1., 2., .1]], requires_grad=True)
    positive_projection_points(p, .001).square().sum().backward()
    assert torch.isfinite(p.grad).all() and p.grad[0, 2] == 0 and p.grad[1, 2] > 0
    print('shoulder tolerance, zero/edge gradient, group normalization, projection guard gradient: PASS')

if __name__ == '__main__':
    main()
