import torch
from intact_tracking.latent_history import LatentHistory
from intact_tracking.memory350_compressed_policy import CompressedConcatMLP


def test_order_padding_reset_and_observation_ownership():
    h=LatentHistory(2,frames=5,latent_dim=1)
    first=h.append(torch.tensor([[1.],[10.]],requires_grad=True))
    for t in range(2,7): latest=h.append(torch.tensor([[float(t)],[float(t*10)]]))
    torch.testing.assert_close(first,torch.tensor([[0.,0.,0.,0.,1.],[0.,0.,0.,0.,10.]]))
    torch.testing.assert_close(latest,torch.tensor([[2.,3.,4.,5.,6.],[20.,30.,40.,50.,60.]]))
    got=h.append(torch.tensor([[7.],[70.]]),reset=torch.tensor([True,False]))
    torch.testing.assert_close(got,torch.tensor([[0.,0.,0.,0.,7.],[30.,40.,50.,60.,70.]]))
    assert not got.requires_grad and h.count.tolist()==[1,5]
    h.clear();assert not h.snapshot().any()


def test_five_frame_heads_have_matching_parameters_and_all_slots_learn():
    a=CompressedConcatMLP(20,1,compression_dims=(32,128),seed=71,latent_dim=320)
    b=CompressedConcatMLP(20,1,compression_dims=(32,128),seed=71,latent_dim=320,fusion='baseline')
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
    x=torch.randn(8,20);z=torch.randn(8,320,requires_grad=True)
    y=a(torch.cat((x,z),-1));torch.testing.assert_close(y,b(x),atol=0,rtol=0)
    y.square().mean().backward()
    assert z.grad is None or not z.grad.any()
    grad=a.head[0].weight.grad[:,128:]
    assert all(grad[:,i*64:(i+1)*64].abs().sum()>0 for i in range(5))
    assert a.head[0].in_features==448


def test_history5_cli_has_user_requested_defaults():
    from intact_tracking.cli.memory350_history5_policy_train import build_parser
    args=build_parser().parse_args(['--fusion','concat','--context-checkpoint','encoder.pt','--output-dir','run'])
    assert args.motion_sampling=='uniform'
    assert args.training_terminations=='original'
    assert args.num_envs==8192 and args.training_ranks==4 and args.until_user_stop
    assert args.policy_precision=='fp32'


def test_minimum_nominal_share_never_rounds_below_ten_percent():
    from intact_tracking.memory350_history5_policy import minimum_nominal_ids
    for worlds,expected in [(8192,820),(512,52),(128,13),(16,2),(10,1)]:
        ids=minimum_nominal_ids(worlds)
        assert len(ids)==expected and len(ids)/worlds>=.1
        assert ids.unique().numel()==expected and ids.min()>=0 and ids.max()<worlds
