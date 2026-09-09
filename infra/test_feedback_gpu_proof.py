import pytest
from infra.feedback_gpu_proof import coefficients, nested_equal

@pytest.mark.parametrize('case', ['lone_illegal', 'equal_illegal'])
def test_illegal_probe_coefficients(case):
    assert coefficients(case,'active_group') == 0
    assert coefficients(case,'fixed_zero') == -1

def test_delayed_positive_both_methods():
    assert coefficients('delayed_positive','active_group') > 0
    assert coefficients('delayed_positive','fixed_zero') == pytest.approx(.9801)

def test_checkpoint_comparator_is_exact():
    import torch
    a={'weights':torch.tensor([1.,2.]),'rng':[torch.tensor([1,2])],'step':4}
    assert nested_equal(a, {'weights':torch.tensor([1.,2.]),'rng':[torch.tensor([1,2])],'step':4})
    assert not nested_equal(a, {'weights':torch.tensor([1.,2.001]),'rng':[torch.tensor([1,2])],'step':4})
    assert not nested_equal(a, {'weights':torch.tensor([1.,2.]),'rng':[torch.tensor([1,2])],'step':3})

def test_actual_adamw_step_on_tiny_policy():
    import torch
    from scripts.train_episode_rl import action_loss_chunk
    for case in ('lone_illegal','equal_illegal','delayed_positive'):
        logits=torch.tensor([.3,-.2],requires_grad=True)
        before=logits.log_softmax(0)[0].detach().item()
        row={'advantage':coefficients(case,'fixed_zero'),'reference_token_logprobs':[before]}
        optimizer=torch.optim.AdamW([logits],lr=1e-6)
        loss,_=action_loss_chunk([logits.log_softmax(0)[:1]],[row],.05,1)
        loss.backward();torch.nn.utils.clip_grad_norm_([logits],1,error_if_nonfinite=True);optimizer.step()
        assert (logits.log_softmax(0)[0].item()-before)*row['advantage'] > 0
