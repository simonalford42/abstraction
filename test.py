import torch

def logsumexp(tensor, dim=-1, mask=None):
    """taken from https://github.com/pytorch/pytorch/issues/32097"""
    if mask is None:
        mask = torch.ones_like(tensor)
    else:
        assert mask.shape == tensor.shape, 'The factors tensor should have the same shape as the original'
    a = torch.cat([torch.max(tensor, dim, keepdim=True) for _ in range(tensor.shape[dim])], dim)
    return a + torch.sum((tensor - a).exp()*mask, dim).log()


def logaddexp(tensor, other, mask=None):
    if mask is None:
        mask = torch.tensor([1, 1])
    else:
        assert mask.shape == (2, ), 'invalid mask provided'
    
    a = torch.max(tensor, other)
    return a + ((tensor - a).exp()*mask[0] + (other - a).exp()*mask[1]).log()


def logsubexp(tensor, other):
    return logaddexp(tensor, other, mask=torch.tensor([1, -1]))

def log_practice():
    a = torch.tensor([0.5, 0.5])
    b = torch.tensor([0.1, 0.9])

    c = torch.tensor([0.75, 0.25])
    a2, b2, c2 = map(torch.log, (a, b, c))

    # print( (a + b).log())
    # print( logaddexp(a2, b2))
    # print( (a - b).log())
    # print( logaddexp(a2, b2, torch.tensor([1., -1.])))

    macro_stops = a * b
    print(f"macro_stops: {macro_stops}")
    still_there = a - macro_stops
    print(f"still_there: {still_there}")
    total_new = sum(macro_stops)
    print(f"total_new: {total_new}")
    redist = total_new * c
    print(f"redist: {redist}")
    new_macro = still_there + redist
    print(f"new_macro: {new_macro}")

    assert torch.isclose(torch.logsumexp(a2, dim=0), torch.tensor(0.))
    macro_stops2 = a2 + b2
    print(f"macro_stops2: {macro_stops2}")
    print(torch.isclose(macro_stops2, torch.log(macro_stops)))
    still_there2 = logsubexp(a2, macro_stops2)
    print(f"still_there2: {still_there2}")
    print(torch.isclose(still_there2, torch.log(still_there)))
    total_new2 = torch.logsumexp(still_there2, dim=0)
    print(torch.isclose(total_new2, torch.log(total_new)))
    redist2 = total_new2 + c2
    print(torch.isclose(redist2, torch.log(redist)))
    new_macro2 = torch.logaddexp(still_there2, redist2)
    print(torch.isclose(new_macro2, torch.log(new_macro)))
    print(torch.logsumexp(new_macro2, dim=0))
    print(new_macro2.exp())
    print(torch.isclose(torch.sum(new_macro2.exp()), torch.tensor(1.)))
    print(torch.isclose(torch.logsumexp(new_macro2, dim=0), torch.tensor(0.)))
