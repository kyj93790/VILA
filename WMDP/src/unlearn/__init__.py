from .FT import FT
from .GA import GA, GA_FT, NPO_FT, IHL_FT


def get_unlearn_method(name, *args, **kwargs):
    if name == "FT":
        unlearner = FT(*args, **kwargs)
    elif "IHL" in name:
        print("Using IHL unlearning method")
        unlearner = IHL_FT(*args, **kwargs)
    elif name == "GA":
        unlearner = GA(*args, **kwargs)
    elif "GA+FT" in name:
        print("Using GA+FT unlearning method")
        unlearner = GA_FT(*args, **kwargs)
    elif "NPO" in name:
        print("Using NPO unlearning method")
        unlearner = NPO_FT(if_kl=True, *args, **kwargs)
    elif "ILA" in name:
        unlearner = FT(*args, **kwargs)
    else:
        raise ValueError("No unlearning method")

    return unlearner