


def get_accuracy(type = None, args = None):
    if type == 'dice':
        return DiceAcc(type)
    if type == None :
        return None
    else:
        raise NotImplementedError



class DiceAcc():
    def __init__(self, name):
        self.name = name

    def compute(self, outputs, targets):
        outputs = (outputs >= 0.5).float()
        return (outputs == targets).float().mean().item()
