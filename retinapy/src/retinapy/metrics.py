
class Meter:
    """An online sum and avarage meter."""
    def __init__(self, name=None):
        self.reset()
        self.name = name

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val 
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        res = f"{self.name} " if self.name else 'Meter'
        res += f"(average -- total) : {self.avg:.4f} -- ({self.sum:.4f})"
        return res


