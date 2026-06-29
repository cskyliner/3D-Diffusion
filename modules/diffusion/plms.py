class PLMSSampler:
    def __init__(self, diffusion) -> None:
        self.diffusion = diffusion

    def sample(self, *args, **kwargs):
        raise NotImplementedError("PLMS sampler interface reserved for a later migration stage.")
