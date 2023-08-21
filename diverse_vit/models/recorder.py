from torch import nn


def find_modules(nn_module, types):
    modules = list(filter(lambda module: module.__class__.__name__ in types, nn_module.modules()))
    return modules


class Recorder(nn.Module):
    def __init__(self, vit, device=None):
        super().__init__()
        self.vit = vit

        self.data = None
        self.input_recordings = []
        self.output_recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

    def _inputs_hook(self, _, input, output):
        self.input_recordings.append(input[0])

    def _outputs_hook(self, _, input, output):
        self.output_recordings.append(input[0])

    def _register_hook(self):
        modules = find_modules(self.vit, ["Attention"])
        for module in modules:
            handle = module.register_forward_hook(self._inputs_hook)
            self.hooks.append(handle)
            handle = module.proj.register_forward_hook(self._outputs_hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self):
        self.input_recordings = []
        self.output_recordings = []

    def forward(self, x, return_io=False, **kwargs):
        assert not self.ejected, "recorder has been ejected, cannot be used anymore"
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        pred = self.vit(x, **kwargs)

        if return_io:
            extras = {}
            extras["inputs"] = self.input_recordings
            extras["outputs"] = self.output_recordings
            self.input_recordings = []
            self.output_recordings = []
            return pred, extras

        return pred
