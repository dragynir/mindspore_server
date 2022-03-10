from mindspore import Tensor, export, load
from mindspore import nn
import numpy as np
import mindspore.context as context
import numpy as np

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Predictor(object):

    def __init__(self):
        self.graph = load("/content/drive/MyDrive/MINDSPORE/checkpoints/eik_model_m.mindir")
        self.net = nn.GraphCell(self.graph)

    def predict(self, x):
        input_tensor = Tensor(x).astype(np.float32)
        output = self.net(input_tensor)
        return np.array(output)