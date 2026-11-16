import torch.nn as nn


class DOANet(nn.Module):
    def __init__(self, input_dim=32):
        super().__init__()

        # 8 hidden layers as in the paper (Linear -> BatchNorm1d -> ReLU)
        self.layers = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Layer 2
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Layer 3
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Layer 4
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # Layer 5
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # Layer 6
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # Layer 7
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            # Layer 8
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # 5 output heads in parallel
        self.class_head = nn.Linear(256, 4)  # predicts K in {1, 2, 3, 4}
        self.head_1 = nn.Linear(256, 1)      # predicts 1 angle if K=1
        self.head_2 = nn.Linear(256, 2)      # predicts 2 angles if K=2
        self.head_3 = nn.Linear(256, 3)      # predicts 3 angles if K=3
        self.head_4 = nn.Linear(256, 4)      # predicts 4 angles if K=4

    def forward(self, x):
        x = self.layers(x)

        class_logits = self.class_head(x)
        doa_1 = self.head_1(x)
        doa_2 = self.head_2(x)
        doa_3 = self.head_3(x)
        doa_4 = self.head_4(x)

        return class_logits, [doa_1, doa_2, doa_3, doa_4]


def export_model_png(model, input_dim=32, filename="doa_net"):
    from torchview import draw_graph

    graph = draw_graph(model, input_size=(2, input_dim), save_graph=True, filename=filename)
    return graph.visual_graph