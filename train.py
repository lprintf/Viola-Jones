from modules.cascader import Cascader
from modules.dataloader import Dataloader


class Trainer:
    def __init__(
        self,
        dataloader: Dataloader,
        cascader: Cascader,
    ) -> None:
        self.dataloader = dataloader
        self.cascader = cascader

    def train(self) -> None:
        self.cascader.train(self.dataloader)

    def save(self, path: str):
        self.cascader.save(path)


if __name__ == "__main__":
    trainer = Trainer(
        Dataloader(
            "/root/playground/Viola-Jones/datasets/set0"
        ),
        Cascader(),
    )
    trainer.train()
