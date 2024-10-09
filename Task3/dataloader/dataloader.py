
class DataLoader():

    """
    Dataloader: a class to load multi-type data from different datasets
    """

    def __init__(self, config : dict):
        self.assertConfig(config)
        self.config = config

    def assertConfig(self, config : dict):
        assert "train_path" in config, "<DataLoader>: train_path is not in config"
        assert "test_path" in config, "<DataLoader>: test_path is not in config"
        assert "val_path" in config, "<DataLoader>: val_path is not in config"

    def loadRaw(self, type : str) -> list :

        """
        loadRaw: load the raw data from the dataset(train, test or val)
        Parameters:
            - type: str, the type of the dataset (train, test, val)
        Returns:
            - data: list, a list of tuple (japanese sentence, english sentence)
        """

        assert type in ["train", "test", "val"], "<DataLoader>: type should be train, test or val"
        path = self.config[type + "_path"]
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # the lines are japanese sentences and english sentences separated by tab
        # we load them as a list of tuple
        data = []
        for line in lines:
            line = line.strip().split("\t")
            data.append(line)

        return data
