from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch

from comp_probing.models.probing import RandomProbe
from comp_probing.datasets.guesswhat import DialogueFeaturesDataset


class RandomProbeTest(ModelTestCase):
    def setUp(self):
        super(RandomProbeTest, self).setUp()
        self.set_up_model('comp_probing/tests/resources/random_probe.json',
                          '../glaleti/data/gw_train.jsonl'
                          )

    def test_forward(self):
        batch_dialogues = Batch(self.instances)

        res = self.model.forward(**batch_dialogues.as_tensor_dict(batch_dialogues.get_padding_lengths()))

        print(res)
