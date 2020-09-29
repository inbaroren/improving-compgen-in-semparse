from overrides import overrides
from typing import List, Iterator

from allennlp.common import JsonDict
from allennlp.predictors.predictor import Predictor, sanitize
from allennlp.data.instance import Instance


@Predictor.register('text2sql_grammar')
class TextToSqlGrammarPredictor(Predictor):
    """Predictor wrapper for the Text2SqlParser"""
    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        instance = self._dataset_reader.read_json_dict(json_dict)
        output = self.predict_instance(instance)
        output['input_tokens'] = [t.text for t in instance.fields['tokens']]
        return {'instance': output}

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.read_json_dict(json_dict)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        outputs['input_tokens'] = [t.text for t in instance.fields['tokens']]
        return sanitize(outputs)


