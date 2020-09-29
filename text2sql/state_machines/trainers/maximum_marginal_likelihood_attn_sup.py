import logging
from typing import Dict, List, Tuple

import torch

from allennlp.nn import util
from allennlp.state_machines.constrained_beam_search import ConstrainedBeamSearch
from allennlp.state_machines.states import State
from allennlp.state_machines.trainers.decoder_trainer import DecoderTrainer
from allennlp.state_machines.transition_functions import TransitionFunction

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MaximumMarginalLikelihoodAttnSup(DecoderTrainer[Tuple[torch.Tensor, torch.Tensor,torch.Tensor, torch.Tensor]]):
    """
    This class trains a decoder by maximizing the marginal likelihood of the targets.  That is,
    during training, we are given a `set` of acceptable or possible target sequences, and we
    optimize the `sum` of the probability the model assigns to each item in the set.  This allows
    the model to distribute its probability mass over the set however it chooses, without forcing
    `all` of the given target sequences to have high probability.  This is helpful, for example, if
    you have good reason to expect that the correct target sequence is in the set, but aren't sure
    `which` of the sequences is actually correct.

    This implementation of maximum marginal likelihood requires the model you use to be `locally
    normalized`; that is, at each decoding timestep, we assume that the model creates a normalized
    probability distribution over actions.  This assumption is necessary, because we do no explicit
    normalization in our loss function, we just sum the probabilities assigned to all correct
    target sequences, relying on the local normalization at each time step to push probability mass
    from bad actions to good ones.

    Parameters
    ----------
    beam_size : ``int``, optional (default=None)
        We can optionally run a constrained beam search over the provided targets during decoding.
        This narrows the set of transition sequences that are marginalized over in the loss
        function, keeping only the top ``beam_size`` sequences according to the model.  If this is
        ``None``, we will keep all of the provided sequences in the loss computation.
    """
    def __init__(self, beam_size: int = None) -> None:
        self._beam_size = beam_size

    @staticmethod
    def _get_attn_sup_loss(attn_weights: torch.Tensor,
                           alignment_mask: torch.Tensor,
                           alignment_sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute the attention supervision CE loss.
        For each step, take the index of the aligned
        """
        # shape: (batch_size, max_decoding_steps, max_input_seq_length
        attn_weights = attn_weights.float()

        alignment_sequence[alignment_sequence == -1] = 0
        # for each attn_weights[batch_index, step_index, :] I want to choose the index of
        # alignment_sequence[batch_index, step_index]
        return util.sequence_cross_entropy_with_logits(attn_weights, alignment_sequence, alignment_mask)

    def decode(self,
               initial_state: State,
               transition_function: TransitionFunction,
               supervision: Tuple[torch.Tensor, torch.Tensor,torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        targets, target_mask, alignment_sequence, alignment_mask = supervision
        beam_search = ConstrainedBeamSearch(self._beam_size, targets, target_mask)
        finished_states: Dict[int, List[State]] = beam_search.search(initial_state, transition_function)

        loss = 0
        for instance_states in finished_states.values():
            scores = [state.score[0].view(-1) for state in instance_states]
            loss += -util.logsumexp(torch.cat(scores))

        # update the loss with attention supervision
        # batch the attention weights
        max_decoding_steps = targets.size(2)
        max_input_sequence_length = max([instance_states[0].debug_info[0][0]['question_attention'].size(0) for instance_states in finished_states.values()])
        batch_attn_weights = []
        for instance_states in finished_states.values():
            step_attention_weights = [step['question_attention'].unsqueeze(0) for step in instance_states[0].debug_info[0]]
            # shape: (num_decoding_steps, input_sequence_length)
            attention_input_weights = torch.cat(step_attention_weights, 0)
            # padding to (max_decoding_steps, max_input_sequence_length)
            if attention_input_weights.size(0) < max_decoding_steps or attention_input_weights.size(1) < max_input_sequence_length:
                m = torch.nn.ZeroPad2d((0, max_input_sequence_length - attention_input_weights.size(1),
                                        0, max_decoding_steps - attention_input_weights.size(0)))
                attention_input_weights = m(attention_input_weights)
            batch_attn_weights.append(attention_input_weights.unsqueeze(0))
        # shape: (batch_size, num_decoding_steps, max_input_sequence_length)
        attention_input_weights = torch.cat(batch_attn_weights, 0)

        attn_sup_loss = self._get_attn_sup_loss(attention_input_weights,
                                                alignment_mask,
                                                alignment_sequence)

        return {'loss': loss / len(finished_states), 'attn_sup_loss': attn_sup_loss}
