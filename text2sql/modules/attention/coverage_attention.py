from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.modules.attention.attention import Attention
from allennlp.nn.util import masked_softmax
from allennlp.common.registrable import Registrable


@Attention.register("coverage")
class CoverageAdditiveAttention(torch.nn.Module, Registrable):
    """
    This attention was introduced in <> by See et al.,
    based on <https://arxiv.org/abs/1409.0473> by Bahdanau et al. (additive attention)

    # Parameters
    vector_dim : `int`, required
        The dimension of the vector, `x`, described above.  This is `x.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_dim : `int`, required
        The dimension of the matrix, `y`, described above.  This is `y.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    normalize : `bool`, optional (default : `True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(self, vector_dim: int, matrix_dim: int, normalize: bool = True) -> None:
        super().__init__()
        self._w_matrix = Parameter(torch.Tensor(vector_dim, vector_dim))
        self._u_matrix = Parameter(torch.Tensor(matrix_dim, vector_dim))
        self._c_vector = Parameter(torch.Tensor(1, vector_dim))
        self._v_vector = Parameter(torch.Tensor(vector_dim, 1))
        self._bias = Parameter(torch.Tensor(1, vector_dim))
        self.reset_parameters()
        self._normalize = normalize

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._w_matrix)
        torch.nn.init.xavier_uniform_(self._u_matrix)
        torch.nn.init.xavier_uniform_(self._v_vector)
        torch.nn.init.xavier_uniform_(self._c_vector)
        torch.nn.init.xavier_uniform_(self._bias)

    def forward(self,  # pylint: disable=arguments-differ
                vector: torch.Tensor,
                matrix: torch.Tensor,
                coverage_vector: torch.Tensor,
                matrix_mask: torch.Tensor = None) -> torch.Tensor:
        similarities = self._forward_internal(vector, matrix, coverage_vector)
        if self._normalize:
            return masked_softmax(similarities, matrix_mask)
        else:
            return similarities

    def _forward_internal(self, decoder_hidden_state: torch.Tensor,
                          encoder_outputs: torch.Tensor,
                          coverage_vector: torch.Tensor) -> torch.Tensor:
        intermediate = decoder_hidden_state.matmul(self._w_matrix).unsqueeze(1) + \
                       encoder_outputs.matmul(self._u_matrix) + \
                       coverage_vector.unsqueeze(2).matmul(self._c_vector)
        intermediate = intermediate.add(self._bias.view(1, 1, -1).expand_as(intermediate))
        intermediate = torch.tanh(intermediate)
        return intermediate.matmul(self._v_vector).squeeze(2)