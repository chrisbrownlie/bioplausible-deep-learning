from typing import Any, Callable, Optional, Tuple, NamedTuple, Optional
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension

cpp_extension = CppExtension(
    name="norse_op",
    sources=["norse/csrc/op.cpp", "norse/csrc/super.cpp"],
    extra_compile_args=["-O3"],
)

@torch.jit.script
def heaviside(data):
    r"""
    A `heaviside step function <https://en.wikipedia.org/wiki/Heaviside_step_function>`_
    that truncates numbers <= 0 to 0 and everything else to 1.
    .. math::
        H[n]=\begin{cases} 0, & n <= 0 \\ 1, & n \gt 0 \end{cases}
    """
    return torch.gt(data, torch.as_tensor(0.0)).to(data.dtype)  # pragma: no cover

class SuperSpike(torch.autograd.Function):
    r"""SuperSpike surrogate gradient as described in Section 3.3.2 of
    F. Zenke, S. Ganguli, **"SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks"**,
    Neural Computation 30, 1514–1541 (2018),
    `doi:10.1162/neco_a_01086 <https://www.mitpressjournals.org/doi/full/10.1162/neco_a_01086>`_
    """

    @staticmethod
    @torch.jit.ignore
    def forward(ctx, input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.alpha = alpha
        return heaviside(input_tensor)

    @staticmethod
    @torch.jit.ignore
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        grad = grad_input / (alpha * torch.abs(inp) + 1.0).pow(
            2
        )  # section 3.3.2 (beta -> alpha)
        return grad, None


@torch.jit.ignore
def super_fn(x: torch.Tensor, alpha: float = 100.0) -> torch.Tensor:
    return SuperSpike.apply(x, alpha)

superspike_fn = super_fn

def threshold(x: torch.Tensor, method: str, alpha: float) -> torch.Tensor:
    if method == "heaviside":
        return heaviside(x)
    elif method == "super":
        return superspike_fn(x, torch.as_tensor(alpha))
    else:
        raise ValueError(
            f"Attempted to apply threshold function {method}, but no such "
            + "function exist. We currently support heaviside, super, "
            + "tanh, tent, circ, and heavi_erfc."
        )

class LIFParameters(NamedTuple):
    """Parametrization of a LIF neuron
    Parameters:
        tau_syn_inv (torch.Tensor): inverse synaptic time
                                    constant (:math:`1/\\tau_\\text{syn}`) in 1/ms
        tau_mem_inv (torch.Tensor): inverse membrane time
                                    constant (:math:`1/\\tau_\\text{mem}`) in 1/ms
        v_leak (torch.Tensor): leak potential in mV
        v_th (torch.Tensor): threshold potential in mV
        v_reset (torch.Tensor): reset potential in mV
        method (str): method to determine the spike threshold
                      (relevant for surrogate gradients)
        alpha (float): hyper parameter to use in surrogate gradient computation
    """

    tau_syn_inv: torch.Tensor = torch.as_tensor(1.0 / 5e-3)
    tau_mem_inv: torch.Tensor = torch.as_tensor(1.0 / 1e-2)
    v_leak: torch.Tensor = torch.as_tensor(0.0)
    v_th: torch.Tensor = torch.as_tensor(1.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    method: str = "super"
    alpha: float = torch.as_tensor(100.0)


default_bio_parameters = LIFParameters(
    tau_syn_inv=torch.as_tensor(1 / 0.5),
    tau_mem_inv=torch.as_tensor(1 / 20.0),
    v_leak=torch.as_tensor(-65.0),
    v_th=torch.as_tensor(-50.0),
    v_reset=torch.as_tensor(-65.0),
)


class LIFFeedForwardState(NamedTuple):
    """State of a feed forward LIF neuron
    Parameters:
        v (torch.Tensor): membrane potential
        i (torch.Tensor): synaptic input current
    """

    v: torch.Tensor
    i: torch.Tensor

class LIFParametersJIT(NamedTuple):
    """Parametrization of a LIF neuron
    Parameters:
        tau_syn_inv (torch.Tensor): inverse synaptic time
                                    constant (:math:`1/\\tau_\\text{syn}`) in 1/ms
        tau_mem_inv (torch.Tensor): inverse membrane time
                                    constant (:math:`1/\\tau_\\text{mem}`) in 1/ms
        v_leak (torch.Tensor): leak potential in mV
        v_th (torch.Tensor): threshold potential in mV
        v_reset (torch.Tensor): reset potential in mV
        method (str): method to determine the spike threshold
                      (relevant for surrogate gradients)
        alpha (torch.Tensor): hyper parameter to use in surrogate gradient computation
    """

    tau_syn_inv: torch.Tensor
    tau_mem_inv: torch.Tensor
    v_leak: torch.Tensor
    v_th: torch.Tensor
    v_reset: torch.Tensor
    method: str
    alpha: torch.Tensor

FeedforwardActivation = Callable[
    # Input        State         Parameters       dt
    [torch.Tensor, torch.Tensor, torch.nn.Module, float],
    Tuple[torch.Tensor, torch.Tensor],
]
@torch.jit.script
def _lif_feed_forward_step_jit(
    input_tensor: torch.Tensor,
    state: LIFFeedForwardState,
    p: LIFParametersJIT,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFFeedForwardState]:  # pragma: no cover
    # compute voltage updates
    dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + state.i)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di

    # compute new spikes
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset
    # compute current jumps
    i_new = i_decayed + input_tensor

    return z_new, LIFFeedForwardState(v=v_new, i=i_new)


def lif_feed_forward_step(
    input_tensor: torch.Tensor,
    state: Optional[LIFFeedForwardState],
    p: LIFParameters = LIFParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFFeedForwardState]:
    r"""Computes a single euler-integration step for a lif neuron-model.
    It takes as input the input current as generated by an arbitrary torch
    module or function. More specifically it implements one integration
    step of the following ODE
    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}
    together with the jump condition
    .. math::
        z = \Theta(v - v_{\text{th}})
    and transition equations
    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + i_{\text{in}}
        \end{align*}
    where :math:`i_{\text{in}}` is meant to be the result of applying an
    arbitrary pytorch module (such as a convolution) to input spikes.
    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        state (LIFFeedForwardState): current state of the LIF neuron
        p (LIFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    try:
        z, v, i = lif_super_feed_forward_step(input_tensor, state, p, dt)
        return z, LIFFeedForwardState(v=v, i=i)
    except NameError:  # pragma: no cover
        pass
    jit_params = LIFParametersJIT(
        tau_syn_inv=p.tau_syn_inv,
        tau_mem_inv=p.tau_mem_inv,
        v_leak=p.v_leak,
        v_th=p.v_th,
        v_reset=p.v_reset,
        method=p.method,
        alpha=torch.as_tensor(p.alpha),
    )
    return _lif_feed_forward_step_jit(input_tensor, state=state, p=jit_params, dt=dt)

class SNNCell(torch.nn.Module):
    """
    Initializes a feedforward neuron cell *without* time.
    Parameters:
        activation (FeedforwardActivation): The activation function accepting an input tensor, state
            tensor, and parameters module, and returning a tuple of (output spikes, state).
        state_fallback (Callable[[torch.Tensor], Any]): A function that can return a
            default state with the correct dimensions, in case no state is provided in the
            forward pass.
        p (torch.nn.Module): The neuron parameters as a torch Module, which allows the module
            to configure neuron parameters as optimizable.
        dt (float): Time step to use in integration. Defaults to 0.001.
    """

    def __init__(
        self,
        activation: FeedforwardActivation,
        state_fallback: Callable[[torch.Tensor], torch.Tensor],
        p: Any,
        dt: float = 0.001,
    ):
        super().__init__()
        self.activation = activation
        self.state_fallback = state_fallback
        self.p = p
        self.dt = dt

    def extra_repr(self) -> str:
        return f"p={self.p}, dt={self.dt}"

    def forward(self, input_tensor: torch.Tensor, state: Optional[Any] = None):
        state = state if state is not None else self.state_fallback(input_tensor)
        return self.activation(input_tensor, state, self.p, self.dt)

class LIFCell(SNNCell):
    """Module that computes a single euler-integration step of a
    leaky integrate-and-fire (LIF) neuron-model *without* recurrence and *without* time.
    More specifically it implements one integration step
    of the following ODE
    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \\end{align*}
    together with the jump condition
    .. math::
        z = \\Theta(v - v_{\\text{th}})
    and transition equations
    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}}
        \\end{align*}
    Example:
        >>> data = torch.zeros(5, 2) # 5 batches, 2 neurons
        >>> l = LIFCell(2, 4)
        >>> l(data) # Returns tuple of (Tensor(5, 4), LIFState)
    Arguments:
        p (LIFParameters): Parameters of the LIF neuron model.
        sparse (bool): Whether to apply sparse activation functions (True) or not (False). Defaults to False.
        dt (float): Time step to use. Defaults to 0.001.
    """

    def __init__(self, p: LIFParameters = LIFParameters(), sparse=False, **kwargs):
        super().__init__(
            activation=(
                lif_feed_forward_step_sparse if sparse else lif_feed_forward_step
            ),
            state_fallback=self.initial_state,
            p=p,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> LIFFeedForwardState:
        state = LIFFeedForwardState(
            v=torch.full(
                input_tensor.shape,
                self.p.v_leak.detach(),
                device=input_tensor.device,
                dtype=torch.float32,
            ),
            i=torch.zeros(
                *input_tensor.shape,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state