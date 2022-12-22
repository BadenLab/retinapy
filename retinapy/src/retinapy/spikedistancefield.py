import numpy as np
import math
from typing import Optional, OrderedDict, Tuple, Iterable
from collections import namedtuple
from collections import defaultdict
import torch
import torch.nn.functional as F
import einops
import retinapy.spikeprediction as sp

"""
Notes:

    - Having a single array that uses +ve and -ve values to signal the direction
      towards the closest spike would be an implementation more close to the
      signed distance function approach. However, there is an issue where a
      timestep might be in the middle of two very distant spikes. In this case,
      we do not want there to be a large penalty for predicting say -50 instead
      of +50, where this vacillation might appear if there is uncertainty
      as to whether the timestep is slightly closer to the earlier or later 
      spike.
    - To avoid the above problem, have the model output both a forward distance
      *and* a reverse distance.
    - [A while later...] The bi-directional approach produces a loss function 
      that value jumps that are too large and abrupt. So, going back to the 
      original idea. A second way to address the concern above is to make it a
      non-signed distance. That's what I'll try next (and probably should have 
      tried first).
"""


def distance_field(spikes: np.ndarray, default_distance: float):
    """
    Calculates the distance field of a spike train.

    Args:
        spikes: a 1D array where a 1 represents a timestep where a spike
            occurred, and 1 where a spike did not occur.
        default_distance: the value to initialize each element of the
            distance field. This functions as a maximum distance. Notably, if
            there are no spikes in the spikes array, then all elements of the
            distance field will be set to this value.
    """
    dist_field = np.full_like(spikes, default_distance, float)
    spike_indicies = (spikes == 1).nonzero()[0]
    all_indicies = np.arange(len(spikes))
    for s in spike_indicies:
        dist_field = np.minimum(dist_field, np.abs(all_indicies - s))
    return dist_field


def spike_interval(spikes: np.ndarray, default_count: int):
    """Count the timesteps between spikes"""
    count_field = np.full_like(spikes, default_count, int)
    spike_indicies = (spikes == 1).nonzero()[0]
    count_field[spike_indicies] = 0
    counts = np.diff(spike_indicies) - 1
    for idx in range(len(counts)):
        count_field[spike_indicies[idx] + 1 : spike_indicies[idx + 1]] = counts[
            idx
        ]
    return count_field


def distance_field2(spikes, default_dist):
    """An alternative (non-vector) distance field implementation.

    Not used at the moment. Leaving it here for reference.
    """
    dist = [
        default_dist,
    ] * len(spikes)
    spike_indicies = [idx for idx, v in enumerate(spikes) if v == 1]

    def _dfs(idx, cur_dist):
        if dist[idx] <= cur_dist:
            return
        dist[idx] = cur_dist
        if idx > 0:
            _dfs(idx - 1, cur_dist + 1)
        if idx < len(spikes) - 1:
            _dfs(idx + 1, cur_dist + 1)

    for s in spike_indicies:
        _dfs(s, cur_dist=0)
    return dist


# Below are various attempts at inference. Very WIP.


def quick_inference(dist, target_interval, threshold=0.1):
    dist = dist[:, target_interval[0] : target_interval[1]]
    num_spikes = torch.sum(dist < threshold, dim=1)
    return num_spikes


def quick_inference2(dist, target_interval, threshold=0.1):
    dist = dist[:, target_interval[0] : target_interval[1]]
    kernel = torch.FloatTensor(
        [[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]]
    ).to(dist.device)
    smoothed = torch.squeeze(F.conv1d(torch.unsqueeze(dist, 1), kernel))
    below_threshold = (smoothed < threshold).float()
    # below_threshold = below_threshold[:,::3]
    transitions = torch.sum((torch.diff(below_threshold, 1, dim=1) > 0), dim=1)
    return transitions


def mle_inference_via_dp(
    dist: torch.Tensor,
    lhs_spike,
    rhs_spike,
    spike_pad,
    max_clamp=None,
    max_num_spikes=100,
    resolution=1,
):
    if len(dist.shape) != 1:
        raise ValueError("Batching isn't supported (yet)")
    init_a = max(0, lhs_spike + spike_pad + 1)
    init_b = min(len(dist) - 1, rhs_spike - spike_pad - 1)
    max_n = int(math.ceil((init_b - init_a) / (spike_pad + 1)))
    max_n = min(max_n, max_num_spikes)
    _len = len(dist)
    device = dist.device

    # If a-1 and b+1 are the indicies of two spikes, what is the energy
    # contributed by the elements in (a,b)?
    memo = {}  # (a,b, num_allowed) -> ('energy')

    zero_spike_memo = torch.zeros(_len, _len, device=device)
    for i in range(_len):
        for j in range(i, _len):
            d_after = torch.arange(
                j - i + 1, dtype=torch.float32, device=device
            )
            d_before = torch.flip(
                torch.arange(j - i + 1, dtype=torch.float32, device=device),
                dims=(0,),
            )
            # The endpoints need special treatment.
            if i == 0:
                d_after += -lhs_spike
            if j == 0:
                d_before += rhs_spike
            d_min = torch.clamp(torch.minimum(d_after, d_before), max=max_clamp)
            energy = torch.sum(torch.abs(dist[i : j + 1] - d_min))
            zero_spike_memo[i, j] = energy

    global_best_energy = math.inf
    low_energy_positions = dist < 20

    def _dfs(
        a, b, energy_so_far, num_allowed_spikes
    ) -> Tuple[float, Tuple[int, ...]]:
        nonlocal global_best_energy
        if a >= b:
            return 0, ()
        if (a, b, num_allowed_spikes) in memo:
            return memo[(a, b, num_allowed_spikes)]
        if energy_so_far > global_best_energy:
            return math.inf, ()
        no_spike_energy = zero_spike_memo[a, b]
        best_energy = no_spike_energy
        best_seq = ()
        if not num_allowed_spikes:
            return best_energy, best_seq
        for candidate_pos in range(a, b + 1, resolution):
            if not low_energy_positions[candidate_pos]:
                continue
            for num_l_spikes in range(num_allowed_spikes):
                for num_r_spikes in range(num_allowed_spikes - num_l_spikes):
                    lhs_energy, lhs_seq = _dfs(
                        a, candidate_pos - 1, energy_so_far, num_l_spikes
                    )
                    spike_pos_energy = min(dist[candidate_pos], max_clamp)
                    energy = energy_so_far + lhs_energy + spike_pos_energy
                    rhs_energy, rhs_seq = _dfs(
                        candidate_pos + 1, b, energy, num_r_spikes
                    )
                    energy += rhs_energy
                    if energy < best_energy:
                        best_energy = energy
                        best_seq = lhs_seq + (candidate_pos,) + rhs_seq
                        if (best_energy + energy_so_far) < global_best_energy:
                            global_best_energy = best_energy
        memo[(a, b)] = (best_energy, best_seq, num_allowed_spikes)
        return best_energy, best_seq

    e, seq = _dfs(0, _len - 1, energy_so_far=0, num_allowed_spikes=max_n)
    return e, seq


# Below, we use an expectation-maximization algorithm approximation to do the
# inference.
class SpikeLinkedList:
    class Node:
        def __init__(self, pos: int, target_dist: torch.Tensor):
            self.pos = pos
            self.lnode = None
            self.rnode = None
            self.target_dist = target_dist
            self.is_l_shared = None
            self.is_r_shared = None
            self.is_removed = False
            self.device = target_dist.device

        def responsibility(self):
            assert not self.is_removed
            if self.lnode is None:
                left_r = 0
            else:
                num_shared = (self.pos - 1) - self.lnode.pos
                # num_responsible = (num_shared + 1) // 2
                num_responsible = num_shared // 2
                left_r = self.pos - num_responsible
            if self.rnode is None:
                right_r = len(self.target_dist)
            else:
                num_shared = (self.rnode.pos - 1) - self.pos
                # num_responsible = (num_shared + 1) // 2
                num_responsible = num_shared // 2
                right_r_inclusive = self.pos + num_responsible
                right_r = self.pos + num_responsible + 1
            assert left_r < right_r
            assert left_r <= self.pos and right_r > self.pos, (
                "Responsibility "
                "must at least include self.pos (self.pos, left_r, right_r)"
                f" was ({self.pos}, {left_r}, {right_r})"
            )
            return (left_r, right_r)

        def dist_arrs(self, l_idx, r_idx):
            """
            r_idx is exclusive (inclusive + 1)
            """
            assert not self.is_removed
            if l_idx > self.pos or r_idx <= self.pos:
                # For the current use case, we always ask for the full responsibility of
                # a node, and so it's correct to assert this bound. However, the method
                # is more general, at it may be desired later to relax this.
                raise ValueError(
                    "Left and right bounds should contain self.pos. Got "
                    f"({l_idx, r_idx}) with self.pos = {self.pos}"
                )
            dist = torch.zeros(size=[r_idx - l_idx], device=self.device)
            dist_inverted = torch.zeros(
                size=[r_idx - l_idx], device=self.device
            )
            rel_pos = self.pos - l_idx
            # LEFT
            if l_idx < self.pos:
                ldist = (
                    torch.flip(
                        torch.arange(self.pos - l_idx, device=self.device),
                        dims=(0,),
                    )
                    + 1
                )
                dist[0:rel_pos] = ldist
                min_replacement_dist = self.pos - l_idx
                if not self.is_l_shared:
                    min_replacement_dist += 1
                if self.lnode:
                    dist_inverted[0:rel_pos] = (
                        torch.flip(ldist, dims=(0,)) + min_replacement_dist
                    )
                else:
                    # Special treatment for the last node
                    assert self.lnode is None
                    dist_inverted[0:rel_pos] = ldist + (r_idx - self.pos)
            # RIGHT
            if r_idx > self.pos:
                rdist = torch.arange(r_idx - self.pos, device=self.device)
                dist[rel_pos:] = rdist
                min_replacement_dist = (r_idx - 1) - self.pos
                if not self.is_r_shared:
                    min_replacement_dist += 1
                # This depends on whether we are the last spike.
                if self.rnode:
                    dist_inverted[rel_pos:] = (
                        torch.flip(rdist, dims=(0,)) + min_replacement_dist
                    )
                else:
                    # Special treatment for the last node
                    assert self.rnode is None
                    # I guess that dist_inverted isn't a good name anymore....
                    # Give all responsibility to the node on the left.
                    dist_inverted[rel_pos:] = rdist + rel_pos
                min_dist = 0  # 0.25
                dist[rel_pos] = min_dist
            return dist, dist_inverted

        def set_lnode(self, lnode):
            assert not self.is_removed
            self.lnode = lnode
            num_shared = self.pos - lnode.pos - 1
            # is the left-most point shared?
            self.is_l_shared = num_shared % 2

        def set_rnode(self, rnode):
            assert not self.is_removed
            self.rnode = rnode
            num_shared = rnode.pos - self.pos - 1
            # is the right-most point shared?
            self.is_r_shared = num_shared % 2

    def __init__(
        self, target_dist: torch.Tensor, lhs_spike: int, max_dist, output_offset
    ):
        if lhs_spike > 0:
            raise ValueError(f"lhs is expected to be negative. ({lhs_spike})")
        self.lhs_spike = lhs_spike
        self.max_dist = max_dist
        self.output_offset = output_offset
        self.target_dist = target_dist
        self.init_dist = torch.clamp(
            torch.arange(len(target_dist), device=self.target_dist.device)
            - self.lhs_spike,
            max=self.max_dist,
        )

        # Create linked list
        self.first_node = self.Node(pos=0, target_dist=self.target_dist)
        self.last_node = self.first_node
        # init_scores = torch.randperm(len(target_dist))
        init_scores = target_dist
        self.scores = [(init_scores[0], self.first_node)]
        self.num_nodes = 1
        for i in range(1, len(target_dist)):
            node = self.Node(pos=i, target_dist=self.target_dist)
            self.last_node.set_rnode(node)
            node.set_lnode(self.last_node)
            self.last_node = node
            self.num_nodes += 1
            self.scores.append((init_scores[i], node))

    def remove(self, node):
        if node is None:
            raise ValueError("Called remove(None)")
        if node.lnode is None:
            # Removing the first node
            self.first_node = node.rnode
        else:
            node.lnode.rnode = node.rnode
        if node.rnode is None:
            # Removing the last node
            self.last_node = node.lnode
        else:
            node.rnode.lnode = node.lnode
        self.num_nodes -= 1
        node.is_removed = True
        return node.rnode

    def should_remove(self, node):
        assert not node.is_removed
        l_idx, r_idx = node.responsibility()
        curr_dist, alternative_dist = node.dist_arrs(l_idx, r_idx)
        if node.lnode is None:
            # We are the first spike, so need special treatment for alternative.
            assert self.lhs_spike < 0
            alternative_dist = torch.min(
                alternative_dist, self.init_dist[0 : len(alternative_dist)]
            )
            curr_dist[0 : node.pos] = torch.min(
                curr_dist[0 : node.pos], self.init_dist[0 : node.pos]
            )
        curr_loss = self.loss(curr_dist, self.target_dist[l_idx:r_idx])
        alternative_loss = self.loss(
            alternative_dist, self.target_dist[l_idx:r_idx]
        )
        threshold = 0.0 # 0.02  # larger, e.g. 0.10 encouranges less spikes
        res = alternative_loss < curr_loss + threshold
        best_loss = min(alternative_loss, curr_loss)
        return res, best_loss

    def loss(self, pdist, tdist):
        # This next line encourages more spikes. The idea is that the target distance
        # function is really sharp in log space, and the model is probably not so capable
        # of reaching the minimum, so don't penalize it when trying to find the MLE set of
        # spikes.
        # This is balanced with the "threshold" parameter above.
        pdist_nn = torch.clamp(
            sp.DistFieldTrainable_._distfield_to_nn_output(
                pdist, self.output_offset
            ),
            min=-3.5,
        )
        # pdist_nn = self.trainable.distfield_to_nn_output(pdist)
        l1_loss = torch.norm(pdist_nn - tdist)
        return l1_loss

    def reduce(self):
        n = self.first_node
        new_scores = []
        for s, n in sorted(self.scores, key=lambda s: s[0]):
            # for s,n in random.sample(self.scores, len(self.scores)):
            # while n is not None:
            should_remove, loss = self.should_remove(n)
            if should_remove:
                self.remove(n)
            else:
                new_scores.append((-loss, n))
            n = n.rnode
        self.scores = new_scores
        return self.num_nodes

    def spike_indicies(self):
        node = self.first_node
        res = []
        while node is not None:
            res.append(node.pos)
            node = node.rnode
        return res

    def infer_spikes(self):
        num_spikes = self.num_nodes
        halt = False
        while not halt:
            reduced_spikes = self.reduce()
            if reduced_spikes == num_spikes:
                halt = True
            num_spikes = reduced_spikes
            break
        spike_indicies = self.spike_indicies()
        assert len(spike_indicies) == num_spikes
        spikes = torch.zeros_like(self.target_dist)
        spikes[spike_indicies] = 1
        return spikes


# TODO: sort out dependencies. Ideally, no dependency on trainable.
def predict(target_dist, lhs_spike: int, max_dist, output_offset, max_eval_len:
            Optional[int] = None
):
    # TODO: handle lhs_spike
    if len(target_dist.shape) > 1:
        raise ValueError("Batches not supported (no batch dim accepted)")
    ll = SpikeLinkedList(target_dist, lhs_spike, max_dist, output_offset)
    res = ll.infer_spikes()
    if max_eval_len is not None:
        res = res[0:max_eval_len]
    return res

def predict_sum(target_dist, lhs_spike: int, max_dist, output_offset, 
                eval_len : Optional[int] = None):
    mle_spikes = predict(target_dist, lhs_spike, max_dist, output_offset)
    l = eval_len if eval_len is not None else len(mle_spikes)
    res = mle_spikes[0:l].sum()
    return res


###############################################################################
#
# Below is the bi-directional distance field code. Not used.
#
###############################################################################


def bi_distance_field(
    spikes: np.ndarray, default_dist: int, max_dist: Optional[int] = None
):
    """
    Calculates the bi-directional distance field of a spike train.

    Two distances are calculated for each timestep:
        - how far the timestep is since the previous spike ("distance after")
        - how far the timestep is since the next spike ("distance before")

    For example, if a spike happens at timestep t=4, then we have:
        a) dist_after[4] = dist_before[4] = 0
        b) dist_after[5] = 1
        c) dist_before[3] = 1
        d) we can't say anything about dist_after[3] or dist_before[5].


    The distance field is initialized with the default_dist value.
    max_dist sets the maximum the maximum distance value.
    """
    raise NotImplementedError("Failing test. See GitHub issue #2.")
    dist_after = np.full_like(spikes, default_dist, int)
    dist_before = np.full_like(spikes, default_dist, int)
    spike_indicies = (spikes == 1).nonzero()[0]
    r = np.arange(0, len(spikes))
    if spike_indicies is not None:
        # after
        endpoints = list(spike_indicies)
        diff = np.diff(endpoints)
        for idx in range(len(endpoints) - 1):
            dist_after[endpoints[idx] : endpoints[idx + 1]] = r[0 : diff[idx]]
        # before
        endpoints = [-1] + list(spike_indicies)
        diff = np.diff(endpoints)
        for idx in range(len(endpoints) - 1):
            dist_before[endpoints[idx] + 1 : endpoints[idx + 1] + 1] = np.flip(
                r[0 : diff[idx]]
            )
    if max_dist:
        dist_before = np.minimum(dist_before, max_dist)
        dist_after = np.minimum(dist_after, max_dist)
    return dist_before, dist_after


def quick_inference_from_bi_df(
    dist_before, dist_after, target_interval, threshold=0.1
):
    dist_before = dist_before[:, target_interval[0] : target_interval[1]]
    dist_after = dist_after[:, target_interval[0] : target_interval[1]]
    # match = torch.sum((dist_after < threshold) * (dist_before < threshold), dim=1)
    match = torch.sum((dist_after < threshold), dim=1)
    return match


def count_inference_from_bi_df(
    dist_before,
    dist_after,
    lhs_spike,
    rhs_spike,
    spike_pad,
    target_interval,
    max_num_spikes,
):
    raise NotImplementedError("TODO: fix broken tests. GitHub issue #1.")
    if len(dist_before) != len(dist_after):
        raise ValueError("dist_before and dist_after must be the same length.")
    # A 1 element query must have r-l >= 2.
    # l r
    # 012
    # |-|
    num_elements = rhs_spike - lhs_spike - 1
    if num_elements <= 0:
        raise ValueError(
            "Invalid query range ((lhs, rhs) = ({lhs_spike}, {rhs_spike}))."
        )
    if target_interval[0] < 0 or target_interval[1] > len(dist_before):
        raise ValueError(
            f"The query interval ({target_interval}) must be "
            "within the range of the distance field "
            f"(len: {len(dist_before)})."
        )

    init_a = max(0, lhs_spike + spike_pad + 1)
    init_b = min(len(dist_before) - 1, rhs_spike - spike_pad - 1)
    max_n = int(math.ceil((init_b - init_a) / (spike_pad + 1)))
    max_n = min(max_n, max_num_spikes)
    target_energy_by_n = torch.zeros(
        (max_n + 1),
    )  # np.zeros((max_n + 1,))

    memo = {}
    # range_cache = np.arange(0, rhs_spike - lhs_spike)
    range_cache = torch.arange(
        0,
        rhs_spike - lhs_spike,
        device=dist_before.device,
    )
    MemoKey = namedtuple("MemoKey", ["l_spike", "r_spike", "a", "b", "n"])

    def energy_to_prob(e):
        b = 1  # / len(dist_before)
        return math.exp(b * -e)

    def _energy(l_spike, r_spike, a, b, n):
        print(f"l_spike: {l_spike}, r_spike: {r_spike}, a: {a}, b: {b}, n: {n}")
        assert l_spike < a <= b < r_spike
        assert n >= 0
        key = MemoKey(l_spike, r_spike, a, b, n)
        if key in memo:
            # print('hit')
            return memo[key]
        ans = []
        if n == 0:
            dist_after_measured = range_cache[(a - l_spike) : (b - l_spike + 1)]
            dist_before_measured = torch.flip(  # np.flip(
                range_cache[(r_spike - b) : (r_spike - a) + 1], dims=(0,)
            )
            assert len(dist_after_measured) == len(dist_before_measured)
            # dist_after_measured = torch.clip(dist_after_measured, 0, max_dist)
            # dist_before_measured = torch.clip(dist_before_measured, 0, max_dist)
            energy_forward = torch.sum(  # np.sum(
                (dist_before[a : b + 1] - dist_before_measured) ** 2
            )
            energy_backward = torch.sum(  # np.sum(
                (dist_after[a : b + 1] - dist_after_measured) ** 2
            )
            ans.append(energy_forward + energy_backward)
            # energy_to_prob(energy_forward + energy_backward)
        else:
            assert n > 0
            start = max(a, l_spike + spike_pad + 1)
            # end *includes* the last valid index.
            # with two spikes needing to be inserted, we need at least
            # 2*(spike_pad+1) from the end.
            end = min(b, r_spike - n * (spike_pad + 1))
            if start > end:
                return 0
            for x in range(start, end + 1):
                # energy_at_x = min(dist_after[x], max_dist) + min(
                #    dist_before[x], max_dist
                # )
                energy_at_x = dist_after[x] + dist_before[x]
                # prob_at_x = energy_to_prob(energy_at_x)
                # prob_rhs = prob_lhs = 1
                energy_rhs = energy_lhs = 0
                if x > a:
                    energy_lhs = _energy(l_spike, x, a, x - 1, n=0)
                if x < b:
                    energy_rhs = _energy(x, r_spike, x + 1, b, n=n - 1)
                if x == b and n > 1:
                    # Zero liklihood if there are too many n left to fit on rhs.
                    energy_rhs = 0
                event_energy = energy_at_x + energy_lhs + energy_rhs
                ans.append(event_energy)
        total_energy = torch.logsumexp(torch.Tensor(ans), dim=0)
        # Save result
        memo[key] = total_energy
        # Check against target interval.
        if a == target_interval[0] and b == target_interval[1] - 1:
            target_energy_by_n[n] += ans
        return ans

    last_p = 0
    for i in range(max_n + 1):
        p = _energy(lhs_spike, rhs_spike, init_a, init_b, i)
        print(f"i: {i}:\t p: {p}")
        if p < last_p:
            break
        last_p = p
    return torch.argmax(target_energy_by_n)  # np.argmax(target_energy_by_n)


def count_inference_from_bi_df2(
    dist_before,
    dist_after,
    lhs_spike,
    rhs_spike,
    spike_pad,
    target_interval,
    max_num_spikes=100,
    resolution=1,
):
    """
    Target_interval will be (start, end), where we will consider spikes to
    be allowed at any index from start to (end-1) inclusive (ie right open
    interval [start, end) ).
    """
    raise NotImplementedError("TODO: fix broken tests. GitHub issue #1.")
    if dist_before.shape != dist_after.shape:
        raise ValueError("dist_before and dist_after must be the same shape.")
    if torch.any(lhs_spike >= 0):
        raise ValueError(
            "The left hand side spike should be -ve steps " "from interval."
        )
    device = dist_before.device
    _len = dist_before.shape[-1]
    batch_len = dist_before.shape[0]
    max_n = int(math.ceil(_len / (spike_pad + 1)))
    # Memo of energy of intervals between two spikes.
    after_memo = torch.zeros((batch_len, _len, _len), device=device)
    before_memo = torch.zeros((batch_len, _len, _len), device=device)
    for i in range(_len):
        for j in range(i, _len):
            # i,j represent the start and end of the interval *between* the
            # spikes. So the spikes happen at i-1 and j+1. This means that
            # i=j is valid.
            # E.g. if (i,j) == (2,4), then result is dist((1,2,3,4), dist_after)
            # TODO: cache these distances/ranges?
            # Need to make the dtype a float, as cdist requires float.
            d_after = torch.arange(
                j - i + 1, dtype=torch.float32, device=device
            ).repeat(batch_len, 1)
            d_before = torch.flip(
                torch.arange(j - i + 1, dtype=torch.float32, device=device),
                dims=(0,),
            ).repeat(batch_len, 1)
            # The endpoints need special treatment.
            if i == 0:
                d_after += -lhs_spike
            if j == 0:
                d_before += rhs_spike
            energy_after = (dist_after[:, i : j + 1] - d_after).pow(2).sum(1)
            energy_before = (dist_before[:, i : j + 1] - d_before).pow(2).sum(1)
            after_memo[:, i, j] = torch.squeeze(energy_after)
            before_memo[:, i, j] = torch.squeeze(energy_before)

    # At this point, it's only valid to query the memo.
    # Make a list of every possible spike sequence.
    spike_sequences = []

    def _dfs(a, spikes: Tuple[int, ...]):
        if a >= _len or len(spikes) >= max_num_spikes:
            # The end of the tree has been reached.
            spike_sequences.append(spikes)
            return
        # Either there is no spike here:
        _dfs(a + 1 + (resolution - 1), spikes)
        # Or there is as spike.
        next_a = a + spike_pad + 1
        _dfs(next_a, spikes + (a,))

    _dfs(0, spikes=())

    # Now get the energy of every possible spike sequence. Bin by the number
    # of spikes in the target interval, as that's why we are here in the
    # first place.
    by_n = defaultdict(list)
    for ss in spike_sequences:
        interval_count = 0
        for i in range(len(ss)):
            # TODO: check, are we off by one here?
            if ss[i] < target_interval[0]:
                # Interval hasn't started yet.
                continue
            # TODO: check, are we off by one here?
            elif ss[i] >= target_interval[1]:
                # Gone past interval.
                break
            interval_count += 1
        # The energy is the sum of:
        #   - the energy all steps where there is a spike
        #   - the energy of all spike-empty intervals thus formed.
        # Before and after energies will be accumulated here.
        # The energies are added, which corresponds to multiplication of
        # probabilities. We could just have one energy tally, but keeping
        # it as two for debugging.
        after_energy = torch.zeros((batch_len,), device=device)
        before_energy = torch.zeros((batch_len,), device=device)
        prev_spike = -1
        for s_idx in range(len(ss)):
            cur_spike = ss[s_idx]
            if cur_spike > prev_spike + 1:
                # There is a spike-empty interval to consider.
                after_energy += after_memo[:, prev_spike + 1, cur_spike]
                before_energy += after_memo[:, prev_spike + 1, cur_spike]
            # Don't do this:
            #    if ss[s_idx + 1] > cur_spike + 1:
            # Don't double count. The next iteration will get the other side.
            # Add the energy of the spike timestep.
            after_energy += dist_after[:, cur_spike]
            before_energy += dist_before[:, cur_spike]
            prev_spike = cur_spike
        # The last spike potentially has an spike-empty interval on the rhs.
        # This if-block also handles the case when ss = (,)
        if prev_spike < (_len - 1):
            after_energy += after_memo[:, prev_spike + 1, _len - 1]
            before_energy += after_memo[:, prev_spike + 1, _len - 1]
        by_n[interval_count].append((before_energy + after_energy))

    # We are almost there. Just find the n with the maximum energy.
    # The energies for each n are currently saved as a list of energies for
    # independent events, so we need to combine them into one value. This
    # is like adding probabilities, which must be done in exponent space
    # (log space).
    total_by_n = torch.zeros((batch_len, max_n + 1), device=device)
    for n in range(max_n + 1):
        if len(by_n[n]):
            total_by_n[:, n] = torch.logsumexp(
                torch.stack(by_n[n], dim=1), dim=1
            )
        else:
            total_by_n[:, n] = 0
    # The actual answer is obtained by a simple argmax.
    res = torch.argmax(total_by_n, dim=1)
    return res
