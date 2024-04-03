import numpy as np

class Binseg():
    def __init__(self, cost, min_size=1, jump=1):
        self.cost = cost  # Cost function object
        self.min_size = min_size  # Minimum segment size
        self.jump = jump  # Step size for evaluating potential changepoints

    def _seg(self, n_bkps=None):
        bkps = [self.n_samples]  # Initialize with entire signal length
        stop = False
        while not stop:
            stop = True
            new_bkps = [self.single_bkp(start, end) for start, end in self.pairwise([0] + bkps)]
            bkp, gain = max(new_bkps, key=lambda x: x[1])
            if bkp is None:
                break
            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False
            if not stop:
                bkps.append(bkp)
            bkps.sort()
        partition = {(start, end): self.cost.error(start, end) for start, end in self.pairwise([0] + bkps)}
        return partition


    def single_bkp(self, start, end):
        segment_cost = self.cost.error(start, end)
        if np.isinf(segment_cost) and segment_cost < 0:  # if cost is -inf
            return None, 0
        gain_list = list()
        for bkp in range(start, end, self.jump):
            if bkp - start >= self.min_size and end - bkp >= self.min_size:
                gain = (segment_cost - self.cost.error(start, bkp) - self.cost.error(bkp, end))
                gain_list.append((gain, bkp))
        try:
            gain, bkp = max(gain_list)
        except ValueError:  # if empty sub_sampling
            return None, 0
        return bkp, gain

    def fit(self, signal):
        self.signal = signal.reshape(-1, 1)
        self.n_samples, _ = self.signal.shape
        self.cost.fit(signal)  # Train the cost function on the signal (if applicable)
        return self

    def predict(self, n_bkps=None):
        partition = self._seg(n_bkps=n_bkps)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    # Helper function for pairwise iteration
    def pairwise(self,iterable):
        a, b = iter(iterable), iter(iterable)
        next(b)
        return zip(a, b)
