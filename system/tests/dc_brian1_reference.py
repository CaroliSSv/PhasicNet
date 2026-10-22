#dc_brian1_reference - an INDEPENDENT transcription of what the authors' Brian 1 script (peter-u-diehl/stdp-mnist, Diehl&Cook_spiking_MNIST.py)
# does per time step, written from the script + the Brian 1 simulator source -- NOT from diehlcook.py. test_diehl_cook.py runs this and the
# PhasicNet modules on the same inputs and requires the same spikes, V, theta and W. It is deliberately slow and literal: loops in the order
# Brian executes them, Brian's own index arithmetic, Brian's own variable names where it helps.
#
#WHAT BRIAN 1 DOES IN ONE STEP (Network.update_schedule_standard; one clock, dt = 0.5 ms):
#   'groups'            every NeuronGroup.update(): state update (Euler for the neuron groups -- their equations are not linear; for the STDP
#                       trace groups the linear updater: EXACT for pre, and for post1/post2 the numerical fallback (100 Euler sub-steps) because
#                       the post group also holds the parameter post2before -> singular matrix -> LinAlgError; CHANGED after review),
#                       then threshold, filtered by refractoriness (next_allowed_spiketime
#                       = t_spike + refractory - 0.5 dt), then LS.push(spikes). PoissonGroup: rand(N) < rate*dt.
#   'connections'       every Connection.do_propagate(): XeAe -> delayed buffer, AeAi -> ge_i, AiAe -> gi_e, and the C-STDP updaters:
#                       pre (immediate) pre = 1 ; pre (delayed by max_delay) w -= nu_pre * post1(past) ; post: post2before = post2,
#                       post1 = post2 = 1, w += nu_post * pre(past) * post2before. Every w line is followed by clip(w, 0, wmax) on that synapse.
#                       CHANGED (review): the ORDER of these objects is not defined by the script (Network.prepare groups them by
#                       (id(source), id(delay)) and runs the groups in Python 2 dict order). Used here: propagation first, then LTD, then LTP --
#                       one of the possible orders, the one the modules use (diehlcook header, deviation 5).
#   'ops after_connections'  DelayConnection.delayed_propagate: ge_e += dr[cdi] ; dr[cdi] = 0 ; cdi += 1 (mod _max_delay)
#   'resets'            exc: StringReset 'v = v_reset_e; theta += theta_plus_e; timer = 0' on this step's spikes (no clamping of v);
#                       inh: scalar reset with refractory -> Refractoriness: v = v_reset_i for every neuron in LS[0:int(refrac/dt)+1]
#   'ops end'           RecentStateMonitors record pre, post1, post2 (used by the delayed STDP lookups)
#
#UNITS here: mV, ms for the neuron and conductance variables (the model's own numbers); SI seconds for the delay arithmetic, exactly as Brian
# (with brian_no_units) computes it, so the integer delay/lag of every synapse is the same as Brian's.
from __future__ import annotations
import numpy as np

MS = 0.001                          #brian_no_units: ms = 0.001 (SI seconds)


class Brian1DC:
    def __init__(self, W_in: np.ndarray, delays_s: np.ndarray, *, theta0: np.ndarray, dt_ms: float = 0.5, test_mode: bool = False,
                 nu_pre: float = 1e-4, nu_post: float = 1e-2, wmax: float = 1.0):
        """W_in (n_input, n_e) float64 -- Brian's XeAe orientation (pre x post). delays_s (n_input, n_e) SI seconds."""
        self.n_in, self.n_e = W_in.shape
        self.W = np.array(W_in, dtype=np.float64)
        self.dt = dt_ms
        self.test_mode = test_mode
        self.nu_pre, self.nu_post, self.wmax = nu_pre, nu_post, wmax
        n_e = self.n_e
        #neuron groups: v = v_rest - 40 mV at creation (the script)
        self.v_e = np.full(n_e, -65.0 - 40.0); self.ge_e = np.zeros(n_e); self.gi_e = np.zeros(n_e)
        self.theta = np.array(theta0, dtype=np.float64); self.timer = np.zeros(n_e)          #timer in SI seconds (it is compared to refrac_e in s)
        self.v_i = np.full(n_e, -60.0 - 40.0); self.ge_i = np.zeros(n_e); self.gi_i = np.zeros(n_e)
        self.next_e = -np.ones(n_e); self.next_i = -np.ones(n_e)                               #_next_allowed_spiketime (ms here)
        self.LS_i = []                                                                         #inh spike history, most recent last
        self.t = 0.0                                                                           #ms
        #delayed connection XeAe
        dt_s = 0.5 * MS if dt_ms == 0.5 else dt_ms * MS
        self.dt_s = dt_s
        self.idt = 1 / dt_s                                                                    #_invtargetdt
        self.max_delay_steps = int((10 * MS) / dt_s) + 1                                       #DelayConnection._max_delay (max_delay=10*ms)
        self.dr = np.zeros((self.max_delay_steps, n_e)); self.cdi = 0
        self.delay = np.array(delays_s, dtype=np.float64)
        #STDP (training only)
        self.pre = np.zeros(self.n_in); self.post1 = np.zeros(n_e); self.post2 = np.zeros(n_e)
        self.stdp_max_delay = self.max_delay_steps * dt_s                                      #CSTDP: max_delay = C._max_delay * dt
        self.ltd_delay_steps = int(self.stdp_max_delay / dt_s)                                 #SpikeMonitor(delay=max_delay): int(delay/dt)
        nd = int(((self.max_delay_steps + 1) * dt_s) / dt_s) + 1                              #RecentStateMonitor num_duration
        self.mon_pre = np.zeros((nd, self.n_in)); self.mon_post1 = np.zeros((nd, n_e)); self.mon_post2 = np.zeros((nd, n_e)); self.cti = 0
        self.nd = nd
        self.LS_X = []                                                                         #input spike history, most recent last

    def _ltp_delay(self, i, j):
        """CHANGED (review): the delay Brian's dense C-STDP column loop READS for synapse (i, j): flat index i + j*n_e of the (n_in, n_e) delay
        matrix (new_c_propagate.iterate_over_col: %SOURCEINDEX% + %TARGETINDEX% * _num_target_neurons), not delay[i, j]."""
        return self.delay.reshape(-1)[i + j * self.n_e]

    def _num_decay(self, tau_ms):
        """stateupdater.get_linear_equations_solution_numerically for a diagonal M: E = 1 + (dt/100)*M, C = E.C, 100 times (SI units)."""
        E = 1.0 + (self.dt_s / 100) * (-1.0 / (tau_ms * MS))
        C = 1.0
        for _ in range(100):
            C = E * C
        return C

    def normalize_weights(self, target: float = 78.0):
        colSums = np.sum(np.copy(self.W), axis=0)
        colFactors = target / colSums
        for j in range(self.n_e):
            self.W[:, j] *= colFactors[j]

    def step(self, x_spikes: np.ndarray):
        """One Brian time step with this step's Poisson spikes (bool, n_input). Returns (exc spikes, inh spikes) as bool arrays."""
        dt, n_e = self.dt, self.n_e
        # ---------------------------------------------------------------- 'groups'
        # exc: Euler on all variables from the old values
        v, ge, gi = self.v_e, self.ge_e, self.gi_e
        v_tmp = ((-65.0 - v) + ge * (-v) + gi * (-100.0 - v)) / 100.0
        ge_tmp = -ge / 1.0
        gi_tmp = -gi / 2.0
        if not self.test_mode:
            th_tmp = -self.theta / 1e7
        self.v_e = v + v_tmp * dt; self.ge_e = ge + ge_tmp * dt; self.gi_e = gi + gi_tmp * dt
        if not self.test_mode:
            self.theta = self.theta + th_tmp * dt
        self.timer = self.timer + 100.0 * self.dt_s                                             #dtimer/dt = 100.0 (no units -> per second)
        cond = (self.v_e > (self.theta - 20.0 + -52.0)) * (self.timer > 5 * MS)
        sp_e = np.flatnonzero(cond)
        sp_e = sp_e[self.next_e[sp_e] <= self.t]
        self.next_e[sp_e] = self.t + 5.0 - 0.5 * dt
        # inh: Euler
        v, ge, gi = self.v_i, self.ge_i, self.gi_i
        v_tmp = ((-60.0 - v) + ge * (-v) + gi * (-85.0 - v)) / 10.0
        self.v_i = v + v_tmp * dt; self.ge_i = ge + (-ge / 1.0) * dt; self.gi_i = gi + (-gi / 2.0) * dt
        sp_i = np.flatnonzero(self.v_i > -40.0)
        sp_i = sp_i[self.next_i[sp_i] <= self.t]
        self.next_i[sp_i] = self.t + 2.0 - 0.5 * dt
        self.LS_i.append(sp_i); self.LS_i = self.LS_i[-5:]
        sp_x = np.flatnonzero(x_spikes)
        self.LS_X.append(sp_x); self.LS_X = self.LS_X[-(self.ltd_delay_steps + 1):]
        # STDP trace groups: linear -> exact
        if not self.test_mode:
            self.pre = self.pre * np.exp(-dt / 20.0)
            self.post1 = self.post1 * self._num_decay(20.0); self.post2 = self.post2 * self._num_decay(40.0)
        # ---------------------------------------------------------------- 'connections'
        for i in sp_x:                                                                          #XeAe (DelayConnection.propagate)
            drind = (self.cdi + np.array(self.idt * self.delay[i, :], dtype=int)) % self.max_delay_steps
            self.dr[drind, np.arange(n_e)] += self.W[i, :]
        for j in sp_e:                                                                          #AeAi one-to-one 10.4
            self.ge_i[j] += 10.4
        for j in sp_i:                                                                          #AiAe all-but-self 17
            row = np.full(n_e, 17.0); row[j] = 0.0
            self.gi_e += row
        if not self.test_mode:
            for i in sp_x:                                                                      #pre, immediate
                self.pre[i] = 1.0
            if len(self.LS_X) > self.ltd_delay_steps:                                           #pre, delayed by max_delay
                for i in self.LS_X[-1 - self.ltd_delay_steps]:
                    for k in range(n_e):
                        t_past = self.stdp_max_delay - self.delay[i, k]
                        post1 = self.mon_post1[(self.nd + self.cti - 1 - int(self.idt * t_past)) % self.nd, k]
                        w = self.W[i, k] - self.nu_pre * post1
                        self.W[i, k] = min(max(w, 0.0), self.wmax)
            for j in sp_e:                                                                      #post
                post2before = self.post2[j]
                self.post1[j] = 1.0; self.post2[j] = 1.0
                for i in range(self.n_in):
                    pre = self.mon_pre[(self.nd + self.cti - 1 - int(self.idt * self._ltp_delay(i, j))) % self.nd, i]
                    w = self.W[i, j] + self.nu_post * pre * post2before
                    self.W[i, j] = min(max(w, 0.0), self.wmax)
        # ---------------------------------------------------------------- 'ops after_connections'
        self.ge_e = self.ge_e + self.dr[self.cdi]
        self.dr[self.cdi] = 0.0
        self.cdi = (self.cdi + 1) % self.max_delay_steps
        # ---------------------------------------------------------------- 'resets'
        self.v_e[sp_e] = -65.0
        if not self.test_mode:
            self.theta[sp_e] += 0.05
        self.timer[sp_e] = 0.0
        for s in self.LS_i:                                                                     #Refractoriness: LS[0:int(2ms/dt)+1] = last 5 steps
            self.v_i[s] = -45.0
        # ---------------------------------------------------------------- 'ops end'
        if not self.test_mode:
            self.mon_pre[self.cti] = self.pre; self.mon_post1[self.cti] = self.post1; self.mon_post2[self.cti] = self.post2
            self.cti = (self.cti + 1) % self.nd
        self.t += dt
        e = np.zeros(n_e, bool); e[sp_e] = True
        i_ = np.zeros(n_e, bool); i_[sp_i] = True
        return e, i_
