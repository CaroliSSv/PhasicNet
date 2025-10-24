#### Model class + methods

#########DRAFT STRUCTURE + PRINT statements
 # Hardcoded
## Todo:    1.  classes --> config, clf, + COMPILE --> FREEZE --> PRODUCE LOGGING HASH (+ AUTO-save?)
            ## Class for building net? YES -> consistency of classes + freeze + dirty flag for pre-compile
#           2. parameters --> default config source (YAML)
#           3. training loops hardcoded --> collect relevant param class
#           4. NOTES -> google OR numpy structure? Bio/NEURO or ML?
#           5. ADD COMPILE function
#           6. INTEGRATE LOGGER + CONFIGS from 
##          7. var-name consistency?

###Spes.g. ide-oppsett, alt i klasse + SNN "wrapper" for moduler (rules + architecture)
class SNN:
    def __init__(self, name= "SNN", seed=None): ##        MUST BE GLOBALSEED IF GIVEN
        self.name = name
        self.seed = seed ##        MUST BE GLOBALSEED IF GIVEN

        #Rules
        self.phase = None
        self.integrator = None
        self.threshold = None
        self.plasticity = None

        #Noise 
        self.noise = None

        #runtime rules
        self.mode = None
        self.device = None
        self.schema = None
        self.datacapture = None

        #Architectural  -> convert to class network-setting/builder/create --> snn.build.set_... called internally?
        ## *** class
        ### Number of nodes
        self.N_input_nodes = 0
        self.N_excitatory_nodes = 0
        self.N_inhibitory_nodes = 0
        ### Matrixes
        self.connectivity = {}
        self.weights = {}
        self._adjacency_matrix = None
        self._weights_matrix = None

    def set_phaserules(self, phase_class=None):
        self.phase = phase_class
        return self.phase

    def set_integrator(self, integrator_class=None):
        self.integrator = integrator_class
        return self.integrator

    def set_threshold(self, threshold_class=None):
        self.threshold = threshold_class
        return self.threshold

    def set_plasticity(self, plasticity_class=None):
        self.plasticity = plasticity_class
        return self.plasticity

    def set_noise(self, noise_class=None):
        self.noise = noise_class
        return self.noise

    def set_rules(self, integrator_class=None, threshold_class=None, plasticity_class=None):
        """Set all rules at once, using class objects."""
        if integrator_class is not None:
            self.set_integrator(integrator_class)
        if threshold_class is not None:
            self.set_threshold(threshold_class)
        if plasticity_class is not None:
            self.set_plasticity(plasticity_class)
        if all(class_input is None for class_input in [integrator_class, threshold_class, plasticity_class]):
            raise ValueError("No new rules set.")
        ## Planned addons for possible errors:
        #### REMOVE??  
        #### *** HOW TO MAKE SURE RUNNING THIS DOES NOT WIPE EX. INTEGRATOR OR PLASTICITY ETC.???
        #### If input_class == input_class -> "f"{input_class-name} already set." --> OVERWRITE initial input or add?
        #### If ABSTRACT_CLASS from input_class == self.ABSTRACT_CLASS in SNN object: -> "f{Abstract class-name} already set. To replace write 'replace', to add write 'add', to cancel write 'c' or press enter."" --> OVERWRITE initial input or add?


        print(f"[{self.name}] Initialized (seed={self.seed}).")
############################
    def set_nodes(self, N_excitatory: int = 0, N_inhibitory: int = 0, N_inputs: int = 0):

        """Set number of nodes in network, create node-matrix***. 
        Args:
        N_excitatory: int = 0, Excitatory neurons
        N_inhibitory: int = 0, Inhibitory neurons
        N_inputs: int = 0 Input nodes, each a source of spikes (from preprocessor)
        Returns:
        """
        if N_excitatory < 1:
            raise ValueError("No excitatory nodes given.")
        self.N_excitatory_nodes = N_excitatory
        self.N_inhibitory_nodes = N_inhibitory
        self.N_input_nodes = N_inputs
        print(f"[{self.name}] Nodes set: number of excitatory nodes: {N_excitatory}, number of inhibitory nodes: {N_inhibitory}, number of input nodes: {N_inputs}")
##############################
    def set_connectivity(self, p_connectivity_all: float = None, p_connectivity_exc_exc: float = None, p_connectivity_exc_inh: float = None, 
                         p_connectivity_inh_exc: float = None, p_connectivity_inh_inh: float = None, p_connectivity_input_exc: float = None):
                         #p_connectivity_input_inh: float = None):
        """Set initial probability of synapse/connection between all neurons/nodes or specific neuronal/node-groups.
        Args:
        
        Returns:
        """
        self.connectivity = dict(
            all = p_connectivity_all,
            exc_exc= p_connectivity_exc_exc,
            exc_inh= p_connectivity_exc_inh,
            inh_exc= p_connectivity_inh_exc,
            inh_inh= p_connectivity_inh_inh,
            input_exc= p_connectivity_input_exc)
            #input_inh= p_connectivity_input_inh)

        print(f"[{self.name}] Connectivity: {self.connectivity}")

############################
    #FIX to classbuilder/net creator
    def set_weights(self, weights_all: float = None, weights_exc_exc: float = None, weights_exc_inh: float = None,
                    weights_inh_exc: float = None, weights_inh_inh: float = None, weights_input_exc: float = None): #weights_input_inh: float = None):
                    
        """Set initial synaptic strengths of for all synapses or for synapses between specific neuron-groups. /OR/ 
        Set initial weights of connections/edges between all nodes or specific node-groups.
        Args:
        
        Returns:
        self. dictionary of initial synaptic strengths"""
        self.weights = dict(
            all = weights_all,
            exc_exc = weights_exc_exc,
            exc_inh = weights_exc_inh,
            inh_exc = weights_inh_exc,
            inh_inh = weights_inh_inh,
            input_exc = weights_input_exc
            )

        print(f"[{self.name}] Initial weights: {self.weights}")

 
    def set_architecture(self): ### FIX utvendig eller invending? 
        """Build architecture internally: set nodes and their interconnections.
        """
        N_inputs, N_excitatory, N_inhibitory = self.N_input_nodes, self.N_excitatory_nodes, self.N_inhibitory_nodes
        N_total = N_inputs + N_excitatory + N_inhibitory
        adj = np.zeros((N_total, N_total), dtype=bool)
        W = np.zeros((N_total, N_total), dtype=float)

        
        idx_input = np.arange(0, N_inputs)
        idx_exc = np.arange(N_inputs, N_inputs + N_excitatory)
        idx_inh = np.arange(N_inputs + N_excitatory, N_total)

        def connect(source, target, probability, weight):
            if probability is None or probability <= 0 or probability > 1:
                raise ValueError("Probability cannot be above 1 or below or = to 0.")
            mask = np.random.rand(len(source), len(target)) < probability
            adj[np.ix_(source, target)] = mask
            W[np.ix_(source, target)] = mask*weight

        # input nodes=SPIKE SOURCES to network nodes
        connect(idx_input, idx_exc,
                self.connectivity.get("input_exc", 0.1),
                self.weights.get("input_exc", 0.1))
                
#        connect(idx_input, idx_inh,
#                self.connectivity.get("input_inh", 0.0),
#                self.weights.get("input_inh", 0.2))

        # recurrent connections
        ### FIX initial values -> param_base_values
        connect(idx_exc, idx_exc,
                self.connectivity.get("exc_exc", 0.1),
                self.weights.get("exc_exc", 0.1))
        connect(idx_exc, idx_inh,
                self.connectivity.get("exc_inh", 0.1),
                self.weights.get("exc_inh", 0.1))
        connect(idx_inh, idx_exc,
                self.connectivity.get("inh_exc", 0.1),
                self.weights.get("inh_exc", -0.4))
        connect(idx_inh, idx_inh,
                self.connectivity.get("inh_inh", 0.1),
                self.weights.get("inh_inh", -0.1))

        self._adjacency_matrix, self._weights_matrix = adj, W
        print(f"[{self.name}] Architecture built: {adj.sum()} edges, density={adj.mean():.3f}).")

        return self._adjacency_matrix, self._weights_matrix

#######################################




    def compile():
        ### FIX RESTRUCTURE CLASSES -> FREEZE setup
        pass

    def load_data():
        #called during train()
        #call dataloader

    def get_network():
        pass
    def get_rules():
        pass

    def train(data = None, set_mode = None, set_device = None, update_schema = None, set_batching = None):
        #call modecontroller.tick() -> self.mode 
        #call phasecontroller.tick() -> self.phase
        #call self.load_data() -> uses data object and network architecture and runtime info to load data during training
        #call self.noise.step() -> ensure noise applied on top of the following
        #call self.integrator.step() -> integrate_membrane
        #call self.threshold.step() -> update_threshold, emit_spikes
        #call self.plasticity.step() -> update_weights
        pass

 
