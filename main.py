# main.py


"""INITIAL/preliminary Pseudocode for imagined front-end. WIP"""

"""Overall structure: MNIST -> Download() -> rawdata -> Preprocessor() -> spikedata    
                                                                Model() -> modelobject + spikedata  --> traininghistoryobject -> Analysis() -> various analytical outputs                                                                
"""
"""Planned classes:
Frontend: Settings, Preprocessor, SleepModel, Analysis.
Backend: Integrator[multiple], Thresholding[multiple], Plasticity[multiple] PhaseController, 
      - Matching Config class-sorting"""
# Settings class controlls overall configuration/runtime
Settings.configure(device="CPU", update_style="time", global_seed=[DEFAULT=], logging=TRUE)

# Running download class
raw_data = Download("MNIST_digits", N = total number of images to download  [OR N_train = number of images to be downloaded from train data, N_test = Number of images to be downloaded from test data ])

###ALTERNATIVE 1: Running pre-processor stepwise

# Loading data into pre-processor class (wrap data)
wr_data = Preprocessor(raw_data)
wr_data.normalize_MNIST()
wr_data.img_to_spikes(encoding="poisson", random_seed = [default: global_seed])
wr_data.add(noise="gaussian", intensity=0.2, random_seed = [default: global_seed])
# Extracting processed data
transformed_data = wr_data.get_data() 

######  Model instantiation  #######
model_1 = SNN()

# Compilation of base structure can be done via 1. compile method, or 2. set_nodes + set_connectivity methods
# Base structure
model_1.compile(GENERATIVE MODEL HERE; builds lattice according to generative algorithms + p_connectivity + N_nodes) #defaults stored in config. this code overrides default settings
model_1.set_nodes(N_exc = , N_inh = , integrator_algorithm = , base_threshold =) Here node_integraion controls the "spatiotemporal summation of inputs" aka the integraion/SUMMATION algorithm in ML. Currently/first: LIF neuron 
model_1.set_connectivity(p_connections_inh_exc, p_connections_exc_inh, p_connections_exc_exc, [OR p_connections_all], initial_weights_inh, initial_weights_exc OR initial_weights_all) #Here the p_connections_ei refers to probabiity of connections excitatory to inhibitory neuron]
# Extended base structure
model_1.set_thresholds(base=...)
model_1.set_adaptible_thresholds(decay_tau_all, [OR decay_tau_inh, decay_tau_exc], spike_rise_all [OR spike_rise_inh, spike_rise_exc])  ##CONCERN: +, - confusion; negative vs positive inputs for inh vs exc
model_1.set_sleep(regularization_strengths=..., trigger_weights= )
model_2.set_STDP(A_plus/LTP_amplitude, A_minus/LTD_amplitude, decay_tau_all, [OR decay_tau_inh, decay_tau_exc], spike_rise_all [OR spike_rise_inh, spike_rise_exc], ...) # This will call the above functions needed to set up an STDP without having to specify the above functions individually. It can be combined with set_adaptible_thresholds() 


###ALTERNATIVE 2: Running pre-processor w/ pipeline
transformed_data = (
    Preprocessor(raw_data)
    .normalize_MNIST()
    .img_to_spikes(encoding="poisson")
    .add(noise="gaussian", intensity=0.2)
    .get_data()
)




# Other functionalities
model_1.add(integrator_noise, integrator_noise_intensity, ...)

model_1.save(filename="compiled_model_1", file_path="/custompath") 
model_1.load(filename="compiled_model_1", file_path="/custompath")

## Compilation methods above are methods directly named for functions they call. 
# POTENTIAL ALTERNATIVE + quicker setup methodology:
model_2 = SNN()
model_2.compile(…) #FREEZE model and create hash

# Ways a model object can "tell us about itself":
model_1.get_network() # Returns connectivity info, number of nodes, type of nodes (integration algorithm), thesholds (adaptive or not), noise, etc.
model_1.plot_connectivity/connectivity_matrix() # return connectivity matrix
model_1.plot_network() # visualize network structure
model_1.get_node_degrees() # returns degree of each node in matrix form
model_1.plot_node_degrees() # plot degree distribution (in initial model)

training_object = model_1.train(data = transformed_data, epochs=..., validation_folds= , save_history=FALSE) ##SHOULD learning_rate be internal to the compilation architecture since the model is stdp SNN? 

#Loading training object/history into Analysis class
analyse_run_1 = training_object.Analysis()
analyse_run_1 .t_sne(...)
pca_object = training_object.PCA()
analyse_run_1 .phi_score(...)
pca_object_run_1.phi_score(...)
analyse_run_1 .plot_spikeraster(...)
analyse_run_1 .plot_degree_distributions(...)
analyse_run_1 .plot_clustering(...)
pca_object = object.plot_clustering(…)