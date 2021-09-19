class Config(object):
    def __init__(self):
        self.seed = None

        self.env = None
        self.env_name = None
        self.env_observation = 'image' #image or vector
        self.env_args = {}

        self.agent_name = None

        self.epsilon = None
        self.max_eps = 1
        self.min_eps = 0.05
        self.max_step = 10000
        self.policy_trans = 10000

        self.replay_buffer = 'buffer' #per, buffer

        self.train_start = 1000
        self.update_interval = 4 #interval of update network at step
        self.max_epi = None
        self.hyperparameters = {}

        self.save_model = True
        self.save_interval = 1000
        self.load_path = ''
        self.save_path = ''

        self.log_interval = 1000

