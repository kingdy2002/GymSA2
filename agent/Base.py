import torch
import utills

class agent_base(object) :
    def __init__(self,config) :
        self.config = config
        self.hyperparameters = config.hyperparameters

        self.action_space = config.env.action_space

        self.observation_space = config.env.observation_space
        self.observation_space_high = config.env.observation_space.high
        self.observation_space_low = config.env.observation_space.low

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.now_epi = 0
        self.tot_step = 0
        self.epi_step = 0

        self.network = None
        self.env = config.env
        self.config = config

        self.epi_return = []
        self.epi_train_time = []
        self.Timer = utills.timer.timer()

    """
    게임 환경 리플레이
    """

    def step(self) :
        pass

    def update(self) :
        pass


    """
    실햄과 학습 관련 유틸 함수
    """

    def select_action(self,observation) :
        pass

    def compute_loss(self,states, actions, rewards,next_states, dones):
        pass

    def recent_return(self) :
        sum_reward = 0
        for i in range(1,10) :
            sum_reward = sum_reward + self.epi_return[-i]

        return sum_reward/9

    def left_train_time(self,epi) :
        sum_time = 0
        for i in range(1,10) :
            sum_time = sum_time + self.epi_train_time[-i]
        
        left_time =  (sum_time/9) * (self.config.max_epi - epi)

        return self.Timer.time_str(left_time)

    """
    모델의 저장과 불러내기
    """


