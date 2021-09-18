import time

class timer(object) :
    def __init__(self):
        self.start_time = time.time()
        self.episode_interval = 0
        self.episode_start = 0

    def start_episode(self) :
        self.episode_start = time.time()

    def finish_episode(self) :
        self.episode_interval = time.time() - self.episode_start
        return self.episode_interval


    def finish_train(self) :
        return self.start_time - time.time()


    def cal_time(self, sec) :
        hour = sec / 3600
        sec = sec % 3600
        minute = sec / 60
        sec = sec % 60
        return (hour,minute,sec)

    def time_str(self,t):
        hour, minute, sec = self.cal_time(t)
        str = '{} hours, {} minutes, {} seconds'.format(hour,minute,sec)
        return str