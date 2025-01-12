class PIDAgent(object):
    def __init__(self, Kp=0.1, Ti=1e7, Td=100):
        self.name = "PID"
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td

    def reset(self, budget, T):
        self.Ki = self.Kp * T / self.Ti
        self.Kd = self.Kp * self.Td / T

        self.budget = budget
        self.target_rho = budget / T
        self.t = 0
        self.acc_payment = 0

        self.last_err = 0
        self.acc_err = 0
        self.lamb = self.target_rho


    def step(self, value):
        self.t += 1
        bid = max(self.lamb, 0)
        return bid

    def update(self, reward, payment):
        self.acc_payment += payment
        self.rho = self.acc_payment / self.t
        err = self.target_rho - self.rho
        self.acc_err += err

        if self.t > 1:
            self.lamb = self.lamb + self.Kp * err + self.Ki * self.acc_err + \
                        self.Kd * (err - self.last_err)

        self.last_err = err
