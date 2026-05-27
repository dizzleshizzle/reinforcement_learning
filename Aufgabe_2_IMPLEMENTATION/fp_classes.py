import numpy as np

class environment:
	def __init__(self):
		self.N_states = 100
		self.target_position = 8
		self.starting_position = 30
		
		self.obstacle_interval = np.arange(9,12)
		self.P_obstacle = 0.0
	
class agent:
	def __init__(self,env_):
		self.N_episodes = 10000
		self.tmax_MSD = 100
		
		self.x = 1
		self.Q = np.zeros((env_.N_states,3))
		self.alpha = 0.01
		self.gamma = 0.9
		self.epsilon = 1.0
		self.target_reward = 10.0
		self.zero_fraction = 0.9

		self.D = 0.125
		self.P_diffstep = 2*self.D
		
		self.x_old = None
		
		if self.P_diffstep > 1.0:
			print(f"self.P_step = {self.P_step} > 1.0 in agent.__init__(...)")
			print("Diffusion constant self.D possibly too large. Pick a smaller self.D.")
			exit()
	
	def random_step(self):
		if np.random.rand()<self.P_diffstep:
			self.x+=2*np.random.randint(0,2) -1
		
	def adjust_epsilon(self,episode):
		if (episode < self.zero_fraction*self.N_episodes):
			self.epsilon = 1 - episode/(self.zero_fraction*self.N_episodes)
		else:
			self.epsilon = 0
	
	def choose_action(self):
		"""
		wählt eine Zufallsaktion aus mit Wahrscheinlichkeit self.epsilon oder falls zwei Aktionen die höchsten Q-Werte haben.
		Andernfalls wird der höchste Wert in der jeweiligen Zeile ausgewählt.
		"""
		if np.random.rand()<self.epsilon:
			self.chosen_action = np.random.randint(0,3)
		else:
			q_values = self.Q[self.x,:]
			max_q = np.max(q_values)
			max_actions = np.where(q_values == max_q)[0]
			self.chosen_action = np.random.choice(max_actions)

	def fix_boundary_crossing(self, env_):
		self.x = self.x % env_.N_states

	def perform_action(self,env_):
		"""
		Hier werden die Aktionen ausgeführt. Der Index der Aktion entspricht der Verschiebung auf der x-Achse + 1
		"""

		"""
		if self.chosen_action == 0:
			self.x -= 1
			self.fix_boundary_crossing(env_)
		if self.chosen_action == 2:
			self.x += 1
			self.fix_boundary_crossing(env_)
		"""
		dx = self.chosen_action -1

		self.x += dx

		self.fix_boundary_crossing(env_)
	
	def update_Q(self,env_):
		"""
		Hier werden die Werte der Q-Matrix nach jeder Aktion entsprechend aktualisiert
		"""
		i = self.x_old
		ip = self.x
		j = self.chosen_action

		if ( i == env_.target_position and j== 1):
			R = self.target_reward
		else:
			R = 0

		self.Q[i,j] = self.Q[i,j] + self.alpha *(R + self.gamma * np.max(self.Q[ip,:]) - self.Q[i,j])

	
	def stoch_obstacle(self,env_):
		pass

