import os

class Interface:
    def __init__(self):
        self.didtrain = False
        self.episodes = None
        self.path_dqn = "model_dqn_saved"
        self.path_a2c = "model_a2c_saved"
        self.filename = None
        self.path = None

        # Création des dossiers si besoin
        os.makedirs(self.path_dqn, exist_ok=True)
        os.makedirs(self.path_a2c, exist_ok=True)

    def ask_mode(self):
        user_input = input("Choose the agent mode - DQN or A2C (d/a): ").lower()
        if user_input == 'd':
            return "dqn"
        elif user_input == 'a':
            return "a2c"
        else:
            print("Invalid input, defaulting to DQN.")
            return "dqn"

    def didtrainfct(self):
        user_input = input("Do you want to train the model? (y/n): ").lower()
        if user_input == 'y':
            self.didtrain = True
            self.episodes = int(input("How many episodes would you like to train the model for? "))

    def didtestfct(self):
        return input("Do you want to test the model? (y/n): ").lower() == 'y'

    # ---------------------------
    # DQN
    # ---------------------------
    def ask_save_dqn(self):
        user_input = input("Do you want to save the DQN model? (y/n): ").lower()
        if user_input == 'y':
            self.filename = input("Enter the filename to save the DQN model (without extension): ") + ".pth"
            self.path = os.path.join(self.path_dqn, self.filename)
            print(f"The DQN model will be saved at: {self.path}")
            return True
        return False

    def ask_load_dqn(self):
        user_input = input("Do you want to load an existing DQN model? (y/n): ").lower()
        if user_input == 'y':
            self.filename = input("Enter the filename of the DQN model to load (without extension): ") + ".pth"
            self.path = os.path.join(self.path_dqn, self.filename)
            print(f"The DQN model will be loaded from: {self.path}")
            return True
        return False

    # ---------------------------
    # A2C
    # ---------------------------
    def ask_save_a2c(self):
        user_input = input("Do you want to save the A2C model? (y/n): ").lower()
        if user_input == 'y':
            self.filename = input("Enter the filename to save the A2C model (without extension): ") + ".pth"
            self.path = os.path.join(self.path_a2c, self.filename)
            print(f"The A2C model will be saved at: {self.path}")
            return True
        return False

    def ask_load_a2c(self):
        user_input = input("Do you want to load an existing A2C model? (y/n): ").lower()
        if user_input == 'y':
            self.filename = input("Enter the filename of the A2C model to load (without extension): ") + ".pth"
            self.path = os.path.join(self.path_a2c, self.filename)
            print(f"The A2C model will be loaded from: {self.path}")
            return True
        return False
