class Interface :
    def __init__(self, didtrain = False) :
        self.didtrain = didtrain
        self.episodes = None
        
    def didtrainfct(self) :
        # Demande si l'utilisateur veut entraîner le model
        user_input = input("Do you want to train the model? (y/n): ").lower()

        if user_input == 'y' :
            self.didtrain = True
            # Demande du nombre d'entraînement
            self.episodes = int(input("How many episodes would you like to train the model for? "))
            
    def didtestfct(self) :
        # Demande si l'utilisateur veut tester le model
        test_input = input("Do you want to test the model? (y/n): ").lower()
            
        if test_input == 'y' :
            return True
        else :
            return False