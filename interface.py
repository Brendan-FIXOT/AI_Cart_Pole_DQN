import os

class Interface :
    def __init__(self) :
        self.didtrain = False
        self.episodes = None
        self.path = "model_saved"
        self.filename = None
        
    def didtrainfct(self) :
        # Demande si l'utilisateur veut entraîner le model
        user_input = input("Do you want to train the model? (y/n): ").lower()

        if user_input == 'y' :
            self.didtrain = True
            # Demande du nombre d'entraînement
            self.episodes = int(input("How many episodes would you like to train the model for? "))
              
    def didtestfct(self) :
        # Demande si l'utilisateur veut tester le model
        user_input = input("Do you want to test the model? (y/n): ").lower()
            
        if user_input == 'y' :
            return True
        else :
            return False
        
    def ask_filename(self) :
        self.filename = input("Enter the filename of this model (without extension): ") + ".pth"
        self.path = os.path.join(self.path, self.filename)  # Ajoute le nom du fichier au chemin
        print(f"The model will be saved at: {self.path}")
        
    def ask_save(self) :
        # Demande si l'utilisateur veut sauvegarder le model
        user_input = input("Do you want to save the model? (y/n): ").lower()
        
        if user_input == 'y' :
            # Demande du nombre d'entraînement
            self.filename = input("Enter the filename to save the model (without extension): ") + ".pth"
            self.path = os.path.join(self.path, self.filename)  # Ajoute le nom du fichier au chemin
            return True
        else :
            return False
        
    def ask_load(self) :
        # Demande si l'utilisateur veut charger un modèle existant
        user_input = input("Do you want to load an existing model? (y/n): ").lower()
        
        if user_input == 'y' :
            # Demande à l'utilisateur d'inscrire le nom du fichier pour charger le modèle
            self.ask_filename()
            return True
        else :
            return False