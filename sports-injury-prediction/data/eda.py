import pandas as pd
if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('injury_data_with_categories.csv')

    #player weight float between 0 and 150 kg    
    #player height float between 0 and 250 cm
    #player age int between 0 and 100
    #previous injury : true or false 
    #training intesntiy : float between 0 and 1 

    print("Positions:", df["Position"].unique()) # ['Midfielder' 'Forward' 'Goalkeeper' 'Defender']
    print("Recovery_Time:", df["Recovery_Time"].unique()) # Int between 1-6
    print("Training_Surface:", df["Training_Surface"].unique()) # ['Artificial Turf' 'Hard Court' 'Grass']
    print("Likelihood_of_Injury:", df["Likelihood_of_Injury"].unique()) # [0,1]




   