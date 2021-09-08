import pickle
import numpy as np

expert_demo = pickle.load(open('expert_demo\expert_demo_coso.p', "rb"))
demonstrations = np.array(expert_demo)
print(demonstrations)
print(demonstrations[1])