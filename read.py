import pickle

data_path = "visual7w_data/train.pkl"
with open(data_path, 'rb') as f:
    data = pickle.load(f)

print(data['img_prefix'].shape)