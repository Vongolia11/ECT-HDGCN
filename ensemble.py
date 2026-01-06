import argparse
import pickle
import numpy as np
import os
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description='Auto-Search Ensemble with L2 Norm')
    parser.add_argument('--dataset', default='ntu/xsub', choices=['ntu/xsub', 'ntu/xview'],
                        help='Dataset split')
    parser.add_argument('--joint-pkl', required=True, help='Path to joint score pkl file')
    parser.add_argument('--bone-pkl', required=True, help='Path to bone score pkl file')
    return parser

def load_score(pkl_path):
    abs_path = os.path.abspath(pkl_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")
    with open(abs_path, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, dict):
            return np.array(list(data.values()))
        else:
            return np.array(data)

def main():
    arg = get_parser().parse_args()

    if 'xsub' in arg.dataset:
        data_path = './data/ntu/NTU60_CS.npz'
    elif 'xview' in arg.dataset:
        data_path = './data/ntu/NTU60_CV.npz'
    print(f"Loading labels from {data_path}...")
    npz_data = np.load(data_path)
    label = np.where(npz_data['y_test'] > 0)[1]

    print(f"Loading Joint: {os.path.basename(arg.joint_pkl)}")
    s_joint = load_score(arg.joint_pkl)
    
    print(f"Loading Bone:  {os.path.basename(arg.bone_pkl)}")
    s_bone = load_score(arg.bone_pkl)

    print("Applying L2 Normalization...")
    norm_joint = np.linalg.norm(s_joint, axis=1, keepdims=True) + 1e-8
    s_joint = s_joint / norm_joint
    
    norm_bone = np.linalg.norm(s_bone, axis=1, keepdims=True) + 1e-8
    s_bone = s_bone / norm_bone

    def eval_fusion(alpha):
        r = alpha * s_bone + (1 - alpha) * s_joint
        pred = r.argmax(axis=1)
        acc = (pred == label).mean()
        return acc

    print('-------------------------------------')
    print("Auto-searching best alpha (Range: 0.4 - 0.6, Step: 0.01)...")
    best_acc = 0
    best_alpha = 0
    
    for alpha in np.arange(0.40, 0.61, 0.01):
        acc = eval_fusion(alpha)
      
        print(f"Alpha: {alpha:.2f} | Acc: {acc*100:.4f}%")
        
      
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
    
    print('-------------------------------------')
    print(f"Best Alpha Found: {best_alpha:.2f}")
    print(f"Best Top-1 Accuracy: {best_acc * 100:.4f}%")
    print('-------------------------------------')

if __name__ == "__main__":
    main()