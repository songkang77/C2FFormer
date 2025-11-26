import os
from datetime import datetime

def visualize_tensors(ori_x, x, path: str):
    import matplotlib.pyplot as plt
    
    
    _, _, n_features = ori_x.shape
    
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_folder = os.path.join(path, f'visualization_{timestamp}')
    os.makedirs(result_folder, exist_ok=True)
    
    # 
    for feature_idx in range(n_features):
        # 
        ori_data = ori_x[0, :, feature_idx].cpu().detach().numpy()
        processed_data = x[0, :, feature_idx].cpu().detach().numpy()
        
        # 
        time_steps = range(len(ori_data))
        
        # 
        plt.figure(figsize=(12, 6))
        
        # 
        plt.plot(time_steps, ori_data, label='Original Data', color='blue')
        
        # 
        plt.plot(time_steps, processed_data, label='Predicted Data', color='red')
        
        # 
        plt.title(f'Comparison of Original and Predicted Data - Feature {feature_idx + 1}')
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        
        # 
        result_path = os.path.join(result_folder, f'feature_{feature_idx + 1}.png')
        plt.savefig(result_path)
        plt.close()
        
    return result_folder


