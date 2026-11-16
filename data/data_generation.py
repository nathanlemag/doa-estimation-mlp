import numpy as np
from torch.utils.data import Dataset

class SnapshotDataset(Dataset):
    def __init__(self,snapshots,doas):
        self.snapshots = snapshots
        self.doas = doas
        
    def __len__(self):
        return len(self.doas)
    
    def __getitem__(self, idx):
        return self.snapshots[idx], self.doas[idx]
    
def sind(degrees):
    return np.sin(np.deg2rad(degrees))

def steering_vector(m_elems,angle_deg):        
    return np.exp(-1j*np.pi*np.arange(m_elems).reshape(-1,1) * sind(angle_deg))

def compute_snapshot(signals, doas, T, m_elems):    
    noise = (np.random.randn(m_elems,T)+ 1j * np.random.randn(m_elems,T)) / np.sqrt(2)        
    a = steering_vector(m_elems,doas) # array response
    noiseless_signal = a @ signals 
    snapshot = noiseless_signal + noise
    return snapshot

def generate_K_source_signals(K,T,gamma,rng):
    gammas = np.ones((K,))*gamma
    s = np.diag(np.sqrt(gammas)) @ (rng.standard_normal((K,T))+1j*rng.standard_normal((K,T)))/np.sqrt(2)
    return s

def generate_snapshot_dataset(nof_snapshots,nof_signals,m_elems,SNRs,fov_start,fov_end,samples_per_snr):
    rng = np.random.default_rng(73)
    # init dicts
    resp_dict = {}
    doa_dict = {}
    for i in range(nof_signals):
        resp_dict[f'{i+1}'] = []
        doa_dict[f'{i+1}'] = []

    # compute snapshots and store respective doas
    for snr in SNRs:
        for _ in range(samples_per_snr):
            gamma = 10**(snr/10)
            for i in range(nof_signals):
                K = i+1
                s = generate_K_source_signals(K=K,T=nof_snapshots,gamma=gamma,rng=rng)
                doa = rng.uniform(fov_start,fov_end,i+1)
                snapshot = compute_snapshot(s,doa,nof_snapshots,m_elems)
                resp_dict[f'{i+1}'].append(snapshot)
                doa_dict[f'{i+1}'].append(doa)

    return resp_dict, doa_dict