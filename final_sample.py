import pandas as pd
import os
def final_sample(attack_types, samples_per_attack_type, RANDOM_SEED=42):
    final_attack_samples = []
    
    dtype_df = pd.read_csv('attack_samples_dtype.csv')
    dtype_dict = dtype_df.set_index('Column Name')['Data Type'].to_dict()
    attack_samples_df = pd.read_csv('attack_samples.csv', dtype=dtype_dict, low_memory=False)  # Load pre-collected attack samples
    
    for attack_type in attack_types:
        attack_type_packets = attack_samples_df[attack_samples_df['Label'] == attack_type]
        n_samples = samples_per_attack_type
        
        print(f"Processing attack type: {attack_type}, available packets: {len(attack_type_packets)}, requested samples: {n_samples}")

        # Shuffle attack packets before sampling
        if len(attack_type_packets) < samples_per_attack_type:
            # Upsample with replacement if not enough packets are available
            attack_type_packets = attack_type_packets.sample(n=n_samples, replace=True, random_state=RANDOM_SEED)
        else:
            # Sample without replacement
            attack_type_packets = attack_type_packets.sample(frac=1, random_state=RANDOM_SEED)
            attack_type_packets = attack_type_packets.head(n=n_samples)

        print(f"Sampled {len(attack_type_packets)} packets for attack type: {attack_type}")
        final_attack_samples.append(attack_type_packets)


    final_attack_df = pd.concat(final_attack_samples, ignore_index=True)
    print(f"Final attack samples size: {len(final_attack_df)}")

    final_attack_df.to_csv('final_attack_samples.csv', index=False)
    print("Final attack samples saved to 'final_attack_samples.csv'")
