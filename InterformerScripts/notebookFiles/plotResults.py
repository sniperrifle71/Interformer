import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def cumavg(m):
    cumsum = np.cumsum(m)
    return cumsum / np.arange(1, cumsum.size + 1)


pred_lens = [48]



models = ['Transformer', 'fsnet', 'AdaRNN']
root_path = '../../../data'


result_path = os.path.join(root_path, 'results')
KKK_result_path = os.path.join(root_path, 'Interformer','test_results','Finetune_ECL_UTSD-1G_INTREPETER_24_Exp25-06-16_09-07-48','ECL.csv','1440')
# Read all result files and concatenate them along the first dimension.
results = []
for file in os.listdir(KKK_result_path):
    if 'true' in file:
        print(file)
        resultsFragement = np.load(os.path.join(KKK_result_path, file), allow_pickle=True)
        results.append(resultsFragement)
        print(resultsFragement.shape)

KKK_data = np.concatenate(results, axis=0)

# print(KKK_data.shape)

KKK_data = np.load('/home/borong/data/Interformer/test_results/forecast_testing_etth1_sr_0.01_Interformer_ECL_theta_500_ftM_sl1440_ll0_pl96_pl96_dm256_nh8_el3_dl3_df512_fc3_ebtimeF_dtTrue_Exp25-07-28_10-56-20/ECL.csv/1440/finalPred.npy')
print(KKK_data.shape)

begin = 1250
end = 1350
channel = 1

# with open(os.path.join(result_path, f"Student_ECL_TimesNet_48_0.5.pkl"), "rb") as f:
#     KKK_results = pickle.load(f)

true_df = pd.read_csv(os.path.join(result_path, "ground_truth.csv"))

#normalize KKK_data
KKK_data = (KKK_data - KKK_data.mean()) / KKK_data.std()

#normalize true_df
# true_df = (true_df - true_df.mean()) / true_df.std()

plt.plot(range(begin, end), true_df.iloc[begin: end, channel], label='Ground Truth', linestyle='-', linewidth=2)
plt.plot(range(begin, end), KKK_data[channel-1, begin: end, 0], label='Interformer', linestyle='-', linewidth=1, alpha=0.5)
for model in models:

    with open(os.path.join(result_path, f"ECL_{model}_96.pkl"), "rb") as f:
        results = pickle.load(f)
        #print(results['pred_res'].shape)
    plt.plot(range(begin, end ), results['pred_res'][begin:end, channel-1], label=model, linestyle='--', linewidth=1, alpha=0.5)
plt.legend()
plt.savefig("results_plot.png")
plt.show()




