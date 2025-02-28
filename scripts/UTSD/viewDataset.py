import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from itertools import islice

utsd = datasets.load_dataset("thuml/utsd", 'UTSD-1G', split='train')
pytorch_utsd = utsd.with_format("torch")

print(pytorch_utsd[0]['target'][0])
for i, sample in enumerate(islice(pytorch_utsd, 500)):
    print(sample['target'][0])
    plt.plot(i, sample['target'][0], linestyle='none', marker='o', color='red', markersize=3)

plt.savefig('UTSD-1G.png')
