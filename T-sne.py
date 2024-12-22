import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# CSV dosyasını oku
filename = "C:\\Users\\ENES\\Desktop\\DiffProje1\\weights.csv"
data = pd.read_csv(filename)

# Epoch ve Run bilgilerini ayır
runs = data.iloc[:, 0].values  # İlk sütun: Run bilgisi
epochs = data.iloc[:, 1].values  # İkinci sütun: Epoch bilgisi
ağırlıklar = data.iloc[:, 2:].values  # Diğer sütunlar: Ağırlıklar

# T-SNE ile boyut indirgeme
tsne = TSNE(n_components=2, random_state=42)
ağırlıklar_2d = tsne.fit_transform(ağırlıklar)

# Her Run için farklı bir renk ve işaret belirle
unique_runs = np.unique(runs)
markers = ['o', 's', 'D', '^', 'v']  # Farklı işaretler
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_runs)))  # Renk skalası

plt.figure(figsize=(12, 8))

# Run'ları görselleştir
for i, run in enumerate(unique_runs):
    mask = runs == run
    plt.scatter(
        ağırlıklar_2d[mask, 0], ağırlıklar_2d[mask, 1],
        c=[colors[i]], label=f"Run {int(run)}", marker=markers[i % len(markers)], s=50
    )

plt.colorbar(label='Epoch')
plt.title("T-SNE Visualization of Weight Trajectories")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.show()