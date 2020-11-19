import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


scores = np.transpose([0.88, 0.78, 0.86])
df = pd.DataFrame(scores, index=['Random Forest', 'SVM', 'KNN'], columns=['Accuracy'])
ax = sns.lineplot(x=['Random Forest', 'SVM', 'KNN'], y='Accuracy', data=df)
ax.plot()
plt.show(block=True)