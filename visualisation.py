import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


scores = np.transpose([0.884, 0.854, 0.851])
df = pd.DataFrame(scores, index=['Random Forest', 'KNN', 'Logistic Regression'], columns=['Accuracy'])
ax = sns.lineplot(x=['Random Forest', 'KNN', 'Logistic Regression'], y='Accuracy', data=df)
ax.plot()
plt.show(block=True)