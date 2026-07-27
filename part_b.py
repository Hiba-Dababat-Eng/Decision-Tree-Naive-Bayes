import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor':   '#ffffff',
    'text.color':       '#000000',
    'axes.labelcolor':  '#000000',
    'xtick.color':      '#000000',
    'ytick.color':      '#000000',
    'axes.edgecolor':   '#cccccc',
})

data = {
    'Outlook':     ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast',
                    'Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild',
                    'Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity':    ['High','High','High','High','Normal','Normal','Normal','High',
                    'Normal','Normal','Normal','High','Normal','High'],
    'Wind':        ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak',
                    'Weak','Weak','Strong','Strong','Weak','Strong'],
    'Tennis':      ['No','No','Yes','Yes','Yes','No','Yes','No',
                    'Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)

X = df_encoded.drop('Tennis', axis=1)
y = df_encoded['Tennis']

dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
nb = GaussianNB()
dt.fit(X, y); nb.fit(X, y)

dt_pred = dt.predict(X)
nb_pred = nb.predict(X)

print("=" * 40)
print(f"Decision Tree Accuracy : {accuracy_score(y, dt_pred):.4f}")
print(f"Naive Bayes   Accuracy : {accuracy_score(y, nb_pred):.4f}")
print("=" * 40)
print("\n── Decision Tree Report ──")
print(classification_report(y, dt_pred, target_names=['No','Yes']))
print("\n── Naive Bayes Report ──")
print(classification_report(y, nb_pred, target_names=['No','Yes']))

# ── Pastel confusion matrices ─────────────────────────────────────────────────
pastel_cmaps = [
    plt.cm.colors.LinearSegmentedColormap.from_list('pink',   ['#ffffff', '#FFD1DC']),
    plt.cm.colors.LinearSegmentedColormap.from_list('yellow', ['#ffffff', '#FFFACD']),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor='#ffffff')
fig.suptitle("Confusion Matrix Comparison", color='#000000', fontsize=14)

for ax, pred, title, cmap in zip(axes,
                                   [dt_pred, nb_pred],
                                   ['Decision Tree', 'Naive Bayes'],
                                   pastel_cmaps):
    cm = confusion_matrix(y, pred)
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max() + 1)
    ax.set_facecolor('#ffffff')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'], color='#000000', fontsize=12)
    ax.set_yticklabels(['No', 'Yes'], color='#000000', fontsize=12)
    ax.set_xlabel('Predicted', color='#000000', fontsize=12)
    ax.set_ylabel('Actual',    color='#000000', fontsize=12)
    ax.set_title(title,        color='#000000', fontsize=13, pad=12)
    ax.spines[:].set_color('#cccccc')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    fontsize=22, fontweight='bold',
                    color='#000000')

plt.tight_layout()
plt.savefig("comparison_cm.png", dpi=150, facecolor='#ffffff')
plt.show()
print("Saved → comparison_cm.png")