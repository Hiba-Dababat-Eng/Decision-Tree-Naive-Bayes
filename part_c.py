import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ── Download Titanic dataset ──────────────────────────────────────────────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("=" * 45)
print("Dataset : Titanic (UCI / Kaggle)")
print(f"Instances : {df.shape[0]}")
print(f"Features  : {df.shape[1]}")
print("=" * 45)

# ── Preprocessing ─────────────────────────────────────────────────────────────
# خد بس الـ features المهمة
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()

# عبّي الـ missing values
df['Age'].fillna(df['Age'].median(), inplace=True)

# حوّل Sex لأرقام
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# ── Split ─────────────────────────────────────────────────────────────────────
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── Train ─────────────────────────────────────────────────────────────────────
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)

print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

# ── Confusion Matrix (pastel theme) ───────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

pastel_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
    'pastel', ['#ffffff', '#FFD1DC'])

fig, ax = plt.subplots(figsize=(7, 6), facecolor='#ffffff')
ax.set_facecolor('#ffffff')

ax.imshow(cm, cmap=pastel_cmap, vmin=0, vmax=cm.max() + 1)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Not Survived', 'Survived'], color='#000000', fontsize=11)
ax.set_yticklabels(['Not Survived', 'Survived'], color='#000000', fontsize=11)
ax.set_xlabel('Predicted', color='#000000', fontsize=12)
ax.set_ylabel('Actual',    color='#000000', fontsize=12)
ax.set_title('Titanic — Confusion Matrix', color='#000000', fontsize=13, pad=15)
ax.spines[:].set_color('#cccccc')

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                fontsize=22, fontweight='bold',
                color='#000000')

plt.tight_layout()
plt.savefig("uci_confusion_matrix.png", dpi=150, facecolor='#ffffff')
plt.show()
print("\nSaved → uci_confusion_matrix.png")