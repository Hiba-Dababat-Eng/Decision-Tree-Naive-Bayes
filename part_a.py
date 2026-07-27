import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------
# Dataset
# --------------------------------------------------
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

# --------------------------------------------------
# Encode categorical variables
# --------------------------------------------------
df_encoded = df.copy()

for col in df.columns:
    encoder = LabelEncoder()
    df_encoded[col] = encoder.fit_transform(df[col])

X = df_encoded.drop('Tennis', axis=1)
y = df_encoded['Tennis']

# --------------------------------------------------
# Train Decision Tree
# --------------------------------------------------
clf = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)

clf.fit(X, y)

# --------------------------------------------------
# Print Rules
# --------------------------------------------------
print("\n── Decision Tree Rules ──\n")
print(export_text(clf, feature_names=list(X.columns)))

# --------------------------------------------------
# Plot Tree
# --------------------------------------------------
fig, ax = plt.subplots(
    figsize=(15, 8),
    facecolor='white'
)

ax.set_facecolor('white')

plot_tree(
    clf,
    feature_names=list(X.columns),
    class_names=['No', 'Yes'],
    filled=False,          # مهم
    rounded=True,
    fontsize=11,
    ax=ax
)

# --------------------------------------------------
# Pastel Colors
# --------------------------------------------------
pastel_colors = [
    '#FFD6E7',  # Light Pink
    '#FFF4B8',  # Light Yellow
    '#E8D9FF'   # Light Lavender
]

# --------------------------------------------------
# Color Nodes
# --------------------------------------------------
color_idx = 0

for text in ax.texts:

    text.set_color('black')
    text.set_fontsize(10)

    bbox = text.get_bbox_patch()

    if bbox is not None:
        bbox.set_facecolor(
            pastel_colors[color_idx % len(pastel_colors)]
        )
        bbox.set_edgecolor('#BBBBBB')
        bbox.set_linewidth(1.5)

        color_idx += 1

# --------------------------------------------------
# Style Lines
# --------------------------------------------------
for line in ax.lines:
    line.set_color('#999999')
    line.set_linewidth(1.2)

# --------------------------------------------------
# Title
# --------------------------------------------------
ax.set_title(
    "PlayTennis — Decision Tree",
    fontsize=18,
    color='black',
    pad=20
)

plt.tight_layout()

# --------------------------------------------------
# Save Figure
# --------------------------------------------------
plt.savefig(
    "decision_tree.png",
    dpi=300,
    facecolor='white',
    bbox_inches='tight'
)

plt.show()

print("\nSaved → decision_tree.png")

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
y_pred = clf.predict(X)

print("\nAccuracy:", clf.score(X, y))

print("\nClassification Report:\n")
print(
    classification_report(
        y,
        y_pred,
        target_names=['No', 'Yes']
    )
)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y, y_pred))