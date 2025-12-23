###Packages###
#basic packages
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

#Sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# BioPython
from Bio.SeqUtils.ProtParam import ProteinAnalysis

###Data pre-processing###
#read in data
read_file = pd.read_excel("https://zenodo.org/records/16894086/files/Antibody%20and%20Nanobody%20Design%20Dataset%20(ANDD).xlsx?download=1")

read_file.to_csv("ANDD.csv", index=None, header=True)

antibody = pd.DataFrame(pd.read_csv("ANDD.csv", low_memory=False))

#filter dataset to only show nanobody/antibody and antigen sequences, as well as the affinity of this interaction

filtered_antibody = antibody[["Ag_Seq", "Ab/Nano H_Chain AA", "Ab/Nano L_Chain AA", "Affinity_Kd(M)"]]

#change \ to NaN
filtered_antibody = filtered_antibody.replace('\\', np.nan)

#remove rows with NaN values
filtered_antibody = filtered_antibody.dropna()
#Remove alternative sequence option for the antigen to simplify analysis
filtered_antibody["Ag_Seq"] = filtered_antibody["Ag_Seq"].str.split("/").str[0]

#remove 'X' amino acid to allow for analysis with Biopython packages
filtered_antibody['Ag_Seq'] = (
    filtered_antibody['Ag_Seq'].astype(str).str.replace('X', '', regex=False)
)

filtered_antibody['Ab/Nano H_Chain AA'] = (
    filtered_antibody['Ab/Nano H_Chain AA'].astype(str).str.replace('X', '', regex=False)
)

filtered_antibody['Ab/Nano L_Chain AA'] = (
    filtered_antibody['Ab/Nano L_Chain AA'].astype(str).str.replace('X', '', regex=False)
)

#Change Affinity column to numeric
print(filtered_antibody['Affinity_Kd(M)'].dtype) 
filtered_antibody["Affinity_Kd(M)"] = pd.to_numeric(
    filtered_antibody["Affinity_Kd(M)"],
    errors="coerce") #replace any nonnumerical value with NaN
filtered_antibody = filtered_antibody.dropna().reset_index(drop=True) #remove NaN and reset index

#make sure that all Kd is above 0 and then convert Kd(M) to -logKd(M)
filtered_antibody["-log10(Kd[M])"] = np.where(
    filtered_antibody['Affinity_Kd(M)'] > 0,
    -np.log10(filtered_antibody['Affinity_Kd(M)']),
    np.nan
)

#Classify amino acid affinity as either low or high, removing intermediate affinity antibodies/nanobodies from dataset

print(filtered_antibody["-log10(Kd[M])"].median())

#Remove antibodies with intermediate affinity (dataset median -log10Kd= 4.946)

# Define thresholds
high_affinity = 5.946
low_affinity = 3.946

# Filter
filtered_antibody = filtered_antibody[
    (filtered_antibody["-log10(Kd[M])"] >= high_affinity) |
    (filtered_antibody["-log10(Kd[M])"] <= low_affinity)
].copy()

#Add classification
filtered_antibody['affinity_class'] = np.where(
    filtered_antibody['-log10(Kd[M])'] >= high_affinity,
    'high',
    'low'
)

###amino acid composition###
#This section was inspired by @gallo33henrique's notebook
tqdm.pandas() #enables progress monitoring

def aa_comp(seq):
    amino_acid_composition = {}
    amino_acids = list("ACDEFGHIKLMNPQRSTVWYU")
    length = len(seq)
    
    for aa in amino_acids:
        amino_acid_composition[aa] = round(float(seq.count(aa)) / len(seq)*100, 3)

    return amino_acid_composition

print("Computing amino acid composition (Ag_Seq)...")
comp_Ag_Seq = (
    filtered_antibody["Ag_Seq"]
    .progress_apply(aa_comp)
    .apply(pd.Series)
    .add_prefix("Ag_")
)

print("Computing amino acid composition (H_Chain)...")
comp_H_Chain = (
    filtered_antibody["Ab/Nano H_Chain AA"]
    .progress_apply(aa_comp)
    .apply(pd.Series)
    .add_prefix("HC_")
)

print("Computing amino acid composition (L_Chain)...")
comp_L_Chain = (
    filtered_antibody["Ab/Nano L_Chain AA"]
    .progress_apply(aa_comp)
    .apply(pd.Series)
    .add_prefix("LC_")
)
#Extract protein properties of amino acid sequence
def extract_properties(sequence):
    analysis = ProteinAnalysis(sequence)
    mw = analysis.molecular_weight()
    arom = analysis.aromaticity()
    iso = analysis.isoelectric_point()
    helix, turn, sheet = analysis.secondary_structure_fraction()

    return {
        "MolecularWeight": mw,
        "Aromaticity": arom,
        "IsoelectricPoint": iso,
        "Helix": helix,
        "Turn": turn,
        "Sheet": sheet,
    }

print("Extracting physicochemical properties (Ag_Seq)...")
prop_Ag_Seq = (
    filtered_antibody["Ag_Seq"]
    .progress_apply(extract_properties)
    .apply(pd.Series)
    .add_prefix("Ag_")
)

print("Extracting physicochemical properties (HC)...")
prop_H_Chain = (
    filtered_antibody["Ab/Nano H_Chain AA"]
    .progress_apply(extract_properties)
    .apply(pd.Series)
    .add_prefix("HC_")
)

print("Extracting physicochemical properties (LC)...")
prop_L_Chain = (
    filtered_antibody["Ab/Nano L_Chain AA"]
    .progress_apply(extract_properties)
    .apply(pd.Series)
    .add_prefix("LC_")
)
#merge protein properties and amino acid composition with dataframe
filtered_antibody = pd.concat(
    [filtered_antibody, comp_Ag_Seq, comp_H_Chain, comp_L_Chain, prop_Ag_Seq, 
    prop_H_Chain, prop_L_Chain],
    axis=1 #specify to add new columns not rows
) 

filtered_antibody.head()

###model building###

X=filtered_antibody.drop(columns=["Ag_Seq", "Ab/Nano H_Chain AA", "Ab/Nano L_Chain AA", "-log10(Kd[M])", "Affinity_Kd(M)", "affinity_class"])
y=filtered_antibody["affinity_class"]

#split into test group and training group
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

#evaluate model
pred_y = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred_y))
print(classification_report(y_test, pred_y, target_names=["high", "low"]))

#cross-validation
cv_scores = cross_val_score(RandomForestClassifier(random_state=42), 
                           X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# get feature importances
importances = rf.feature_importances_
# list of feature columns
cols = X.columns.tolist()
#combine into dataframe
feature_importance_df = pd.DataFrame({'Feature': cols, 'rf': importances})

# Sort by importance (descending order)
feature_importance_df = feature_importance_df.sort_values(by='rf', ascending=False)

# plot importances
plt.figure(figsize=(20, 12))
plt.bar(feature_importance_df['Feature'], feature_importance_df['rf'])
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.xticks(rotation=90)
plt.show()

#confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, pred_y)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['high', 'low'],
            yticklabels=['high', 'low'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
