

# ------------------------------------------------------gini index

# import pandas as pd
# from part1 import Tree
# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt

# # Define the training dataset
# training_data = {
#     'Name': ['Tim', 'Joe', 'Sue', 'John', 'Mary', 'Fred', 'Pete', 'Jacob', 'Sofia'],
#     'Debt': ['low', 'high', 'low', 'medium', 'high', 'low', 'low', 'high', 'medium'],
#     'Income': ['low', 'high', 'high', 'low', 'low', 'low', 'medium', 'medium', 'low'],
#     'Married': [0, 1, 1, 0, 1, 1, 0, 1, 0],
#     'Owns_Property': [0, 1, 0, 0, 0, 0, 1, 1, 0],
#     'Gender': [0, 0, 1, 0, 1, 0, 0, 0, 1],
#     'Risk': [0, 0, 0, 2, 2, 2, 0, 0, 0]
# }

# # Create a DataFrame from the training data
# df_train = pd.DataFrame(training_data)

# # Convert categorical variables to numerical form
# df_train['Debt'] = df_train['Debt'].map({'low': 0, 'medium': 1, 'high': 2})
# df_train['Income'] = df_train['Income'].map({'low': 0, 'medium': 1, 'high': 2})

# # Define features and target variable
# features_train = ['Debt', 'Income', 'Married', 'Owns_Property', 'Gender']
# target_train = 'Risk'

# # Convert the trained decision tree to Tree class
# tree_root = Tree.train(df_train[features_train].values.T, df_train[target_train].values)

# # Visualize the decision tree without graphviz
# plt.figure(figsize=(12, 8))
# plot_tree(tree_root, feature_names=features_train, class_names=[str(i) for i in range(tree_root.p.max() + 1)], filled=True, rounded=True)
# plt.savefig('decision_tree.png')
# plt.show()

# # Display the path to the saved image file
# print("Decision Tree visualization saved as 'decision_tree.png'")

# # Define the test data for prediction
# test_data = {
#     'Name': ['Tom', 'Ana'],
#     'Debt': ['low', 'low'],
#     'Income': ['low', 'medium'],
#     'Married': [0, 1],
#     'Owns_Property': [1, 1],
#     'Gender': [0, 1]
# }

# # Create a DataFrame from the test data
# df_test = pd.DataFrame(test_data)

# # Convert test data to numerical form
# df_test['Debt'] = df_test['Debt'].map({'low': 0, 'medium': 1, 'high': 2})
# df_test['Income'] = df_test['Income'].map({'low': 0, 'medium': 1, 'high': 2})

# # Predict the credit risk for the test data using the decision tree from part1.py
# predicted_risk_test = Tree.predict(tree_root, df_test[features_train].values.T)

# # Display the predicted credit risk for Tom and Ana
# for i, name in enumerate(['Tom', 'Ana']):
#     print(f"{name}'s predicted credit risk: {predicted_risk_test[i]}")




# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier, export_text

# # Define the training dataset
# training_data = {
#     'Name': ['Tim', 'Joe', 'Sue', 'John', 'Mary', 'Fred', 'Pete', 'Jacob', 'Sofia'],
#     'Debt': ['low', 'high', 'low', 'medium', 'high', 'low', 'low', 'high', 'medium'],
#     'Income': ['low', 'high', 'high', 'low', 'low', 'low', 'medium', 'medium', 'low'],
#     'Married': [0, 1, 1, 0, 1, 1, 0, 1, 0],
#     'Owns_Property': [0, 1, 0, 0, 0, 0, 1, 1, 0],
#     'Gender': [0, 0, 1, 0, 1, 0, 0, 0, 1],
#     'Risk': [0, 0, 0, 2, 2, 2, 0, 0, 2]
# }

# # Create a DataFrame from the training data
# df_train = pd.DataFrame(training_data)

# # Convert categorical variables to numerical form
# df_train['Debt'] = df_train['Debt'].map({'low': 0, 'medium': 1, 'high': 2})
# df_train['Income'] = df_train['Income'].map({'low': 0, 'medium': 1, 'high': 2})

# # Define features and target variable
# features_train = ['Debt', 'Income', 'Married', 'Owns_Property', 'Gender']
# target_train = 'Risk'

# # Create and train the decision tree model
# tree_model = DecisionTreeClassifier()
# tree_model.fit(df_train[features_train], df_train[target_train])

# # Visualize the decision tree using scikit-learn's export_text
# tree_rules = export_text(tree_model, feature_names=features_train)
# print("Decision Tree Rules:")
# print(tree_rules)

# # Define the test data for prediction
# test_data = {
#     'Name': ['Tom', 'Ana'],
#     'Debt': ['low', 'low'],
#     'Income': ['low', 'medium'],
#     'Married': [0, 1],
#     'Owns_Property': [1, 1],
#     'Gender': [0, 1]
# }

# # Create a DataFrame from the test data
# df_test = pd.DataFrame(test_data)

# # Convert test data to numerical form
# df_test['Debt'] = df_test['Debt'].map({'low': 0, 'medium': 1, 'high': 2})
# df_test['Income'] = df_test['Income'].map({'low': 0, 'medium': 1, 'high': 2})

# # Predict the credit risk for the test data using the decision tree model
# predicted_risk_test = tree_model.predict(df_test[features_train])

# # Display the predicted credit risk for Tom and Ana
# for i, name in enumerate(['Tom', 'Ana']):
#     print(f"{name}'s predicted credit risk: {predicted_risk_test[i]}")

# -----------------------------------------------------------------------

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# Define the training dataset
training_data = {
    'Name': ['Tim', 'Joe', 'Sue', 'John', 'Mary', 'Fred', 'Pete', 'Jacob', 'Sofia'],
    'Debt': ['low', 'high', 'low', 'medium', 'high', 'low', 'low', 'high', 'medium'],
    'Income': ['low', 'high', 'high', 'low', 'low', 'low', 'medium', 'medium', 'low'],
    'Married': [0, 1, 1, 0, 1, 1, 0, 1, 0],
    'Owns_Property': [0, 1, 0, 0, 0, 0, 1, 1, 0],
    'Gender': [0, 0, 1, 0, 1, 0, 0, 0, 1],
    'Risk': [0, 0, 0, 2, 2, 2, 0, 0, 0]
}

# Create a DataFrame from the training data
df_train = pd.DataFrame(training_data)

# Convert categorical variables to numerical form
df_train['Debt'] = df_train['Debt'].map({'low': 0, 'medium': 1, 'high': 2})
df_train['Income'] = df_train['Income'].map({'low': 0, 'medium': 1, 'high': 2})

# Define features and target variable
features_train = ['Debt', 'Income', 'Married', 'Owns_Property', 'Gender']
target_train = 'Risk'

# Create and train the decision tree model with entropy as the criterion
tree_model = DecisionTreeClassifier(criterion='entropy')
tree_model.fit(df_train[features_train], df_train[target_train])

# Visualize the decision tree using scikit-learn's export_text
tree_rules = export_text(tree_model, feature_names=features_train)
print("Decision Tree Rules:")
print(tree_rules)

# Define the test data for prediction
test_data = {
    'Name': ['Tom', 'Ana'],
    'Debt': ['low', 'low'],
    'Income': ['low', 'medium'],
    'Married': [0, 1],
    'Owns_Property': [1, 1],
    'Gender': [0, 1]
}

# Create a DataFrame from the test data
df_test = pd.DataFrame(test_data)

# Convert test data to numerical form
df_test['Debt'] = df_test['Debt'].map({'low': 0, 'medium': 1, 'high': 2})
df_test['Income'] = df_test['Income'].map({'low': 0, 'medium': 1, 'high': 2})

# Predict the credit risk for the test data using the decision tree model
predicted_risk_test = tree_model.predict(df_test[features_train])

# Display the predicted credit risk for Tom and Ana
for i, name in enumerate(['Tom', 'Ana']):
    print(f"{name}'s predicted credit risk: {predicted_risk_test[i]}")
