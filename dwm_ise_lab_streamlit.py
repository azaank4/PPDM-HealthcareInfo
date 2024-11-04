import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from deap import base, creator, tools, algorithms
import random
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("Heart Disease Association Rule Mining")
    st.write("Upload your CSV file and analyze association rules")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Load and process data
        df = pd.read_csv(uploaded_file)
        
        # Display raw data
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        # Data preprocessing
        st.subheader("Data Preprocessing")
        with st.expander("View preprocessing steps"):
            # Discretize 'age' and 'chol' into categories
            df['age_group'] = pd.cut(df['age'], bins=[20, 40, 60, 80], labels=['young', 'middle', 'old'])
            df['chol_level'] = pd.cut(df['chol'], bins=[0, 200, 400], labels=['low', 'high'])
            
            # Select and one-hot encode categorical columns
            df_encoded = pd.get_dummies(df[['age_group', 'chol_level', 'target']])
            st.write("Encoded Data Preview:")
            st.dataframe(df_encoded.head())

        # Association Rules Parameters
        st.subheader("Association Rules Parameters")
        min_support = st.slider("Minimum Support", 0.1, 1.0, 0.1, 0.1)
        min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.1)

        if st.button("Generate Association Rules"):
            with st.spinner("Generating rules..."):
                # Generate frequent itemsets and rules
                frequent_itemsets = apriori(df_encoded, 
                                         min_support=min_support, 
                                         use_colnames=True)
                rules = association_rules(frequent_itemsets, 
                                        metric="confidence", 
                                        min_threshold=min_confidence)

                # Display rules
                st.subheader("Association Rules")
                st.dataframe(rules[['antecedents', 'consequents', 'support', 
                                  'confidence', 'lift']])

                # Visualizations
                st.subheader("Visualizations")
                
                # Network Graph
                with st.expander("View Network Graph"):
                    fig_network, ax = plt.subplots(figsize=(12, 8))
                    G = nx.from_pandas_edgelist(rules, 'antecedents', 'consequents', 
                                              ['confidence'])
                    nx.draw(G, with_labels=True, node_size=3000, 
                           node_color='skyblue', font_size=10, 
                           font_color='black', edge_color='gray')
                    st.pyplot(fig_network)

                # Heatmap
                with st.expander("View Support Heatmap"):
                    fig_heatmap, ax = plt.subplots(figsize=(10, 6))
                    pivot_table = rules.pivot(index='antecedents', 
                                            columns='consequents', 
                                            values='support').fillna(0)
                    sns.heatmap(pivot_table, cmap='Blues', annot=True)
                    plt.title("Heatmap of Rule Support")
                    st.pyplot(fig_heatmap)

                # Download options
                st.subheader("Download Results")
                csv = rules.to_csv(index=False)
                st.download_button(
                    label="Download Rules as CSV",
                    data=csv,
                    file_name="association_rules.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    # Clear any existing DEAP creators to avoid conflicts
    if 'FitnessMin' in creator.__dict__:
        del creator.FitnessMin
    if 'Individual' in creator.__dict__:
        del creator.Individual
        
    main() 