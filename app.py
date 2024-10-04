import streamlit as st
import pandas as pd
import numpy as np
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# Indlæs data funktion med caching (Bruger ny version af cach))
@st.cache_data
def load_data():
    return pd.read_csv("kiva_loans.csv")

from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset
af = load_data()
af_usa = af[af['country'] == 'United States']

from sklearn.preprocessing import StandardScaler
import pandas as pd

# Funktion til at preprose dataen
def preprocess_data(df):
    af_usa = df[df['country'] == 'United States']
    af_usa = af_usa.dropna(subset=['loan_amount', 'term_in_months', 'lender_count'])
    
    # Definer target variablerne og features
    y = af_usa['lender_count']
    features = ['loan_amount', 'term_in_months']
    X = af_usa[features]
    
    # Opret en ny feature, der giver noget insidt i lånet og hvor meget der bliver tilbagebetalt.
    # Kort sagt hjælper loan amount per month modellen med at forstå "byrden" af lånet for låntageren, hvilket i sidste ende kan påvirke, hvor mange långivere der er
    X['loan_amount_per_month'] = X['loan_amount'] / X['term_in_months']
    
    # Standardiser features undtaget lender count
    scaler = StandardScaler()
    X[['loan_amount', 'term_in_months', 'loan_amount_per_month']] = scaler.fit_transform(X[['loan_amount', 'term_in_months', 'loan_amount_per_month']])
    
    return X, y

# Tilføj 'Introduktion' til navigation
st.sidebar.title("📊 Navigation")

# Update sidebar to include the new page "Lender Count Prediction"
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🌟 Introduction", "🏦 Loan Activities", "🤖 Machine Learning", "🔮 Recommender"])

# Side-navigation
if page == "🌟 Introduction":
    st.title('🎉 Welcome to US Loan Data Analysis 🇺🇸')
    st.subheader("🔍 Exploring loans by business activities and gender 💼")
    st.write("""
        📝 Here, you can explore different visualizations and statistical analysis that provide insights 
        into the loan patterns, gender distribution of loan recipients, and the different activities 
        for which the loans are used. 🚀 Navigate through the sidebar to explore specific analyses.
    """)
    st.image("DALL·E 2024-09-13 17.47.25 - A simple illustrative cover for a research paper. The background features the US flag, providing a patriotic theme. In the foreground, a woman and a m.webp", width=600)

# Låneaktivitetersiden
elif page == "🏦 Loan Activities":
    st.header("📊 Loan Activities Analysis")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Rengøring af data
    af = af.dropna(subset=['borrower_genders'])  # Clean missing borrower_gender rows
    top_genders = af['borrower_genders'].value_counts().nlargest(2).index
    af = af[af['borrower_genders'].isin(top_genders)]  # Keep only the top 2 genders

    af_usa = af[af['country'] == 'United States']  # Filter for USA data

    # Definér activity_counts
    activity_counts = af_usa['activity'].value_counts().head(10)

    # Opret knapper for at skifte mellem visninger
    col1, col2 = st.columns(2)
    show_all_loans = col1.button("Show All Loans")
    show_smaller_loans = col2.button("Show Smaller Loans (Under Q1)")

    if show_smaller_loans or not show_all_loans:
        # Hvis 'Vis mindre lån' er valgt eller ingen knap er valgt
        st.subheader("📉 Loans Under Q1 (Lower 25%)")
        minimum = af_usa['loan_amount'].min()
        q1 = af_usa['loan_amount'].quantile(0.25)
        af_filtered_usa = af_usa[(af_usa['loan_amount'] >= minimum) & (af_usa['loan_amount'] <= q1)]

        # Sortér låneaktiviteter for mindre lån i faldende rækkefølge
        activity_counts_underq1 = af_filtered_usa['activity'].value_counts().sort_values(ascending=False)
        top_activities_under_q1 = activity_counts_underq1.head(10)

        # Plot lånebeløb distribution for alle lån
        fig, ax = plt.subplots()
        sns.boxplot(data=af_filtered_usa, y='loan_amount', ax=ax)
        st.pyplot(fig)

        st.subheader("🏅 Top 10 Loan Activities for Small Loans (Under Q1)")

        # Brug seaborn eller matplotlib for at lave det sorterede søjlediagram
        fig, ax = plt.subplots()
        sns.barplot(x=top_activities_under_q1.values, y=top_activities_under_q1.index, ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("Activity")
        ax.set_title("Top 10 Loan Activities for Small Loans (Under Q1)")
        st.pyplot(fig)

        with st.expander("🇺🇸 Key Insights"):
            st.markdown("""
            📊 For the smaller loans (under Q1), the median loan is around $1,500, with loan amounts ranging between $500 and $2,500. 
            🏆 **Services** remains the top activity for small loans, followed by **Clothing** and **Cosmetics Sales**.
            """)
    
    elif show_all_loans:
        st.subheader("💵 Loan Amount Distribution in the USA")

         # Sortér låneaktiviteter i faldende rækkefølge
        minimum = af_usa['loan_amount'].min()
        q1 = af_usa['loan_amount'].quantile(0.25)
        af_filtered_usa = af_usa[(af_usa['loan_amount'] >= minimum) & (af_usa['loan_amount'] <= q1)]
        activity_counts = af_usa['activity'].value_counts().sort_values(ascending=False).head(10)
        activity_counts_underq1 = af_filtered_usa['activity'].value_counts().sort_values(ascending=False)
        top_activities_under_q1 = activity_counts_underq1.head(10)

        # Plot lånebeløb distribution for alle lån
        fig, ax = plt.subplots()
        sns.boxplot(data=af_usa, y='loan_amount', ax=ax)
        st.pyplot(fig)

        st.subheader("🏆 Top 10 Loan Activities in the USA")

         # Brug seaborn eller matplotlib for at lave det sorterede søjlediagram
        fig, ax = plt.subplots()
        sns.barplot(x=activity_counts.values, y=activity_counts.index, ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("Activity")
        ax.set_title("Top 10 Loan Activities in the USA")
        st.pyplot(fig)

        # Sammenligning af låneaktiviteter
        comparison_df = pd.DataFrame({'all_loans': activity_counts, 'loans_under_q1': top_activities_under_q1})
        comparison_df['percentage_small_loans'] = (comparison_df['loans_under_q1'] / comparison_df['all_loans']) * 100
        comparison_df_sorted = comparison_df.sort_values(by='percentage_small_loans', ascending=False)

        st.subheader("📊 Percentage of Small Loans (Q1) by Sector")

        # Opdateret søjlediagram for percentage_small_loans
        fig, ax = plt.subplots()
        sns.barplot(x=comparison_df_sorted['percentage_small_loans'].values, y=comparison_df_sorted.index, ax=ax)
        ax.set_xlabel("Percentage of Small Loans")
        ax.set_ylabel("Activity")
        ax.set_title("Percentage of Small Loans (Q1) by Sector")
        st.pyplot(fig)

        with st.expander("🇺🇸 Key Insights"):
            st.markdown("""
            📊 In the first boxplot, we see that the median loan is around $5,000, with loans ranging from $2,000 to $10,000. 
            🏆 The bar chart shows that **Services** is the most funded business activity, followed by **Food Production/Sales**.
            """)

# Klyngedannelse og Anbefalinger
elif page == "🤖 Machine Learning":
    st.header("🤖 Maskinlæring: Klyngedannelse & Recommender")

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from scipy.cluster.hierarchy import linkage, dendrogram
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Vi vælger kun data fra USA
    af_usa = af[af['country'] == 'United States']

    # Filtrering af outliers (lån større end 50.000)
    outlier_threshold = 50000
    af_usa = af_usa[af_usa['loan_amount'] <= outlier_threshold]

    # Databehandling (kun nødvendige kolonner)
    af_usa = af_usa.dropna(subset=['loan_amount', 'term_in_months', 'lender_count', 'sector'])
    features = af_usa[['loan_amount', 'term_in_months', 'lender_count']]

    # Label Encoding for at kode 'sector' variablen til numre
    le = LabelEncoder()
    af_usa['sector_encoded'] = le.fit_transform(af_usa['sector'])

    # Normalisering af dataen
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # PCA for at reducere datasættet til 2 komponenter
    # Vi bruger PCA til at forenkle dataen, mens vi bevarer variansen.
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)

    # Elbow-metoden for at finde det optimale antal af klynger
    st.subheader("🔷 Elbow-metoden for at finde det optimale antal klynger")
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)

    # Plot af elbow kurve
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), sse, marker='o')
    ax.set_title("Elbow-metoden")
    ax.set_xlabel("Antal klynger")
    ax.set_ylabel("Sum af kvadratiske fejl (SSE)")
    st.pyplot(fig)

    # K-Means clustering med PCA
    st.subheader("🔷 K-Means Klyngedannelse")
    kmeans = KMeans(n_clusters=3, random_state=42)
    af_usa['cluster_kmeans'] = kmeans.fit_predict(pca_features)

    # Plot af K-Means clustering med PCA og specifikke sektorer
    fig, ax = plt.subplots()

    # Filter for Sektor 1: Kosmetik, Underholdning, Tøj
    sector1_filter = af_usa['sector'].isin(['Cosmetics Sales', 'Entertainment', 'Clothing'])
    af_usa_sector1 = af_usa[sector1_filter]

    # Filter for Sektor 2: Services, Madpruduktion/Salg, Restaurant
    sector2_filter = af_usa['sector'].isin(['Services', 'Food Production/Sales', 'Restaurant'])
    af_usa_sector2 = af_usa[sector2_filter]

    # Plot clustering af hele datasættet og fremhæv de specifikke sektorer
    sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=af_usa['cluster_kmeans'], palette='Set2', ax=ax)

    # Beregn gennemsnitspunktet for Sektor 1
    sector1_means = af_usa_sector1[['loan_amount', 'term_in_months', 'lender_count']].mean()
    sector1_means_scaled = scaler.transform([sector1_means])  # Normaliserer gennemsnittet
    sector1_means_pca = pca.transform(sector1_means_scaled)  # Reducer med PCA

    # Beregn gennemsnitspunktet for Sektor 2
    sector2_means = af_usa_sector2[['loan_amount', 'term_in_months', 'lender_count']].mean()
    sector2_means_scaled = scaler.transform([sector2_means])  # Normaliserer gennemsnittet
    sector2_means_pca = pca.transform(sector2_means_scaled)  # Reducer med PCA

    # Tilføj et rødt punkt for gennemsnitspunktet for Sektor 1
    ax.scatter(sector1_means_pca[0, 0], sector1_means_pca[0, 1], color='red', s=200, label='Gennemsnit (Sektor 1)', marker='X')

    # Tilføj et blåt punkt for gennemsnitspunktet for Sektor 2
    ax.scatter(sector2_means_pca[0, 0], sector2_means_pca[0, 1], color='blue', s=200, label='Gennemsnit (Sektor 2)', marker='X')

    ax.set_title("K-Means Klynger med PCA (Sektor 1 vs Sektor 2) og gennemsnitspunkt")
    ax.legend()
    st.pyplot(fig)

    with st.expander("K-means Nøgleindsigter"):
        st.markdown("""
        📊 Klyngerne viser et punkt for hver lån. Punkterne er baseret på lånebeløb, lånetider og antallet af långivere. 
        Gennemsnitslånene for Sektor 1 (Kosmetik, Underholdning, Tøj) og Sektor 2 (Services, Madproduktion/Salg, Restaurant) er plottet for at give indsigt i deres position i forhold til andre sektorer.
        """)
    
    # Hierarkisk Clustering på et sample af USA data
    st.subheader("🔷 Hierarkisk Clustering")
    af_sample = af_usa.sample(n=1000, random_state=42)
    Z = linkage(af_sample[['loan_amount', 'term_in_months', 'lender_count']], 'ward')
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, truncate_mode='level', p=5, ax=ax)
    ax.set_title("Hierarkisk Clustering Dendrogram (USA Data)")
    st.pyplot(fig)

    # Beregn variance og standardafvigelse for Lender Count og Loan Term på tværs af sektorer
    st.subheader("📊 Variance og Standardafvigelse for Lender Count og Loan Term")

    # Filtrer for relevante sektorer
    selected_sectors = ['Cosmetics Sales', 'Entertainment', 'Clothing', 'Services', 'Food Production/Sales', 'Restaurant']
    af_selected = af_usa[af_usa['sector'].isin(selected_sectors)]

    # Gruppér data efter sektor og beregn variance og standardafvigelse
    sector_stats = af_selected.groupby('sector').agg({
        'lender_count': ['var', 'std', 'mean'],
        'term_in_months': ['var', 'std', 'mean']
    })

    # Omdøb kolonnerne for bedre læsbarhed
    sector_stats.columns = ['Lender Count Variance', 'Lender Count Std', 'Lender Count Mean',
                            'Term Length Variance', 'Term Length Std', 'Term Length Mean']

    # Vis resultaterne i Streamlit
    st.write("### Variance og Standardafvigelse efter Sektor")
    st.write(sector_stats)

    st.write("""
        - **Lender Count Variance: Måler hvor meget antallet af långivere varierer inden for hver sektor.
        - **Term Length Variance: Måler variationen i lånetid for hver sektor.
        - **Standard Deviation (Std): Viser spredningen af data omkring gennemsnittet.
    """)
    
# Supervised Learning
elif page == "🔮 Recommender":
    st.header("📊 XGBoost Regression Model & SHAP Explainability")
    
    import matplotlib.pyplot as plt
    import shap
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from xgboost import XGBRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import pandas as pd
    import re
    import string

    # Definer funktionen til at forbehandle tekst
    def preprocess_text(text):
        if isinstance(text, str):
            # Converter til små bogstaver
            text = text.lower()
            # Fjern tegnsætning
            text = text.translate(str.maketrans('', '', string.punctuation))
            # Fjern tal
            text = re.sub(r'\d+', '', text)
            # Fjern ekstra mellemrum
            text = ' '.join(text.split())
        else:
            # Hvis der ikke er tekst i stringen
            text = ''
        return text

    # Sikr at kolonnen 'use_cleaned' eksisterer
    if 'use_cleaned' not in af_usa.columns:
        # Antager at kolonnen 'use' eksisterer og skal rengøre
        af_usa['use_cleaned'] = af_usa['use'].apply(preprocess_text)
    #Brug TfidfVectorizer til at konvertere tekstdata til funktione
    vectorizer = TfidfVectorizer(max_features=50)
    tfidf_matrix = vectorizer.fit_transform(af_usa['use_cleaned'])
    # Opret en ny kolonne der angiver, om lån har høj sandsynlighed (1) eller lav sandsynlighed (0)
    af_usa['high_chance'] = np.where(af_usa['lender_count'] >= af_usa['lender_count'].median(), 1, 0)
    # Definer funktionerne og målene til den superviserede model
    X_supervised = pd.DataFrame(tfidf_matrix.toarray())
    y_supervised = af_usa['high_chance']
    # Split data i trænings- og testdata
    X_train, X_test, y_train, y_test = train_test_split(X_supervised, y_supervised, test_size=0.2, random_state=42)
    # Vi bruger logistisk regression model
    classifier = LogisticRegression(random_state=42)
    classifier.fit(X_train, y_train)

    # Funktion til at anbefale baseret på nøgleord
    def recommend_chance_supervised(use_description):
        use_description_cleaned = preprocess_text(use_description)
        input_tfidf = vectorizer.transform([use_description_cleaned]).toarray()
        predicted_chance = classifier.predict(input_tfidf)[0]
        return "High Chance" if predicted_chance == 1 else "Low Chance"
    # Brugerinput til beskrivelse af lånet
    st.subheader("Predict Loan Chance Based on Description")
    use_description_input = st.text_area("Enter loan use description:")
    # Brugerinput til lånebeløb
    loan_input = st.number_input("Enter loan amount:", min_value=int(af_usa['loan_amount'].min()), max_value=int(af_usa['loan_amount'].max()))

    if st.button("Predict Loan Chance and Find Similar Loans"):
        if use_description_input:
            predicted_chance = recommend_chance_supervised(use_description_input)
            st.write(f"Loan Chance Prediction: **{predicted_chance}**")
            
            # Anbefaling baseret på K-Means klyngedannelse
            st.subheader("🔶 Recommender")
            
            # Definer og tilpas scaler
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(af_usa[['loan_amount', 'term_in_months', 'lender_count']])
            
            # Definer og tilpas PCA
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(scaled_features)
            
            # Definer og tilpas KMeans model
            kmeans = KMeans(n_clusters=5, random_state=42)
            af_usa['cluster_kmeans'] = kmeans.fit_predict(pca_features)
            
            # Find nærmeste klynge baseret på inputdata
            input_data = [[loan_input, af_usa['term_in_months'].mean(), af_usa['lender_count'].mean()]]
            input_scaled = scaler.transform(input_data)
            input_pca = pca.transform(input_scaled)
            input_cluster = kmeans.predict(input_pca)[0]
            
            # Anbefalinger fra samme klynge
            recommended_loans = af_usa[af_usa['cluster_kmeans'] == input_cluster].head(5)
            st.write("📌 Loans similar to your input (USA Data):")
            st.write(recommended_loans[['loan_amount', 'term_in_months', 'lender_count', 'sector', 'activity']])
        else:
            st.error("Please enter a loan description.")