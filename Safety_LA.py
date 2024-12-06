import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pydeck as pdk
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# URL to the LAPD crime data
DATA_URL = "https://data.lacity.org/api/views/2nrs-mtv8/rows.csv?accessType=DOWNLOAD"

def load_data():
    # Load data directly from the URL
    data = pd.read_csv(DATA_URL, encoding='utf-8', on_bad_lines='skip')
    
    # Remove any null characters from the dataframe
    data.replace({'\0': ''}, regex=True, inplace=True)
    
    # Rename columns for consistency
    data = data.rename(columns={'LAT': 'lat', 'LON': 'lon'})
    
    # Convert lat/lon to numeric and drop rows with invalid values
    data['lat'] = pd.to_numeric(data['lat'], errors='coerce')
    data['lon'] = pd.to_numeric(data['lon'], errors='coerce')
    data = data.dropna(subset=['lat', 'lon'])
    
    # Define LA bounding box and filter data within it
    lat_min, lat_max = 33.5, 34.5
    lon_min, lon_max = -118.8, -118.0
    data = data[(data['lat'] >= lat_min) & (data['lat'] <= lat_max) & 
                (data['lon'] >= lon_min) & (data['lon'] <= lon_max)]
    
    # Convert 'DATE OCC' to datetime
    data['DATE OCC'] = pd.to_datetime(data['DATE OCC'], errors='coerce')
    data = data[data['DATE OCC'] >= pd.to_datetime('2022-01-01')]
    
    return data

# Menu options
menu = ["Home", "Rank", "Visualization", "Filter", "About", "Contact"]
df = load_data()
# Sidebar menu with "Home" as the default page
if 'choice' not in st.session_state:
    st.session_state['choice'] = "Home"

choice = st.sidebar.selectbox("Menu", menu, index=menu.index(st.session_state['choice']))

if choice == "Home":
    st.title("Safety LA")
    st.subheader("LA Crime Data Exploration Dashboard")
    
    # Date range for filtering
    min_occurence = pd.to_datetime(df['DATE OCC']).min().date()
    max_occurence = pd.to_datetime(df['DATE OCC']).max().date()

    # Time Slider for filtering data based on time
    st.write("Select Time Range for the map")
    start_time, end_time = st.slider("Timeline", min_value=min_occurence, max_value=max_occurence, value=[min_occurence, max_occurence])
    filtered_data = df[(df['DATE OCC'] >= pd.to_datetime(start_time)) & (df['DATE OCC'] <= pd.to_datetime(end_time))]

    st.write(f"Filtering data between {start_time} and {end_time}")
    st.write(f"Data Points: {len(filtered_data)}")


    # Map visualization centered around Los Angeles
    st.map(filtered_data)

    # Descriptive data analysis
    st.subheader("Crime Statistics Overview")

    # Total number of crimes
    st.write(f"Total number of crimes: {len(filtered_data)}")

    # Crime counts by year and month
    df['Year'] = df['DATE OCC'].dt.year
    df['Month'] = df['DATE OCC'].dt.month
    crime_by_year = df['Year'].value_counts().sort_index()
    st.write("Crimes by Year:")
    st.bar_chart(crime_by_year)

    # Top 5 Most Common Crimes
    st.subheader("Top 5 Most Common Crimes")
    top_5_crimes = df['Crm Cd Desc'].value_counts().head(5).sort_values(ascending=False)
    st.bar_chart(top_5_crimes)

    # Top 5 Crimes Resulting in Deaths 
    st.subheader("Top Crimes Resulting in Death")
    # Identifying crimes involving death based on description keywords
    death_related_crimes = df[df['Crm Cd Desc'].str.contains('HOMICIDE|MANSLAUGHTER|MURDER|DEAD BODY', case=False, na=False)]
    top_5_death_crimes = death_related_crimes['Crm Cd Desc'].value_counts().head(5).sort_values(ascending=False)
    st.write("Top crimes involving death:")
    st.bar_chart(top_5_death_crimes)

    # Crime occurrences by victim age and gender
    st.subheader("Crime Occurrences by Victim Age and Gender")
    victim_age_gender = df[['Vict Age', 'Vict Sex']].dropna()
    age_bins = [0, 18, 30, 45, 60, 100]
    age_labels = ['0-18', '19-30', '31-45', '46-60', '60+']
    victim_age_gender['Age Group'] = pd.cut(victim_age_gender['Vict Age'], bins=age_bins, labels=age_labels)
    age_gender_distribution = victim_age_gender.groupby(['Age Group', 'Vict Sex']).size().unstack().sort_index()
    st.write("Victim age and gender distribution:")
    st.bar_chart(age_gender_distribution)

    # Box plot for Victim Age
    st.subheader("Victim Age Distribution")
    fig, ax = plt.subplots()
    victim_age_gender = df['Vict Age'].dropna()

    # Create the box plot
    ax.boxplot(victim_age_gender, vert=False)
    ax.set_title('Box Plot of Victim Age')
    ax.set_xlabel('Age')
    st.pyplot(fig)


    # Crimes by time of day - Pie Chart
    st.subheader("Crimes by Time of Day")
    df['Hour'] = df['TIME OCC'] // 100  # Extract the hour from TIME OCC (e.g., 1300 -> 13)
    time_of_day = pd.Categorical(
        pd.cut(df['Hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False),
        categories=['Morning', 'Afternoon', 'Evening', 'Night'],
        ordered=True
    )
    crime_by_time_of_day = time_of_day.value_counts().sort_index()

    # Create a pie chart using Matplotlib
    fig, ax = plt.subplots()
    ax.pie(
        crime_by_time_of_day,
        labels=crime_by_time_of_day.index,
        autopct='%1.1f%%',  # Display percentage
        startangle=90,
        counterclock=False
    )
    ax.set_title("Crime Distribution by Time of Day")
    st.pyplot(fig)

    # Crimes by day of the week - Heatmap
    st.subheader("Crimes by Day of the Week")
    df['Day of Week'] = pd.Categorical(
        df['DATE OCC'].dt.day_name(),
        categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        ordered=True
    )
    # Create a pivot table to prepare data for the heatmap
    day_hour_heatmap_data = df.pivot_table(
        index='Day of Week', 
        columns='Hour', 
        values='DR_NO',  # Use a unique identifier column to count occurrences
        aggfunc='count', 
        fill_value=0
    )

    # Create a heatmap using Seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(day_hour_heatmap_data, cmap='YlOrRd', ax=ax)
    ax.set_title("Heatmap of Crimes by Day of the Week and Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    st.pyplot(fig)

    # Crime occurrences by area
    st.subheader("Crime Occurrences by Area")
    top_5_areas = df['AREA NAME'].value_counts().head(5).sort_values(ascending=False)
    st.write("Top 5 areas with the highest crime rates:")
    st.bar_chart(top_5_areas)



elif choice == "Rank":
    # Crime Rankings Page
    st.title("Crime Rankings")
    st.write("Rank different crime types based on occurrence and rank areas based on crime frequency.")

    # Define columns for layout with equal widths
    col1, col2 = st.columns(2)  # Make columns evenly divided

    # Top 15 most common crime types
    with col1:
        st.subheader("Top 15 Most Common Crime Types")
        crime_type_counts = df['Crm Cd Desc'].value_counts().head(15).reset_index()
        crime_type_counts.columns = ['Crime Type', 'Count']
        st.dataframe(crime_type_counts)

    with col2:
        st.subheader("Crime Type Chart")
        # Sort the data for better visualization
        crime_type_counts_sorted = crime_type_counts.sort_values(by='Count', ascending=True)
        st.bar_chart(crime_type_counts_sorted.set_index('Crime Type')['Count'])

    # Define columns for the next section with equal widths
    col3, col4 = st.columns(2)  # Make columns evenly divided

    # Area rankings based on crime frequency
    with col3:
        st.subheader("Area Rankings by Crime Frequency")
        area_crime_counts = df['AREA NAME'].value_counts().reset_index()
        area_crime_counts.columns = ['Area', 'Count']
        st.dataframe(area_crime_counts)

    with col4:
        st.subheader("Area Crime Frequency Chart")
        # Sort the data for better visualization
        area_crime_counts_sorted = area_crime_counts.sort_values(by='Count', ascending=True)
        st.bar_chart(area_crime_counts_sorted.set_index('Area')['Count'])

    # Additional rankings for deeper insights
    col5, col6 = st.columns(2)

    # Ranking age groups most susceptible to crimes
    with col5:
        st.subheader("Victim Susceptibility to Crime by Age Group")
        age_bins = [0, 18, 30, 45, 60, 100]
        age_labels = ['0-18', '19-30', '31-45', '46-60', '60+']
        df['Age Group'] = pd.cut(df['Vict Age'], bins=age_bins, labels=age_labels)
        age_crime_counts = df['Age Group'].value_counts().reset_index()
        age_crime_counts.columns = ['Age Group', 'Count']
        st.dataframe(age_crime_counts)

    with col6:
        st.subheader("Age Group Crime Susceptibility Chart")
        # Sort the data for better visualization
        age_crime_counts_sorted = age_crime_counts.sort_values(by='Count', ascending=True)
        st.bar_chart(age_crime_counts_sorted.set_index('Age Group')['Count'])

    # Define columns for ranking crime trends over time
    col7, col8 = st.columns(2)

    # Ranking crimes by their trend over time
    with col7:
        st.subheader("Crime Trends Over Time")
        recent_months = df['DATE OCC'].dt.to_period('M').value_counts().sort_index().tail(6)
        recent_months = recent_months.reset_index()
        recent_months.columns = ['Month', 'Count']
        st.dataframe(recent_months)

    with col8:
        st.subheader("Crime Trends Chart (Last 6 Months)")
        st.line_chart(recent_months.set_index('Month')['Count'])

    # Define columns for ranking weapons used in crimes
    col9, col10 = st.columns(2)

    # Ranking most commonly used weapons in crimes
    with col9:
        st.subheader("Top 10 Most Common Weapons Used")
        weapon_counts = df['Weapon Desc'].value_counts().head(10).reset_index()
        weapon_counts.columns = ['Weapon', 'Count']
        st.dataframe(weapon_counts)

    with col10:
        st.subheader("Weapon Use Chart")
        st.bar_chart(weapon_counts.set_index('Weapon')['Count'])

        

elif choice == "Visualization":
    st.title("Visualization")
    st.write("Explore crime density across different areas of Los Angeles. "
             "Use the filters to focus on specific areas and time ranges to better understand crime trends.")

    # Sidebar selection for filtering areas
    area = st.sidebar.multiselect(
        "Select the Area:",
        options=df["AREA NAME"].unique(),
        default=["Central"]  # Set default to "Central"
    )

    # Filter data based on selected areas
    df = df[df['AREA NAME'].isin(area)]

    # Date range for filtering
    min_occurence = pd.to_datetime(df['DATE OCC']).min().date()
    max_occurence = pd.to_datetime(df['DATE OCC']).max().date()

    st.write("Select Time Range for the map")
    start_time, end_time = st.slider("Timeline", min_value=min_occurence, max_value=max_occurence, value=[min_occurence, max_occurence])
    df = df[(df['DATE OCC'] >= pd.to_datetime(start_time)) & (df['DATE OCC'] <= pd.to_datetime(end_time))]

    # Extract only longitude and latitude columns for mapping
    df_map = df[['lon', 'lat']]

    st.write(f"Data Points: {len(df_map)}")

    # Display map visualization using pydeck
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=df_map['lat'].mean(),
            longitude=df_map['lon'].mean(),
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'HexagonLayer',
                data=df_map,
                get_position='[lon, lat]',
                radius=100,
                elevation_scale=7,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=df_map,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=100,
            ),
        ],
    ))

elif choice == "Filter":
    st.title("Advanced Crime Data Filtering")
    st.subheader("Explore Detailed Crime Information")

    # Filter by Area
    st.subheader("Filter by Area")
    selected_areas = st.multiselect(
        'Select area(s) to explore (leave empty for all):',
        options=df['AREA NAME'].unique()
    )
    if selected_areas:
        df = df[df['AREA NAME'].isin(selected_areas)]

    # Filter by Date Range
    st.subheader("Filter by Date Range")
    min_date = df['DATE OCC'].min().date()
    max_date = df['DATE OCC'].max().date()
    start_date, end_date = st.date_input(
        'Select date range (leave as default for all):',
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    df = df[(df['DATE OCC'] >= pd.to_datetime(start_date)) & (df['DATE OCC'] <= pd.to_datetime(end_date))]

    # Filter by Crime Type
    st.subheader("Filter by Crime Type")
    selected_crime_types = st.multiselect(
        'Select crime type(s) (leave empty for all):',
        options=df['Crm Cd Desc'].unique()
    )
    if selected_crime_types:
        df = df[df['Crm Cd Desc'].isin(selected_crime_types)]

    # Filter by Victim Age
    st.subheader("Filter by Victim Age")
    min_age, max_age = st.slider(
        'Select victim age range (leave default for all):',
        min_value=int(df['Vict Age'].min()),
        max_value=int(df['Vict Age'].max()),
        value=(int(df['Vict Age'].min()), int(df['Vict Age'].max()))
    )
    df = df[(df['Vict Age'] >= min_age) & (df['Vict Age'] <= max_age)]

    # Filter by Victim Gender
    st.subheader("Filter by Victim Gender")
    selected_genders = st.multiselect(
        'Select victim gender(s) (leave empty for all):',
        options=df['Vict Sex'].dropna().unique()
    )
    if selected_genders:
        df = df[df['Vict Sex'].isin(selected_genders)]

    # Filter by Weapon Type
    st.subheader("Filter by Weapon Type")
    selected_weapon_types = st.multiselect(
        'Select weapon type(s) (leave empty for all):',
        options=df['Weapon Desc'].dropna().unique()
    )
    if selected_weapon_types:
        df = df[df['Weapon Desc'].isin(selected_weapon_types)]

    # Filter by Premise Description
    st.subheader("Filter by Premise Description")
    selected_premises = st.multiselect(
        'Select premise type(s) (leave empty for all):',
        options=df['Premis Desc'].dropna().unique()
    )
    if selected_premises:
        df = df[df['Premis Desc'].isin(selected_premises)]

    # Display Filtered Results
    st.subheader("Filtered Crime Data")
    st.write(f"Total Data Points After Filtering: {len(df)}")
    st.dataframe(df[['DATE OCC', 'TIME OCC', 'AREA NAME', 'Crm Cd Desc', 'Vict Age', 'Vict Sex', 'Weapon Desc', 'Premis Desc', 'LOCATION']])

elif choice == "About":
    st.title("About SafetyLA")

    st.subheader("Purpose")
    st.write(
        """
        SafetyLA is designed to help users, especially newcomers and visitors, explore and understand 
        the safety levels across various neighborhoods in Los Angeles. By leveraging real-time crime 
        data from the LAPD, our application provides a reliable resource for those looking to gain 
        insights into safety trends and risks in specific areas.
        """
    )

    st.subheader("App Features")
    st.write(
        """
        - **Crime Map:** Visualize crime hotspots and trends in different areas of Los Angeles with up-to-date data.
        - **Ranking & Analysis:** Rank crime types and areas by frequency to identify high-risk zones.
        - **Custom Filtering:** Tailor your search by selecting specific crime types, areas, and time periods to view targeted insights.
        - **Descriptive Statistics:** Get an overview of crime patterns, total occurrences, and key trends in an easy-to-understand format.
        """
    )

    st.subheader("Tech Stack & Process")
    st.write(
        """
        - **ETL Pipeline:** Automates the extraction of data from the LAPD database, transforms it into a 
          structured format, and loads it directly into our online database, ensuring real-time updates 
          without manual interventions.
        - **SQL Queries:** Utilizes advanced SQL queries to enable users to filter and explore data based 
          on various criteria like location, crime type, and date range.
        - **Streamlit & Python:** The application is built with Streamlit, providing an interactive user 
          interface, while Python handles data processing and backend operations.
        - **Data Visualization:** Uses Pydeck and other visualization libraries to render interactive maps 
          and charts, offering intuitive insights into crime statistics.
        """
    )

    st.subheader("Our Vision")
    st.write(
        """
        SafetyLA aims to empower users with data-driven insights, making Los Angeles a safer and more informed city for all.
        """
    )
    
elif choice == "Contact":
    image = Image.open('IMG_3435.jpg')

    # Set the title
    st.title("Contact")

    # Use columns for side-by-side layout (text on the left, image on the right)
    col1, col2 = st.columns([2, 1])  # Adjust column widths, more space for text

    # Add text in the left column
    with col1:
        st.write("""
        **Kailong Wang**  
        Data Scientist with a background in Economics & Data Science.  
        Passionate about applying advanced analytics to solve complex business problems.
        """)
        st.write("📧 Email: kailongwang1@gmail.com")

    # Add image in the right column
    with col2:
        st.image(image, caption="Kailong Wang", width=250)

