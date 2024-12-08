import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 

##############################################################

from sklearn.preprocessing import LabelEncoder,PolynomialFeatures,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error,confusion_matrix,accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# color plate
colors10 = ['#387478', '#4682B4', '#32CD32', '#FFD700','#001F3F','#B17457','#F2E5BF','#DA8359','#FFD09B','#A66E38']  # You can define your own colors
blue_1=['#2D4356', '#435B66', '#A76F6F', '#EAB2A0']
blue_2=['#0C134F', '#1D267D', '#2D4263', '#347474']
green1=['#1A1A19', '#31511E', '#859F3D', '#88C273']
brown1=['#A79277', '#D1BB9E', '#EAD8C0', '#FFF2E1']
yel_gre1=['#F3CA52', '#F6E9B2', '#0A6847', '#7ABA78']
red_tel=['#C96868', '#FADFA1', '#FFF4EA', '#7EACB5']
cofee=['#EAC696', '#C8AE7D', '#765827', '#65451F']
pastel=['#B5C0D0', '#CCD3CA', '#B4B4B8', '#B3A398']
retro=['#060047', '#B3005E', '#E90064', '#FF5F9E']
white_blue=['#04009A', '#77ACF1', '#77ACF1', '#C0FEFC']
cold_blue=['#240750', '#344C64', '#577B8D', '#57A6A1']
cold_green=['#006769', '#40A578', '#9DDE8B', '#E6FF94']
happy=['#D2E0FB', '#F9F3CC', '#D7E5CA', '#8EACCD']
sky=['#00A9FF', '#89CFF3', '#A0E9FF', '#CDF5FD']
grad_brown=['#8D7B68', '#A4907C', '#C8B6A6', '#F1DEC9']
grad_black=['#2C3333', '#2E4F4F', '#0E8388', '#CBE4DE']
grad_green=['#439A97', '#62B6B7', '#97DECE', '#CBEDD5']
grad_blue=['#164863', '#427D9D', '#9BBEC8', '#DDF2FD']
night=['#003C43', '#135D66', '#77B0AA', '#E3FEF7']



#################################################################################

# Read Data
df=pd.read_csv('retail.csv')

# Convert the date columns to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Extract day, month, and year from 'Order Date'
df['Order Day'] = df['Order Date'].dt.day
df['Order Month'] = df['Order Date'].dt.month
df['Order Year'] = df['Order Date'].dt.year

# Extract day, month, and year from 'Ship Date'
df['Ship Day'] = df['Ship Date'].dt.day
df['Ship Month'] = df['Ship Date'].dt.month
df['Ship Year'] = df['Ship Date'].dt.year

# Configure page
st.set_page_config(page_title="Sales Dashboard",page_icon=":bar_chart:",layout='wide',initial_sidebar_state="expanded")

# Apply custom CSS for background color, font type, and font color

st.sidebar.image("kpi.png",caption="sales analysis")

st.sidebar.header("please Filter here :")


# Sidebar multiselect filters
Category = st.sidebar.multiselect(
    "Select the Category:",
    options=df['Category'].unique(),
    default=df['Category'].unique()
)

Sub_Category = st.sidebar.multiselect(
    "Select the Sub-Category:",
    options=df['Sub_Category'].unique(),
    default=df['Sub_Category'].unique()
)

Segment = st.sidebar.multiselect(
    "Select the Segment:",
    options=df['Segment'].unique(),
    default=df['Segment'].unique()
)
 

# Apply custom CSS for multiselect, sidebar styling, and new elements
st.markdown(
    """
    <style>
    /* Main background color */
    .main {
        background-color: #00172B;
    }

    /* Sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #063970;
        color: blue;
    }

    /* Sidebar content color */
    .sidebar-content {
        color: black;
    }

    /* Multiselect styling */
    .stMultiSelect .css-1wa3eu0 {
        background-color: #063970 !important; /* Background color for the multiselect */
        color: #063970 !important; /* Text color inside the multiselect */
    }

    /* Multiselect selected items color */
    .stMultiSelect .css-1n76uvr .css-1vbd788 {
        background-color: #063970 !important; /* Selected option background */
        color: black !important; /* Selected option text color */
    }

    /* Sidebar multiselect label color */
    .stSidebar .css-10trblm, .stSidebar .css-1d391kg {
        color: #063970 !important; /* Label color for multiselect options */
    }

    /* Box shadow for options */
    .stSidebar .css-2b097c-container {
        box-shadow: 0 0 2px #686664;
    }

    /* Metric container styling */
    [data-testid="metric-container"] {
        box-shadow: 0 0 2px #686664;
        padding: 5px;
    }

    /* Plot container styling */
    .plot-container > div {
        box-shadow: 0 0 2px #686664;
        padding: 5px;
    }

    /* Expander button styling */
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.3rem;
        color: #686664;
    }

    /* Metric container styling */
    [data-testid="metric-container"] {
        box-shadow: 0 0 2px #686664;
        padding: 5px;
    }

    /* Plot container styling */
    .plot-container > div {
        box-shadow: 0 0 2px #686664;
        padding: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Filtering the data based on sidebar selection
df_selection = df.query("Category == @Category & Sub_Category == @Sub_Category & Segment == @Segment")

# Center the title using HTML and CSS
st.title(":bar_chart: Sales Dashboard")
st.markdown("##")



st.markdown(
    "<h6 style='color: brown;'>This dataset is a sample of sales transactions from a retail store. It includes details of each order’s <b>date</b>, <b>shipping method</b>, <b>customer information</b> (name, location), <b>product details</b> (name, category), and <b>sales metrics</b> like quantity, discount, and profit.</h6>",
    unsafe_allow_html=True
)


# Display the dataframe
st.dataframe(df_selection)

#################################################################################


# Main Page 

# total kpi's 

def Home():        
    total_quantity=int(df_selection["Quantity"].sum())
    total_sales =int(df_selection["Sales"].sum())
    avg_sales=round(df_selection["Sales"].mean(),2)
    max_sale=int(df_selection["Sales"].max())
    min_sale=round(df_selection["Sales"].min(),2)


    total1,total2,total3,total4,total5=st.columns(5)

    with total1:
        st.info("Total Sales",icon="📌")
        st.metric(label="Total sales",value=f"{total_sales:,.0f}")
        

    with total2:
        st.info("total_quantity",icon="📌")
        st.metric(label="total_quantity",value=f"{total_quantity:,.0f}")
        
    with total3:
        st.info("avg_sales",icon="📌")
        st.metric(label="avg_sales",value=f"{avg_sales:,.0f}")
        
    with total4:
        st.info("max_sale",icon="📌")
        st.metric(label="max_sale",value=f"{max_sale:,.0f}" )
        
    with total5:
        st.info("min_sale",icon="📌")
        st.metric(label="min_sale",value=numerize(min_sale) )        
        
    st.markdown("---")

Home()

st.markdown("##")


#################################################################################


sales_per_day=df_selection.groupby(by=["Order Day"])['Sales'].sum()
fig_daily_sales=px.bar(sales_per_day,
           x=sales_per_day.index,y='Sales',
           title="<b>sales_per_order_day</b>",
           color_discrete_sequence=["#00A9FF"]*len(sales_per_day),
           template="plotly_white")

fig_daily_sales.update_layout(
    xaxis=dict(tickmode="linear"),
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=(dict(showgrid=False))
)



sales_per_shipday=df_selection.groupby(by=["Ship Day"])['Sales'].sum()
fig_dailyship_sales=px.bar(sales_per_shipday,
           x=sales_per_shipday.index,y='Sales',
           title="<b>sales_per_ship_day</b>",
           color_discrete_sequence=["#89CFF3"]*len(sales_per_shipday),
           template="plotly_white")

fig_daily_sales.update_layout(
    xaxis=dict(tickmode="linear"),
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=(dict(showgrid=False))
)


plot1,plot2=st.columns(2)
plot1.plotly_chart(fig_daily_sales,use_container_width=True)
plot2.plotly_chart(fig_dailyship_sales,use_container_width=True)



st.markdown("---")

#################################################################################

# plot 1 

# Distribution per day for each order
st.header("The Distribution Per Day for Each Order")
st.markdown(
    "<h4 style='color: green;'>It appears that <b>day 21</b> and <b>day 22</b> have the highest Count, while <span style='color: red;'>day 31 has the lowest</span>.</h4>",
    unsafe_allow_html=True
)

order_day_counts = df_selection['Order Day'].value_counts().reset_index()
order_day_counts.columns = ['Order Day', 'count']

# Create the bar chart with custom colors
fig_daily_count = px.bar(
    order_day_counts,
    x='Order Day',
    y='count',
    color='Order Day',
    color_discrete_sequence=cold_green
)

fig_daily_count.update_layout(
    xaxis=dict(tickmode="linear"),
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=(dict(showgrid=False))
)

# Show plot in Streamlit
st.plotly_chart(fig_daily_count)



st.markdown("---")

#################################################################################


# plot 2
st.header("The Distribution Per Ship Mode for Each Order")
st.markdown(
    "<h4 style='color: green;'>It appears that Standard Class have the highest Count, while <span style='color: red;'>Same day has the lowest</span>.</h4>",
    unsafe_allow_html=True
)
fig=px.histogram(df_selection['Ship Mode'].value_counts().reset_index(), x='Ship Mode', y='count', color='Ship Mode',color_discrete_sequence=retro)
st.plotly_chart(fig)


st.divider()

################################################################################

# plot 3

st.header("The Distribution Per Category for Each Order")
fig = px.histogram(df_selection['Category'].value_counts().reset_index(), y='count', x='Category',color='Category',color_discrete_sequence=cofee)
st.plotly_chart(fig)

st.divider()


################################################################################



# plot 4 
st.header(" the sum of sales for Category and Sub-Category and Product Name ")

fig = px.treemap(df_selection, path=['Category', 'Sub_Category','Product Name'], values='Sales',
                 color='Category', color_discrete_sequence=grad_black)

st.plotly_chart(fig)

st.divider()
################################################################################

#plot 6
st.header(" the sum of sales for Category and Sub-Category")

fig = px.sunburst(df_selection, path=['Category', 'Sub_Category'], values='Sales',
                  color='Category', color_discrete_sequence=colors10)

st.plotly_chart(fig)

st.divider()
################################################################################


# plot 7 
st.header(" the Category that each customer purchase and customer's name")

# Group by Product Name to get counts, then sort the values in descending order
Customer_counts = df_selection['Customer Name'].value_counts().reset_index()
Customer_counts.columns = ['Customer Name', 'count']
Customer_counts = Customer_counts.sort_values(by='count', ascending=False)

fig = px.treemap(df_selection, path=['Category','Customer Name'], values='Sales',
                 color='Category', color_discrete_sequence=cold_blue)
st.plotly_chart(fig)

st.divider()

################################################################################

# plot 8

st.header(" the sales for each one of 'Region','State','City'")

fig = px.treemap(df_selection, path=['Region','State','City'], values='Sales',
                 color='Region', color_discrete_sequence=cold_blue)

st.plotly_chart(fig)

st.divider()

################################################################################
# plot 9

st.header("the sales for each one of 'State','Category','Sub_Category '")

fig = px.treemap(df_selection, path=['State','Category','Sub_Category'], values='Sales',color='State', color_discrete_sequence=yel_gre1)

st.plotly_chart(fig)



            


