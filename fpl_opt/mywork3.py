import pulp
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import fastprogress
import sys
sys.path.append("..")
from fpl_opt.transfers import TransferOptimiser


header=st.beta_container()
dataset=st.beta_container()
features=st.beta_container()
modelTraining=st.beta_container()

@st.cache
def get_data(filename):
	df = pd.read_csv(filename)
	return df


with header:
    st.title('Visualize data')
    st.text('In this project I look into the Fantasy Premier League ')
    st.subheader('Number of pickups by hour')

df = get_data('2021.csv')

df.sort_values(by=['element_type'])


positions = df["element_type"]
clubs = df["team_code"]
fname = df["first_name"]
lname = df["second_name"]




with st.echo(code_location='below'):
    import plotly.express as px

    fig = px.scatter(
        x=df["now_cost"] / 10,
        y=df["total_points"],
        
    )
    fig.update_layout(
        xaxis_title="COST",
        yaxis_title="TOTAL POINTS",
    )

    st.write(fig)


st.bar_chart(df["total_points"])


