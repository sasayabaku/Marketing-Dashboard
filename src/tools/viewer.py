import streamlit as st
from st_aggrid import AgGrid

import numpy as np

def table_editor(df):
    AgGrid(df)

def run(df):
    table_editor(df)