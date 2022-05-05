import openpyxl
import streamlit as st

import pandas as pd
import openpyxl

st.set_page_config(layout="wide")

selected_tool = st.selectbox("tools", (
    '自動分析', 'aaa'
));

data_file = st.file_uploader("Upload Data File")

if data_file != None:

    sheetname_list = [""]
    sheetname_list.extend(openpyxl.load_workbook(data_file).sheetnames)

    sheetname = st.selectbox(
        "シート名",
        sheetname_list
    )



    if (sheetname != ""):
        df = pd.read_excel(data_file, sheet_name=sheetname, index_col=0)
        
        try:
            df['date'] = pd.to_datetime(df['date'])
        except:
            df['date'] = df['日付']
            df['date'] = pd.to_datetime(df['date'])


        if selected_tool in ('自動分析'):
            from tools import regression
            regression.run(df)

else:
    st.info("Please Upload CSV File")