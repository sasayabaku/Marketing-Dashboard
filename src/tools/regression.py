import streamlit as st

import numpy as np

from sklearn.linear_model import LinearRegression

import plotly.express as px
import plotly.graph_objects as go

import sys
sys.path.append("../")
import config

def single_regression(df):

    line1_spacer1, line1_1, line1_2, line1_spacer2 = st.columns((.1, 1.0, 2.2, .1))

    dataset = df.copy()

    attribute_list = []
    for name, dtype in zip(dataset.dtypes.index, dataset.dtypes):
        if dtype in config.ACCEPT_DTYPE:
            attribute_list.append(name)

    with line1_1:
        x_attribute = st.selectbox(
            "説明変数",
            attribute_list
        )
        y_attribute = st.selectbox(
            "目的変数",
            attribute_list
        )

    dataset = dataset.dropna(subset=[x_attribute, y_attribute])
    X = dataset[x_attribute].values.reshape(-1, 1)
    Y = dataset[y_attribute].values.reshape(-1)

    #　回帰直線 計算
    model = LinearRegression()
    model.fit(X, Y)

    # Plot用のダミーデータ
    X_range = np.linspace(X.min(), X.max(), 100)
    Y_range = model.predict(X_range.reshape(-1, 1))

    # 決定係数計算
    score = model.score(X, Y)

    # 相関係数 計算
    corr_data = dataset[[x_attribute, y_attribute]].corr()[x_attribute][y_attribute]

    with line1_1:
        st.info("回帰直線 決定係数: {}".format(score))
        st.success("相関係数: {}".format(corr_data))

    with line1_2:
        fig = px.scatter(dataset, x=x_attribute, y=y_attribute)
        fig.add_trace(go.Scatter(x=X_range, y=Y_range.reshape(-1), name="回帰直線"))

        st.plotly_chart(fig, use_container_width=True, config=config.go_config)




def run(df):
    
    st.markdown("#### 回帰分析")
    single_regression(df)