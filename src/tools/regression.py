import streamlit as st

import numpy as np
import pandas as pd

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

def multiple_regression(df):

    line1_spacer1, line1_1, line1_2, line1_spacer2 = st.columns((.1, 1.0, 2.2, .1))

    dataset = df.copy()

    attribute_list = []
    for name, dtype in zip(dataset.dtypes.index, dataset.dtypes):
        if dtype in config.ACCEPT_DTYPE:
            attribute_list.append(name)

    with line1_1:
        x_attribute = st.multiselect(
            "説明変数",
            attribute_list,
            key="multi_reg_attr_x"
        );

        y_attribute = st.selectbox(
            "目的変数",
            attribute_list,
            key="multi_reg_attr_y"
        )
    
    if len(x_attribute) > 0:
        dataset = dataset.dropna(subset=[y_attribute].extend(x_attribute))
        X = dataset[x_attribute].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))).values
        Y = dataset[y_attribute]
        Y = ((Y - Y.mean()) / (Y.max() - Y.min())).values.reshape(-1)

        model = LinearRegression()
        model.fit(X, Y)

        with line1_2:
            st.table(
                pd.DataFrame(model.coef_, index=x_attribute, columns=[y_attribute])
                .sort_values(y_attribute, ascending=False)
            )

    else: 
        st.info("Select X Attribute")


def correlation(df):

    line1_spacer1, line1_1, line1_2, line1_spacer2 = st.columns((.1, 0.6, 2.6, .1))

    dataset = df.copy()

    with line1_1:
        st.caption("【一般論での相関係数の目安】")
        st.caption("x < 0.2 : 相関がない")
        st.caption("0.2 < x < 0.4 : 弱い相関")
        st.caption("0.4 < x < 0.7 : 通常の相関")
        st.caption("0.7 < x < 1.0 : 強い相関")
        st.write("")
        threshold = st.slider("相関係数 閾値", 0.0, 1.0)


    dataset_corr = dataset.corr()
    if threshold > 0:
        plot_data = dataset_corr.where(((dataset_corr > threshold) | (dataset_corr < -threshold)), 0)
    else:
        plot_data = dataset_corr


    with line1_2:

        fig = px.imshow(
            plot_data,
            color_continuous_scale=px.colors.diverging.Picnic,
            zmin=-1,
            zmax=1
        )

        st.plotly_chart(fig, use_container_width=True, config=config.go_config)

def run(df):
    
    st.markdown("#### 回帰分析")
    single_regression(df)
    st.markdown("##### 多重回帰")
    multiple_regression(df)

    st.markdown("#### 相関分析")
    correlation(df)