from django.shortcuts import render

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import plotly.graph_objects as go


def index(response):
    # read data
    df = read_data('data.csv')
    df_y = to_year_column(df)

    # implement initial_model
    train, test = get_train_test_sets(df)
    initial_model = set_model(train, (1,1,1))
    final_model = set_model(df, (1,1,1))

    # validation and residuals
    validation = pd.Series(initial_model.forecast(len(test))[0], index=test.index).to_frame('Number of Cases')
    validation.index = pd.to_datetime(validation.index)
    residuals = test - validation

    # forecast
    forecast = pd.Series(final_model.forecast(24)[0], index=pd.date_range(start=df.index[-1], periods=24, freq='M', closed='right')).to_frame('Number of Cases')

    # graphs
    raw_line_a = line_area(df, 'Number of HIV Cases from 2010 to Present')
    raw_line_s = line_seasonal(df_y)

    residuals_line_a = line_area(residuals, 'Residuals', 'blue', 'Month and Year')
    validation_line_c = line_comparison(df, validation)
    validation_bar_c = bar_comparison(test, validation)

    forecast_line_c = line_forecast(df, forecast)
    forecast_bar_s = bar_simple(forecast)

    context = {'raw_line_a': raw_line_a,
               'raw_line_s': raw_line_s,
               'residuals_line_a': residuals_line_a,
               'validation_line_c': validation_line_c,
               'validation_bar_c': validation_bar_c,
               'forecast_line_c': forecast_line_c,
               'forecast_bar_s': forecast_bar_s}

    return render(response, "dashboard/index.html", context)


# read and clean data

def read_data(path):
    df = pd.read_csv(path, index_col='Date')
    df.index = pd.to_datetime(df.index)
    return df


def to_year_column(df):
    df_y = df.copy(deep=True)
    df_y.columns = [None]

    df_y.index = df_y.index.strftime('%b-%Y').str.split('-', expand=True)
    df_y = (df_y.rename_axis(index=['month', ''])
            .unstack()
            .sort_index(axis=0, key=lambda x: pd.to_datetime(x, format='%b').month)
            )
    df_y.columns = [x[1] for x in list(df_y.columns)]

    return df_y


# model

def get_train_test_sets(df):
    seventy_percent = int(((len(df)) / 10) * 7.5)
    train = df[:seventy_percent]
    test = df[seventy_percent:]
    return train, test


def set_model(train, order):
    model = ARIMA(train, order=order, freq='M').fit()
    return model


# graphs

def line_area(df, title, color='red', y_name='Year'):
    fig = px.line(df, x=df.index, y="Number of Cases", width=650, height=280)
    fig.update_traces(
        line_color=color,
        selector=dict(type='scatter'))
    fig.update_layout(
        margin=dict(l=100, r=20, t=30, b=20),
        xaxis=dict(title_text=y_name),
        font=dict(size=11),
        title=dict(
            text=title,
            x=0.159,
            y=0.95,
            font=dict(
                family="Helvetica",
                size=13,
                color='#000000'
            )
        ),
    )
    line_a = fig.to_html(full_html=False)
    return line_a


def line_seasonal(df_y):
    fig = go.Figure()
    for year in df_y.columns:
        fig.add_trace(go.Scatter(x=df_y.index, y=df_y[str(year)], name=str(year)))

    fig.update_layout(
        width=650, height=280,
        margin=dict(l=100, r=0, t=30, b=20),
        yaxis=dict(title_text="Number of Cases", ),
        xaxis=dict(title_text="Month"),
        font=dict(size=11),
        title=dict(
            text="Number of HIV Cases per Year",
            x=0.159,
            y=0.95,
            font=dict(
                family="Helvetica",
                size=13,
                color='#000000'
            )
        ),
    )

    line_s = fig.to_html(full_html=False)
    return line_s


def line_comparison(df_1, df_2):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_1.index, y=df_1['Number of Cases'],
        name='actual'
    ))

    fig.add_trace(go.Scatter(
        x=df_2.index, y=df_2['Number of Cases'],
        name='prediction'
    ))

    fig.update_layout(
        width=650, height=280,
        margin=dict(l=100, r=0, t=30, b=20),
        yaxis=dict(title_text="Number of Cases", ),
        xaxis=dict(title_text="Year"),
        font=dict(size=11),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            orientation="h"
        ),
        title=dict(
            text="Actual Values VS Predicted Values",
            x=0.159,
            y=0.95,
            font=dict(
                family="Helvetica",
                size=13,
                color='#000000'
            )
        ),
    )

    line_c = fig.to_html(full_html=False)
    return line_c


def line_forecast(df_1, df_2):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_1.index.append(df_2.index), y=df_1['Number of Cases'],
        name='actual'
    ))

    fig.add_trace(go.Scatter(
        x=df_2.index, y=df_2['Number of Cases'],
        name='prediction'
    ))

    fig.update_layout(
        width=650, height=280,
        margin=dict(l=100, r=0, t=30, b=20),
        yaxis=dict(title_text="Number of Cases", ),
        xaxis=dict(title_text="Year"),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            orientation="h"
        ),
        font=dict(size=11),
        title=dict(
            text="Actual Values and Predicted Values",
            x=0.159,
            y=0.95,
            font=dict(
                family="Helvetica",
                size=13,
                color='#000000'
            )
        ),
    )

    line_f = fig.to_html(full_html=False)
    return line_f


def bar_simple(df):
    fig = go.Figure(data=[
        go.Bar(name='actual', x=df.index, y=df['Number of Cases'])
    ])

    fig.update_layout(
        width=650, height=280,
        margin=dict(l=100, r=0, t=30, b=20),
        yaxis=dict(title_text="Number of Cases", ),
        xaxis=dict(title_text="Month and Year"),
        font=dict(size=11),
        title=dict(
            text="Predicted Values in the Next Two Years",
            x=0.159,
            y=0.95,
            font=dict(
                family="Helvetica",
                size=13,
                color='#000000'
            )
        ),
    )

    bar_s = fig.to_html(full_html=False)
    return bar_s

def bar_comparison(df_1, df_2):
    fig = go.Figure(data=[
        go.Bar(name='actual', x=df_2.index, y=df_1['Number of Cases']),
        go.Bar(name='prediction', x=df_2.index, y=df_2['Number of Cases'])
    ])

    fig.update_layout(
        barmode='group',
        width=650, height=280,
        margin=dict(l=100, r=0, t=30, b=20),
        yaxis=dict(title_text="Number of Cases", ),
        xaxis=dict(title_text="Month and Year"),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            orientation="h"
        ),
        font=dict(size=11),
        title=dict(
            text="Actual Values VS Predicted Values",
            x=0.159,
            y=0.95,
            font=dict(
                family="Helvetica",
                size=13,
                color='#000000'
            )
        ),
    )

    bar_c = fig.to_html(full_html=False)
    return bar_c
