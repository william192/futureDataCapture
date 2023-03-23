#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import re
import logging
from sqlalchemy import create_engine, Table, MetaData, or_, except_, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session
from sqlalchemy import func
from sqlalchemy import and_
import json
import os
import seaborn as sns
import quandl
from yahoo_quote_download import yqd, validater
from itertools import compress
import pickle
import trading_calendars as tc# import get_calendar, TradingCalendar
import talib as ta
import requests
import traceback
import io


k_weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


# When ODIN was developed a new format for parameters was used and this converts
# from the old to the new format
def convert_to_newparams(all_param):
    converted_params = []

    for param_in in all_param:
        t = {}
        param_v = eval(param_in)
        t['option_symbols'] = param_v[0]
        t['trade_market'] = param_v[1]
        p = {
            "series": [param_v[2]],
            "delta_region": 'inside',
            "put_call_op": param_v[8],
            "signal_map": param_v[9],
            "signal_ma": param_v[7],
            "near_exp": param_v[5],
            "far_exp": param_v[6]
        }
        t['indicator_params'] = p
        t['lower_delta'] = param_v[3]
        t['upper_delta'] = param_v[4]
        converted_params.append(t)

    return converted_params


# Thor is a simple long only equity model which uses options information to improve the risk
# adjusted return of a simple portfolio
def generate_thor_parameters():

    mkt1 = {
        "option_symbols": ['QQQ', 'QQQQ'],
        "trade_market": 'QQQ'
    }
    mkt2 = {
        "option_symbols": ['SPX', 'SPXM'],
        "trade_market": 'SPX',
    }
    mkts = [mkt1, mkt2]

    params1 = {
        "lower_delta": 0.6,
        "upper_delta": 1.0,
        "indicator_params":
            {
                "delta":
                    {"series": ['delta_dol_vol'],
                     "delta_region": 'inside',
                     "put_call_op": 'diff',
                     "signal_map": 3,
                     "signal_ma": 2,
                     "near_exp": 2,
                     "far_exp": 200}
            }
    }

    params2 = {
        "lower_delta": 0.0,
        "upper_delta": 1.0,
        "indicator_params":
            {
                "delta":
                    {"series": ['delta_dol_vol'],
                     "delta_region": 'inside',
                     "put_call_op": 'diff',
                     "signal_map": 3,
                     "signal_ma": 2,
                     "near_exp": 10,
                     "far_exp": 90}
            }
    }

    params3 = {
        "lower_delta": 0.3,
        "upper_delta": 0.7,
        "indicator_params":
            {
                "delta":
                    {"series": ['gamma_vol'],
                     "delta_region": 'inside',
                     "put_call_op": 'diff',
                     "signal_map": 4,
                     "signal_ma": 5,
                     "near_exp": 1,
                     "far_exp": 100}
            }
    }

    params = [params1, params2, params3]
    param_set = [{**y, **x} for x in params for y in mkts]
    return param_set


def quicksave(output_folder, title_str, ax, fig):
    if not isinstance(ax, list):
        ax.set_title(title_str)
    file_name = os.path.join(output_folder, title_str + '.png')
    fig.savefig(file_name, bbox_inches="tight")
    plt.close(fig)

    return

def write_thor_charts_for_options(df_options, param_set, output_folder):
    s = pd.IndexSlice
    for i, df_option in zip(param_set, df_options):
        param_name = get_param_set_names(i)[0].replace('\'','')
        series = i['indicator_params']['delta']['series']
        lb = i['indicator_params']['delta']['signal_ma']
        df_temp = df_option.loc[s[:, :, True, :], :].groupby(['date', 'symbol', 'call_put']).sum().loc[:, series]
        df_temp = np.abs(df_temp.unstack(level=['call_put', 'symbol']))
        ax = df_temp.rolling(lb).mean().tail(252).plot(figsize=(10,6))
        ax.set_title(param_name)

        filename = os.path.join(output_folder, f'chart_options_{param_name}.png')
        fig = ax.get_figure()
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)

    return

def write_thor_charts(df_indicators, df_trades, output_folder, traded_symbols, allocation, modelid):

    s = pd.IndexSlice
    ax = df_trades.loc[:, s[:, 'pct_ret']].tail(1000).cumsum().plot(figsize=(8, 8))
    fig = ax.get_figure()
    quicksave(output_folder, f'{modelid}_chart_price_4_year_cum_return', ax, fig)

    ax = df_trades.loc[:, s[:, 'pct_ret']].tail(1000).plot(figsize=(8, 8))
    fig = ax.get_figure()
    quicksave(output_folder, f'{modelid}_chart_price_4_year_return', ax, fig)

    ax = df_indicators.loc[:, s[:, :, 'sig']].tail(1000).plot(figsize=(15, 15))
    fig = ax.get_figure()
    quicksave(output_folder, f'{modelid}_chart_signal_history_equities', ax, fig)

    ax = df_indicators.loc[:, s[:, :, 'sig']].tail(120).plot(figsize=(15, 15))
    fig = ax.get_figure()
    quicksave(output_folder, f'{modelid}_chart_signal_6months_equities', ax, fig)

    ax = df_indicators.loc[:, s[:, :, 'ret']].cumsum().plot(figsize=(15, 15))
    fig = ax.get_figure()
    quicksave(output_folder, f'{modelid}_chart_signal_raw_performance_equities', ax, fig)

    ###################################
    ## signal distribution charts
    names = df_indicators.loc[:, s[:, :, 'sig']].columns
    sns.set(style="darkgrid")

    ncols = 2
    nrows = (len(names) // ncols)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex='all', sharey='all', figsize=(10, 10))

    for name, ax in zip(names, axes.flat):
        temp_data = df_indicators.loc[:, name]
        temp_data.hist(bins=30, color='b', ax=ax)
        ax.set_title(name)
        last_val = temp_data.tail(1)
        x = last_val.values
        ax.axvline(x=x, ymin=0, ymax=1, color='k', linewidth=2)
        ax.axvline(x=-0.3, ymin=0, ymax=1, color='r', linewidth=2)
        ax.axvline(x=0.3, ymin=0, ymax=1, color='r', linewidth=2)
        ax.set_xlim(-4, 2)

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, f'{modelid}_chart_signal_distributions.png'), bbox_inches="tight")
    plt.close(fig)

    #######################################
    ## Weekly Market Patterns
    ncols = 1
    nrows = 5

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex='all', sharey='all', figsize=(15, 20))
    temp = df_trades.loc[:, s[:, ['sig_pct_ret']]].copy()
    temp = pd_pivot_dayofweek(temp['2010':])

    for name, ax in zip(k_weekday_names, axes.flat):
        temp_data = temp.loc[:, s[:, :, name]]
        temp_data.cumsum().plot(ax=ax)
        ax.set_title(name)

    quicksave(output_folder, f'{modelid}_chart_weekly_market_return_pattern', [], fig)

    #######################################
    ## Benchmark versus portfolio return distribution
    final = generate_benchmark_comparison(df_trades, traded_symbols, allocation)
    equity_names = list(df_trades.columns.get_level_values(0).unique().values[-2:])
    temp = final.loc[:, s[equity_names, :]]

    ncols = 1
    nrows = 2
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex='all', sharey='all', figsize=(15,17))
    names = temp.columns.values

    bins = np.arange(-0.06, 0.06, 0.0015).tolist()

    name = equity_names[0]
    ax = axes[0]
    temp.loc[:,s[name,'Benchmark']].hist(bins=bins, color='k', ax=ax, label='Benchmark')
    temp.loc[:,s[name,'ThorShield']].hist(bins=bins, color='c', ax=ax, label='ThorShield', alpha=0.75)
    ax.set_title(name + " - percent return", fontsize=15)
    ax.legend(prop={'size': 20})

    name = equity_names[1]
    ax = axes[1]
    temp.loc[:, s[name, 'Benchmark']].hist(bins=bins, color='k', ax=ax, label='Benchmark')
    temp.loc[:, s[name, 'ThorShield']].hist(bins=bins, color='c', ax=ax, label='ThorShield', alpha=0.75)
    ax.set_title(name + " - percent return", fontsize=15)
    ax.legend(prop={'size': 20})

    fig.savefig(os.path.join(output_folder, f'{modelid}_chart_distribution_comparison.fullsample.png'), bbox_inches="tight")
    plt.close(fig)

    ######################################################
    ## Performance Charts
    plotsym = equity_names + ['Portfolio']
    sns.set(style="whitegrid")
    fs = (15,6)
    slices = [s[:], s['2008-09-01':'2009-09-01'], s['2018-10-01':'2019-03-01'], s['2020-02-01':'2020-06-01'], s['2020-01-01':]]
    slice_name= ['full', '2008', '2018', '2020', '2020 YTD']

    for period_name, period_slice in zip(slice_name, slices):
        for sym in plotsym:
            data = final.loc[period_slice, s[sym,:]].cumsum()

            ax = data.plot(figsize=fs, fontsize=20, linewidth=2)
            ax.legend(fontsize=15)
            ax.xaxis.label.set_visible(False)

            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.0%}"))
            ax.set_title(f"Estimated {sym} return - no compounding", fontsize=20)
            fig = ax.get_figure()
            filename = os.path.join(output_folder, f'{modelid}_chart_pnl_{period_name}_{sym}_portfolio_vs_benchmark.png')
            fig.savefig(filename, bbox_inches="tight")
            plt.close(fig)

    ######################################################
    # Performance on $100K of each constituent
    temp_perf = df_trades.loc[:, s[:, ['sig_dol_ret']]].tail(252)
    temp_perf['ThorShield'] = df_trades.loc[:, s[:, ['sig_dol_ret']]].sum(axis=1).tail(252)
    temp_perf = temp_perf.droplevel(level='stage', axis=1)

    ax = temp_perf.cumsum().plot(figsize=(20, 6), fontsize=15, linewidth=2)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("${x:.0f}"))
    ax.set_title("ThorShield - $100,000 - no compounding", fontsize=20)
    ax.legend(fontsize=15)
    ax.xaxis.label.set_visible(False)
    fig = ax.get_figure()
    fig.savefig(os.path.join(output_folder, f'{modelid}_chart_pnl_strategy_1yr.png'), bbox_inches="tight")
    plt.close(fig)

    ######################################################
    # YTD PCT Return
    temp_pnl = df_trades.loc['2020', s[:, ['pct_ret']]].cumsum()
    temp_pnl['ThorShield'] = temp_pnl.mean(axis=1)
    temp_pnl = temp_pnl.droplevel('stage', axis=1)
    ax = temp_pnl.plot(figsize=(20, 6))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.0%}"))
    plt.title('YTD Non-Compounding Percent Return')
    ax.legend(fontsize=15)
    ax.xaxis.label.set_visible(False)
    fig = ax.get_figure()
    fig.savefig(os.path.join(output_folder, f'{modelid}_chart_pctret_YTD.png'), bbox_inches="tight")
    plt.close(fig)

    ######################################################
    # Rolling Correlation
    temp_perf = df_trades.loc[:, s[equity_names, ['pct_ret']]].copy()
    temp_perf['ThorShield'] = df_trades.loc[:, s[:, ['sig_pct_ret']]].sum(axis=1)
    temp_perf = temp_perf.droplevel(level='stage', axis=1)
    rc = temp_perf.rolling(252).corr()['ThorShield'].unstack('mkt')
    rc = rc[equity_names].tail(252)
    ax = rc.plot(figsize=(15, 6), fontsize=15, linewidth=2)
    ax.legend(fontsize=15)
    ax.xaxis.label.set_visible(False)

    ax.set_title("Rolling 1 year correlation coefficient with the portfolio", fontsize=20)
    fig = ax.get_figure()
    fig.savefig(os.path.join(output_folder, f'{modelid}_chart_rolling_correlation.png'), bbox_inches="tight")
    plt.close(fig)

    ######################################################
    # Comparison of SPY and QQQ in and out of Thor
    sns.set(style="whitegrid")
    ax = final.loc[:, s[equity_names, :]].tail(1000).cumsum().plot(figsize=(20, 6), fontsize=15, linewidth=2)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.0%}"))
    ax.set_title("Estimated return - no compounding", fontsize=20)
    ax.legend(fontsize=15)
    ax.xaxis.label.set_visible(False)

    fig = ax.get_figure()
    fig.savefig(os.path.join(output_folder, f'{modelid}_chart_pnl_market_portfolio_vs_benchmark.png'), bbox_inches="tight")
    plt.close(fig)

    ######################################################
    # Charts to compare SPY performance at critical points in time
    final = generate_benchmark_comparison(df_trades, traded_symbols, allocation)

    temp = final.loc[:, s['SPY', :]]

    slices = [s['2020'], s['2018-10':'2019-05'], s['2008-05':'2010-01'], s['2011-05':'2012-01']]
    slice_names = ['2020', '2018', '2008', '2011']

    for temp_s, temp_name in zip(slices, slice_names):
        ax = temp[temp_s].cumsum().plot(figsize=(10, 10))
        fig = ax.get_figure()
        fig.savefig(os.path.join(output_folder, f'{modelid}_SPY Performance-' + temp_name), bbox_inches="tight")
        plt.close(fig)

    ######################################################
    # Rolling 10 day return chart
    df_temp = final.loc[:, s['Portfolio', :]].rolling(10).sum().dropna()
    ax = df_temp.loc[:, s['Portfolio', 'ThorShield']].hist(bins=30, color='b')
    ax.set_title('Rolling 10 Day Return')
    last_value = df_temp.loc[:, s['Portfolio', 'ThorShield']].tail(1)[0]
    ax.axvline(x=last_value, ymin=0, ymax=1, color='r', linewidth=2)
    ax.axvline(x=0.00001, ymin=0, ymax=1, color='w', linewidth=2, linestyle='--')
    ax.axvline(x=-0.00005, ymin=0, ymax=1, color='k', linewidth=2, linestyle='--')
    fig = ax.get_figure()
    fig.savefig(os.path.join(output_folder, f'{modelid}_chart_pnl_hist_10_day.png'), bbox_inches="tight")

    return


def combine_signal_spot_data(param_set, sig, use_weekday_modulation=False):
    # pull historic spot data
    s = pd.IndexSlice
    spot = list()
    for i in param_set:
        params = i
        temp_spot = query_spot_data([params['trade_market']])
        spot.append(temp_spot)

    # insert historic spot data as percent return
    sg_ret = []
    for sp, sg in zip(spot, sig):
        temp_columns_ret = sg.loc[:, s[:, :, 'lev_val']].rename(columns={'lev_val': 'ret'}).columns
        temp_columns_lev = sg.loc[:, s[:, :, ['lev_val', 'sig']]].columns

        mkt = sg.columns.get_level_values(level=0)[0]
        x = sp.loc[:, (mkt, 'pct_ret')].to_frame().copy()
        temp_sig = sg.droplevel(axis=1, level=1)
        
        merged_sig = temp_sig.merge(x, how='outer', on='date', sort=True).loc[:, s[:, ['lev_val', 'sig', 'pct_ret']]]
        temp_lev = merged_sig.loc[:, s[:, 'lev_val']].shift(2)
        temp_lev = temp_lev.fillna(method='ffill')
        tempret = temp_lev * merged_sig.loc[:, s[:, 'pct_ret']].values
        tempret.rename(columns={'lev_val': 'ret'}, inplace=True)
        tempret.columns = temp_columns_ret
        templev = merged_sig.loc[:, s[:, ['lev_val', 'sig']]]
        templev.columns = temp_columns_lev

        # calculate leverage returns
        sg_ret.append(pd.concat([tempret, templev], axis=1))

    # concat all to a single dataframe but remove any duplicate columns which are from duplicate spot history
    df_indicators = pd.concat(sg_ret.copy(), axis=1)

    return df_indicators


def pd_add_str_col_level(df_in, level_name, column_name):
    df_in.columns = pd.MultiIndex.from_tuples([(column_name,) + (x,) for x in df_in.columns], names=(level_name,) + df_in.columns.names)
    return df_in


def generate_odin_benchmark_comparison(df_trades):

    s = pd.IndexSlice

    temp_perf = df_trades.loc[:, s[:, ['pct_ret', 'sig_pct_ret']]]
    temp_perf = temp_perf.rename(columns={'pct_ret': 'Market', 'sig_pct_ret': 'Odin'})

    final = temp_perf
    return final


def generate_benchmark_comparison(df_trades, traded_symbols, allocation):
    s = pd.IndexSlice

    temp = df_trades.loc[:, s[traded_symbols, ['pct_ret']]] * allocation
    temp = temp.rename(columns={'pct_ret': 'Benchmark', 'sig_pct_ret': 'ThorShield'})
    temp2 = df_trades.loc[:, s[traded_symbols, ['sig_pct_ret']]] * allocation
    temp2 = temp2.rename(columns={'pct_ret': 'Benchmark', 'sig_pct_ret': 'ThorShield'})
    tt = pd.concat([temp, temp2], axis=1)
    ttt = tt.groupby(level='stage', axis=1).sum()
    temp_perf_port = pd_add_str_col_level(ttt, 'mkt', 'Portfolio')
    final = pd.concat([tt, temp_perf_port], axis=1)

    return final


def generate_thor_trading_table(signal_symbols, traded_symbols, allocation, k_portfolio_size, psn):
    #################################
    ## Create all the dataframes used for reporting
    ##
    s = pd.IndexSlice

    # get data on the traded markets
    market_data = [query_spot_data([x]) for x in traded_symbols]
    market_data = [x.drop(columns=['local_symbol'], axis=1, level='stage') for x in market_data]

    # insert the three positions (pct, dol and shares)
    for p, t, w, d in zip(signal_symbols, traded_symbols, allocation, market_data):
        close = d.loc[:, s[t, 'close']]
        d.loc[:, s[t, 'sig_pct_psn']] = psn.loc[:, p]
        d.loc[:, s[t, 'sig_dol_psn']] = psn.loc[:, p] * k_portfolio_size * w
        d.loc[:, s[t, 'sig_share_psn']] = np.round(d.loc[:, s[t, 'sig_dol_psn']] / close)
        d.loc[:, s[t, 'sig_pct_ret']] = d.loc[:, s[t, 'sig_pct_psn']].shift(2) * d.loc[:, s[t, 'pct_ret']]
        d.loc[:, s[t, 'sig_dol_ret']] = d.loc[:, s[t, 'sig_share_psn']].shift(2) * d.loc[:, s[t, 'dol_ret']]

    df_trades = pd.concat(market_data, axis=1).dropna()

    return df_trades

def generate_thor_position(param_set, sig, use_version_2=False):
    #################################
    ## Join spot data and create psn dataframe
    ##
    # indicators are the signals combined and modulated - they contain a return
    # which is relative to the signal market, not the traded market.
    s = pd.IndexSlice
    df_indicators = combine_signal_spot_data(param_set, sig)

    if use_version_2:
        # create the position as the mean of all the indicators in equities
        psn = df_indicators.loc[:, s[:, :, 'lev_val']].groupby('mkt', axis=1).mean()
        psn = psn + 0.1
        psn = ((psn - 0.2) * 2 + 0.1).clip(lower=0, upper=1)
    else:
        # create the position as the mean of all the indicators in equities
        psn = df_indicators.loc[:, s[:, :, 'lev_val']].groupby('mkt', axis=1).mean()
        psn.loc[:, 'QQQ'] = psn.loc[:, 'QQQ'] + 0.1
        psn = ((psn - 0.2) * 2 + 0.1).clip(lower=0, upper=1)

        # bonds will reflect the total exposure to equities - if equities drop we drop bonds
        bond_lev = (psn.sum(axis=1) * 1.25).clip(lower=0, upper=1)
        psn.loc[:, 'TLT'] = bond_lev
        # ief is known to be low vol so maintain some exposure
        psn.loc[:, 'IEF'] = (bond_lev + 0.3).clip(lower=0, upper=1)
        psn = psn.clip(lower=0, upper=1)

    return psn, df_indicators


def generate_thorgold_position(param_set, sig):
    #################################
    ## Join spot data and create psn dataframe
    ##
    # indicators are the signals combined and modulated - they contain a return
    # which is relative to the signal market, not the traded market.
    s = pd.IndexSlice
    df_indicators = combine_signal_spot_data(param_set, sig)

    # create the position as the mean of all the indicators in equities
    psn = df_indicators.loc[:, s[:, :, 'lev_val']].groupby('mkt', axis=1).mean()

    return psn, df_indicators


def generate_thor_signal(param_set):

    engine = create_engine(os.getenv('DATABASE_CONNECTION'))
    session_factory = sessionmaker(bind=engine)
    metadata = MetaData()
    metadata.reflect(bind=engine)
    iv = Table('ivolatility', metadata, autoload=True, autoload_with=engine)
    ep = Table('equity_price', metadata, autoload=True, autoload_with=engine)
    session = scoped_session(session_factory)

    df_options = list()
    sig = list()

    for i in param_set:
        params = i
        df_option = query_delta_gamma_data(session, iv, params['option_symbols'], eod=True,
                                               strike_threshold=1,
                                               lower_abs_delta_threshold=params['lower_delta'],
                                               upper_abs_delta_threshold=params['upper_delta'])
        df_option = prepare_options_data(df_option, params['trade_market'])
        df_options.append(df_option)

        temp_sig = pd.concat([calc_signal_deltavol_v1(
            df_option,
            params['indicator_params'][x])
            for x in params['indicator_params']], axis=1)

        # this adds the market to the dataframe
        temp_sig.columns = pd.MultiIndex.from_tuples([(params['trade_market'],) + x for x in temp_sig.columns])
        temp_sig.columns.names = ['mkt', 'param', 'stage']
        sig.append(temp_sig)

    engine.dispose()
    return sig, df_options

########################################################################
### ODIN Rules Below
########################################################################
def generate_odin_position(trade_market, sig, spot_data):
    #################################
    ## Join spot data and create psn dataframe
    ##
    # indicators are the signals combined and modulated - they contain a return
    # which is relative to the signal market, not the traded market.
    s = pd.IndexSlice
    # pull historic spot data
    s = pd.IndexSlice

    # insert historic spot data as percent return
    sg = sig

    x = spot_data.loc[:, (trade_market, 'pct_ret')].to_frame().copy()
    x, dummy = x.align(sg, join='right', axis=0)

    # calculate leverage returns
    temp = sg.loc[:, s[:, :, 'lev_val']].shift(2).multiply(x.values, axis=1).fillna(0)
    temp.rename(columns={'lev_val': 'ret'}, inplace=True)
    sg = pd.concat([sg, temp], axis=1)
    df_indicators = sg

    # create the position as the mean of all the indicators in equities
    psn = df_indicators.loc[:, s[:, :, 'lev_val']].groupby('mkt', axis=1).mean()
    psn = ((psn - 0.2) * 2 + 0.1).clip(lower=0, upper=1)
    psn = psn.clip(lower=0, upper=1)

    # this adds the market to the dataframe
    psn.columns = pd.MultiIndex.from_tuples([(trade_market, 'psn')], names=['mkt', 'stage'])

    return psn, df_indicators


def generate_odin_trading_table(traded_symbol, portfolio_size, psn):
    s = pd.IndexSlice

    # get data on the traded markets
    market_data = query_spot_data([traded_symbol])
    market_data = market_data.drop(columns='local_symbol', axis=1, level='stage')
    market_data = pd.concat([market_data, psn], join='inner', axis=1)
    market_data = market_data.rename(columns={'psn': 'sig_pct_psn'})

    # insert the three positions (pct, dol and shares)
    close = market_data.loc[:, s[traded_symbol, 'close']]
    t = traded_symbol
    # first calc the desired dollar exposure, then calc shares and round and then calc final dollars
    market_data.loc[:, s[t, 'sig_dol_psn']] = psn.loc[:, t] * portfolio_size
    market_data.loc[:, s[t, 'sig_share_psn']] = np.round(market_data.loc[:, s[t, 'sig_dol_psn']] / close)
    market_data.loc[:, s[t, 'sig_dol_psn']] = market_data.loc[:, s[t, 'sig_share_psn']] * close
    
    market_data.loc[:, s[t, 'sig_pct_ret']] = market_data.loc[:, s[t, 'sig_pct_psn']].shift(2) * market_data.loc[:,
                                                                                                 s[t, 'pct_ret']]
    market_data.loc[:, s[t, 'sig_dol_ret']] = market_data.loc[:, s[t, 'sig_share_psn']].shift(2) * market_data.loc[:,
                                                                                                   s[t, 'dol_ret']]

    df_trades = market_data

    return df_trades


def generate_odin_signal(all_parameters, calcCorr = False, start_date='2008-01-01', useORATS=False):
    sig_set = []
    trade_market = []
    df_sig_cache = {}
    df_spot_cache = {}
    param_features = []

    for param_set in all_parameters['key']:

        param_v = eval(param_set)
        signal_markets = param_v[0]
        trade_market = param_v[1]
        p = {
            "series": [param_v[2]],
            "delta_region": 'inside',
            "put_call_op": param_v[8],
            "signal_map": param_v[9],
            "signal_ma": param_v[7],
            "near_exp": param_v[5],
            "far_exp": param_v[6]
        }
        dlow = param_v[3]
        dhigh = param_v[4]

        print(param_v)

        if trade_market not in df_spot_cache:
            df_temp = query_spot_data([trade_market])
            df_spot_cache[trade_market] = df_temp
            kurtosis_underlying = df_temp.loc[:, pd.IndexSlice[:, 'pct_ret']].kurtosis(axis=0).values[0]

        if ''.join(signal_markets) not in df_sig_cache:
            metadata = MetaData()
            engine = create_engine(os.getenv('DATABASE_CONNECTION'))
            metadata.reflect(bind=engine)
            if useORATS:
                sd = Table('v_sig_data_hist', metadata, autoload=True, autoload_with=engine)
            else:
                sd = Table('sig_data', metadata, autoload=True, autoload_with=engine)
            session_factory = sessionmaker(bind=engine)
            session = scoped_session(session_factory)
            df_temp = query_signal_data(session, sd, signal_markets, start_date, '2030-01-01')
            df_sig_cache[''.join(signal_markets)] = df_temp
            session.remove()
            engine.dispose()

        df_spot = df_spot_cache[trade_market].copy(deep=True)

        df_sig = df_sig_cache[''.join(signal_markets)].copy(deep=True)
        df_sig = prepare_options_data(df_sig, trade_market)

        idx = (df_sig['delta'] >= dlow) & (df_sig['delta'] <= dhigh)
        df_temp_sig = df_sig[idx]

        indicator_params = p

        r = calc_signal_deltavol_v1(df_temp_sig, p)

        if calcCorr:
            sharpe_drop, df_ret = trade_signal(r, df_spot)
            name_str = get_indicator_name(indicator_params)
            df_ret.name = name_str

            corr_with_underlying = pd.concat([df_ret, df_spot.loc[:, pd.IndexSlice[:, 'pct_ret']]], axis=1).dropna().corr()
            temp_result = get_standard_performance_metrics(df_ret.to_frame())

            param_features.append([corr_with_underlying.values[1][0],
                                   temp_result['kurtosis'].values[0],
                                   temp_result['kurtosis'].values[0]/kurtosis_underlying])

        r.columns = pd.MultiIndex.from_tuples([(trade_market,) + x for x in r.columns])
        r.columns.names = ['mkt', 'param', 'stage']
        sig_set.append(r.copy())

    param_features = pd.DataFrame(param_features, columns=['corr_w_under', 'kurtosis', 'kurtosis_ratio'])

    return sig_set, trade_market, param_features


## Indicators Start ---------------------------------------------------------

def RSI(stock, column="adj_close", period=14, usesign=False):
    # Wilder's RSI
    close = stock[column]

    if usesign:
        delta = np.sign(close.pct_change())
    else:
        delta = close.pct_change()

    up, down = delta.copy(), delta.copy()

    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the exponential moving averages (EWMA)
    roll_up = up.ewm(com=period - 1, adjust=False).mean()
    roll_down = down.ewm(com=period - 1, adjust=False).mean().abs()

    # Calculate RS based on exponential moving average (EWMA)
    rs = roll_up / roll_down  # relative strength =  average gain/average loss

    rsi = 100 - (100 / (1 + rs))
    stock['RSI'] = rsi

    return stock


def ultosc(px, t1, t2, t3):
    temp = px.copy()
    high = px['high'].to_numpy()
    low = px['low'].to_numpy()
    close = px['close'].to_numpy()
    temp['ind'] = ta.ULTOSC(high, low, close, timeperiod1=t1, timeperiod2=t2, timeperiod3=t3)
    temp['ind'] = z_score(temp['ind'], 252)
    return (-temp['ind']).clip(1,-1)

def mfi(px, t1):
    temp = px.copy()
    high = px['high'].to_numpy()
    low = px['low'].to_numpy()
    close = px['close'].to_numpy()
    volume = px['volume'].to_numpy()
    temp['ind'] = ta.MFI(high, low, close, volume, timeperiod=t1)
    temp['ind'] = rm.z_score(temp['ind'], 252)
    return temp['ind'].clip(1,-1)

def bbp(px, n):
    up, mid, low = ta.BBANDS(px['adj_close'].to_numpy(), timeperiod=n, nbdevup=2, nbdevdn=2, matype=0)
    bbp = (((px['adj_close'] - low) / (up - low))*1.2).clip(1,-1)
    return bbp

def adosc(px, fast, slow):
    temp = px.copy()
    high = px['high'].to_numpy()
    low = px['low'].to_numpy()
    close = px['close'].to_numpy()
    volume = px['volume'].to_numpy()
    temp['real_ad'] = ta.ADOSC(high, low, close, volume, fastperiod=fast, slowperiod=slow)
    return temp['real_ad'].clip(1,-1)

def obv(px):
    temp = px.copy()
    close = px['close'].to_numpy()
    volume = px['volume'].to_numpy()
    temp['ind'] = ta.OBV(close, volume)
    temp['ind'] = rm.z_score(temp['ind'].pct_change(), 15)
    return temp['ind'].clip(1,-1)

def rsi(df_in, n):
    df_temp = df_in.copy()
    df_temp['RSI'] = ta.RSI(df_in['adj_close'].to_numpy(), timeperiod=n)
    sig = ((df_temp['RSI'] - 30) * 6).clip(100, 0) / 100
    return sig

def williams_pct(df_in, n):
    df_temp = df_in.copy()
    x = ta.WILLR(df_temp.high, df_temp.low, df_temp.close, timeperiod=n)
    x = np.clip((x+85)/35,1,-1)
    return x

## Indicators End ---------------------------------------------------------

def z_score(df_in, lookback):
    col_mean = df_in.rolling(lookback).mean()
    col_std = df_in.rolling(lookback).std()
    return (df_in - col_mean) / col_std


def z_score_keep_sign(df_in, lookback):
    col_mean = df_in.rolling(lookback).mean()
    col_std = df_in.rolling(lookback).std()
    df_in_sign = np.sign(df_in)
    return df_in_sign * np.abs(df_in - col_mean) / col_std


def calc_credit_risk_signal():
    return


def calc_signal(df, cat, min_days_exp, max_days_exp, lookback, method):
    idx = (df['days_to_expiry'] >= min_days_exp) & (df['days_to_expiry'] <= max_days_exp)
    df = df[idx]

    z = df.groupby(['date', 'symbol', 'delta_region', 'call_put'])[[cat]].sum()
    z = z.unstack(level='call_put')

    if method == 'diff':
        z = z[(cat, 'P')] - z[(cat, 'C')]
    elif method == 'sum':
        z = z[(cat, 'P')] + z[(cat, 'C')]
    else:
        ratio = np.abs(z[(cat, 'P')] / z[(cat, 'C')])
        idx = ratio < 0.00001
        ratio[idx] = 0.00001
        z = np.log(ratio)

    z = z.unstack(level='symbol')
    z = z.unstack(level='delta_region')
    z.columns = z.columns.droplevel()

    v = z.replace(-np.inf, np.nan).replace(np.inf, np.nan).fillna(method='ffill').fillna(0)
    r = z_score(v, 504).rolling(lookback).mean()

    r.rename(columns={True: 'inside', False: 'outside'}, inplace=True)

    return r


def get_indicator_name(indicator_params):
    signal_map = indicator_params['signal_map']
    ma = indicator_params['signal_ma']

    s = ".".join(indicator_params['series'])
    near = "{:02.0f}".format(indicator_params['near_exp'])
    far = "{:03.0f}".format(indicator_params['far_exp'])
    op = indicator_params['put_call_op']
    subtext = s + "." + near + "." + far + "." + op + "." + f'{ma}.{signal_map}'

    return subtext


def save_params(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_params(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_param_set_names(params):
    names = []

    low_delta = "{:02.0f}".format(params['lower_delta'] * 100)
    high_delta = "{:02.0f}".format(params['upper_delta'] * 100)
    maintext = params['trade_market'] + "." + low_delta + "." + high_delta

    for key, value in params['indicator_params'].items():
        subtext = get_indicator_name(value)
        names.append(maintext + "." + subtext)

    return names


def calc_signal_deltavol_v1(df_p1, indicator_params):
    series = indicator_params['series']
    method = indicator_params['put_call_op']
    signal_map = indicator_params['signal_map']
    ma = indicator_params['signal_ma']
    near = indicator_params['near_exp']
    far = indicator_params['far_exp']
    delta_region = indicator_params['delta_region']
    name = get_indicator_name(indicator_params)

    labels = series
    results1 = [calc_signal(df_p1, x, near, far, ma, method) for x in labels]
    results1 = [x.reset_index() for x in results1]
    results1 = [x.set_index('date') for x in results1]

    results2 = [x.loc[:, delta_region].to_frame() for x in results1]
    results3 = [x.rename(columns={delta_region: y}) for x, y in zip(results2, labels)]

    r_base = pd.concat(results3, axis=1)
    r = r_base
    r.loc[:, 'sig'] = r.mean(axis=1).to_frame()

    if 3 == signal_map:
        r['lev_val'] = 0.5
        idx_increase_risk = r.loc[:, 'sig'] >= 0.3
        idx_decrease_risk = r.loc[:, 'sig'] < -0.3
        r.loc[idx_increase_risk, 'lev_val'] = 1.0
        r.loc[idx_decrease_risk, 'lev_val'] = 0
    elif 4 == signal_map:
        r.loc[:, 'lev_val'] = (np.tanh(r.loc[:, 'sig'] * 2) + 1) / 2
    elif 5 == signal_map:
        r.loc[:, 'lev_val'] = (np.tanh(r.loc[:, 'sig'] * 3) + 1) / 2
    else:
        r['lev_val'] = 0.75
        idx_increase_risk = r.loc[:, 'sig'] >= 0
        idx_decrease_risk = r.loc[:, 'sig'] < 0
        r.loc[idx_increase_risk, 'lev_val'] = 1.0
        r.loc[idx_decrease_risk, 'lev_val'] = 0.25

    r.name = name
    r.columns = pd.MultiIndex.from_product([[r.name], r.columns])
    r.columns.names=['param', 'stage']
    return r

###############################
## Pandas DataFrame utilities
def get_next_trade_date(last_date):
    us_calendar = tc.get_calendar('XNYS')
    next_date = us_calendar.next_open(last_date + pd.DateOffset(hours=18))
    next_date = next_date.floor('d').tz_localize(None)
    return next_date


def get_previous_trade_date(curr_date):
    us_calendar = tc.get_calendar('XNYS')
    prev_date = us_calendar.previous_close(curr_date)
    prev_date = prev_date.floor('d').tz_localize(None)
    return prev_date


def get_n_trade_dates_future(curr_date, num_days):
    us_calendar = tc.get_calendar('XNYS')

    date_count = 0
    test_date = curr_date

    i = 0
    while date_count < num_days:
        test_date = test_date + pd.DateOffset(days=1)
        if us_calendar.is_session(test_date):
            date_count += 1
        i += 1
        if i > (10 + num_days):
            break

    return test_date


def adjust_held_for_next_open(df_in):
    # adjust a dataframe to include the next trade date as the final row
    df_out = df_in.copy()
    last_date = pd.Timestamp(df_in.tail(1).index[0])
    next_date = get_next_trade_date(last_date)

    new_row = (df_out.tail(1) * 0).reset_index()
    new_row.date = next_date
    new_row = new_row.set_index('date')

    df_out = df_out.append(new_row)
    df_out = df_out.shift(1)
    # remove the first row
    df_out = df_out.drop(df_out.index[0])
    return df_out


def pd_plot_per_level_index(df_in, idx):
    s = pd.IndexSlice
    names = df_in.columns.get_level_values(idx).unique()
    for n in names:
        if idx == 0:
            ax = df_in.loc[:,s[n,:,:]].plot(figsize=(10,10))
        elif idx == 1:
            ax = df_in.loc[:,s[:,n,:]].plot(figsize=(10,10))
        elif idx == 2:
            ax = df_in.loc[:,s[:,:,n]].plot(figsize=(10,10))

        ax.set_title(n)
    return

def pd_plot_per_level(df_in, level_name):
    mkts = df_in.columns.get_level_values(level=level_name).unique()
    for l in mkts:
        ax = df_in[l].plot(figsize=(10,10))
        ax.set_title(l)
    return

def pd_add_col_level(df_in, level_name, column_name):
    df_in.columns = pd.MultiIndex.from_tuples([(column_name,) + x for x in df_in.columns], names=(level_name,) + df_in.columns.names)
    return df_in


def pd_pivot_dayofweek(df_in):
    idx_day = df_in.index.day_name()
    # the normal call to dt.week will return to ISO week num which for
    # Dec 31 on a Monday means the week num is 1, not 52
    idx_week = list(map(lambda x: x.strftime("%U"), df_in.index))
    idx_year = df_in.index.year

    idx = pd.MultiIndex.from_arrays([idx_year, idx_week, idx_day], names=('year', 'week', 'dayofweek'))
    df_in.index = idx
    df_in = df_in.unstack(level='dayofweek')
    return df_in

# This is used to adjust option interest history to reflect the premium service
# of getting t-1 open interest at 4am
def lead_open_interest(df_iv):
    df_temp = df_iv.reset_index(drop=True).set_index(['symbol', 'date', 'option_expiration', 'strike', 'call_put']).copy()
    df_oi = df_temp.groupby(['symbol', 'option_expiration', 'strike', 'call_put'])['open_interest'].shift(-1)
    df_temp['open_interest'] = df_oi
    return df_temp

###############################
## Measurment and Stats


def prepare_options_data(df_p1, symbol):

    df_p1['symbol'] = symbol
    df_p1.set_index('date', inplace=True)
    df_p1['volume'] = df_p1['volume'].replace(0, 0.01)
    # add days to expiry
    df_p1.reset_index(inplace=True)
    df_p1['days_to_expiry'] = (df_p1['option_expiration'] - df_p1['date']) / np.timedelta64(1, 'D')
    df_p1 = df_p1.set_index(['date', 'symbol', 'delta_region', 'option_expiration', 'call_put'])

    return df_p1


def get_standard_trade_metrics(df_positions):
    # multi-level indices enabled
    num_traded_days = (np.abs(df_positions.diff().dropna()) > 1e-8).sum(axis=0)
    pct_days_traded = pd.DataFrame((num_traded_days / df_positions.shape[0]), columns=['pct_days_traded'])

    num_traded_days = (np.abs(df_positions.diff().dropna())).sum(axis=0)
    avg_change_per_day = pd.DataFrame((num_traded_days / df_positions.shape[0]), columns=['avg_change_per_day'])
    df_stats = pd.concat([pct_days_traded, avg_change_per_day], axis=1)

    df_stats = df_stats.sort_index()

    return df_stats


def get_standard_performance_metrics(df_returns):
    # multi-level indices enabled
    # Get stats on the market and strategy
    temp = df_returns
    sharpe = pd.DataFrame(temp.mean(axis=0) / temp.std(axis=0) * np.sqrt(252), columns=['sharpe'])
    kurtosis = pd.DataFrame(temp.kurtosis(axis=0), columns=['kurtosis'])
    skew = pd.DataFrame(temp.skew(axis=0), columns=['skew'])
    annual_vol = pd.DataFrame(temp.std(axis=0), columns=['annual_vol']) * np.sqrt(252)
    total_return = pd.DataFrame(temp.sum(axis=0)*100, columns=['% return'])
    df_stats = pd.concat([sharpe, kurtosis, skew, annual_vol, total_return], axis=1)

    df_stats = df_stats.sort_index()

    return df_stats

def trade_signal(r, spot):

    s = pd.IndexSlice
    pnl = spot.loc[:, s[:, 'pct_ret']]
    pnl.columns = pnl.columns.droplevel(level='mkt')

    try:
        r_temp = r.copy()
        r_temp.columns = r_temp.columns.droplevel(level='param')
        r_temp.rename(columns={"lev_val": "psn"}, inplace=True)
        pnl = pd.merge(pnl, r_temp['psn'].shift(2), on='date', how='inner')
    except:
        print('arhh')

    pnl['gamma_ret'] = pnl['psn'] * pnl['pct_ret']
    #pnl = pnl.fillna(0)
    s = calc_sharpe_drop(pnl['gamma_ret'].to_frame())

    return s, pnl['gamma_ret']


def trade_signal_openclose(r, spot):

    s = pd.IndexSlice
    pnl = spot.loc[:, s[:, 'pct_openclose_ret']]
    pnl.columns = pnl.columns.droplevel(level='mkt')

    try:
        r_temp = r.copy()
        r_temp.columns = r_temp.columns.droplevel(level='param')
        r_temp.rename(columns={"lev_val": "psn"}, inplace=True)
        pnl = pd.merge(pnl, r_temp['psn'].shift(1), on='date', how='inner')
    except:
        print('arhh')

    pnl['gamma_ret'] = pnl['psn'] * pnl['pct_openclose_ret']
    #pnl = pnl.fillna(0)
    s = calc_sharpe_drop(pnl['gamma_ret'].to_frame())

    return s, pnl['gamma_ret']


def calc_parkinson_vol(df_in, n):

    # parkinson is a measure of vol which uses high/low data
    c = 1# / (4 * np.log(2))
    x = np.power(np.log(df_in['high']/df_in['low']), 2) * c
#    x = np.log(df_in['high']/df_in['low'])
    df_out = x.rolling(n).mean()

    return df_out


# This measure shows the change in Sharpe after the top 1% of the largest returns are removed.
def calc_sharpe_drop(df_in):

    df_in = df_in.dropna()
    idx_end = df_in.shape[0] + 1
    # remove 1% of the largest positive returns
    idx_start = int(idx_end * 0.99)

    # pull numpy array and sort all columns independently
    np_sorted = np.sort(df_in.values, axis=0)
    np_sorted = np_sorted[:idx_start]

    temp = np_sorted
    impaired_sharpe = (np.nanmean(temp, axis=0) / np.nanstd(temp, axis=0) * np.sqrt(252))

    temp = df_in.values
    normal_sharpe = (np.nanmean(temp, axis=0) / np.nanstd(temp, axis=0) * np.sqrt(252))

    sharpe_drop = impaired_sharpe - normal_sharpe

    col_names = ['']
    if any([x == 'mkt' for x in df_in.columns.names]):
        col_names = df_in.columns.get_level_values('mkt')
    else:
        col_names = df_in.columns

    df_out = pd.DataFrame(np.stack((normal_sharpe, impaired_sharpe, sharpe_drop)),
                          index=['normal', 'impaired', 'difference'],
                          columns=col_names)
    df_out.name = 'Sharpe Impairment Analysis'

    return df_out.T


def rolling_sharpe(df_in, lookback = 1008, annualize=True):
    m = df_in.rolling(lookback).mean()
    s = df_in.rolling(lookback).std()
    df_out = m/s

    if annualize:
        df_out = df_out * np.sqrt(252)

    return df_out


def indicator_parkinson_ratio(df_in, n):

    # The difference between a parkinson measure and std
    # is used to adjust leverage so that we get smaller tails.
    # The output is a leverage factor which goes from 0.5
    # to 1.

    y = calc_parkinson_vol(df_in, n).to_frame()
    z = df_in['adj close'].pct_change().rolling(n).std()
    y = y.join(z)
    y.columns = ['pk', 'std']
    col_name = 'leverage' + "_" + df_in.name
    y[col_name] = (y['std'] / y['pk']).diff(1).rolling(50).mean()
    df_out = z_score(y[col_name], 252).to_frame()
    df_out[col_name] = np.tanh(df_out[col_name]) * 0.5 + 0.75
    df_out.name = df_in.name

    return df_out

###############################
## DATA Capture and Pre-Process

def get_report_folder_name():

    engine = create_engine(os.getenv('DATABASE_CONNECTION'))
    session_factory = sessionmaker(bind=engine)
    metadata = MetaData()
    metadata.reflect(bind=engine)
    ep = Table('equity_price', metadata, autoload=True, autoload_with=engine)
    session = scoped_session(session_factory)

    ss = session.query(func.max(ep.c.date).filter(ep.c.local_symbol=='SPX'))
    last_trade_day = ss.all()[0][0]
    next_date = get_next_trade_date(last_trade_day)

    session.remove()
    engine.dispose()

    report_date_folder_name = next_date.strftime('%Y%m%d')
    with open('data.config.json') as f:
        data_config = json.load(f)
    outputDataRoot = os.path.join(data_config['PROCESSED_DATA'],report_date_folder_name)

    try:
        os.mkdir(outputDataRoot)
    except FileExistsError:
        print('directory exists...')
    return outputDataRoot

def get_quandl_data(ticker):
    with open('config.json') as config_file:
        data = json.load(config_file)
    quandl.ApiConfig.api_key = data['QUANDL_KEY']
    df = quandl.get(ticker)
    return df


def load_gamma_file(filename_in, symbol):
    parse_dates = ['date', 'option_expiration']
    df_orig = pd.read_csv(filename_in, parse_dates=parse_dates)
    df_p1 = df_orig.copy()
    # some data series such as QQQ need both QQQ and QQQQ tickers this makes sure we aggregate correctly
    return df_p1


def get_yahoo_data(ticker, start_date='20190101', end_date='20200813'):

    raw_data = yqd.load_yahoo_quote(ticker, start_date, end_date)
    # search for null and only use lines with no null
    reg_result = [not re.findall(r"null", line) for line in raw_data]
    raw_data = list(compress(raw_data, reg_result))
    parsed_data = [sub.split(",") for sub in raw_data]
    header = parsed_data[0]
    df = pd.DataFrame(parsed_data[1:], columns=header)
    df.columns = [str.lower(t) for t in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.rename(columns={'adj close': 'adj_close'})
    df.name = ticker
    df = df.reset_index().drop_duplicates(subset='date').set_index('date')

    return df


def get_tiingo_data(ticker, key, start_date='20190101', end_date='20200813'):
    requestResponse = requests.get(
        f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&token={key}&format=csv")

    TESTDATA = io.StringIO(requestResponse.text)
    df = pd.read_csv(TESTDATA)

    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.set_index('date')
    df = df.drop(['divCash', 'splitFactor', 'volume', 'adjOpen', 'adjHigh', 'adjLow'], axis=1)
    df = df.rename(columns={'adjVolume': 'volume'})
    df = df.rename(columns={'adjClose': 'adj_close'})

    df.columns = [str.lower(t) for t in df.columns]
    df.name = ticker

    return df

def query_and_insert_stock_data(local_symbol, source_symbol, cur_date_str, ep, session, start_date_str):

    engine = create_engine(os.getenv('DATABASE_CONNECTION'))

    with open('config.json') as f:
        std_config = json.load(f)
    tiingo_token = std_config['TIINGO_KEY']

    # this means it is an index and we need to pull it from Yahoo.com
    if source_symbol[0] == '^':
        print(f'yahoo data pull: {source_symbol}')
        df = get_yahoo_data(source_symbol, start_date=start_date_str, end_date=cur_date_str)
    else:
        print(f'tiingo data pull: {source_symbol}')
        df = get_tiingo_data(source_symbol, tiingo_token, start_date=start_date_str)

    df.reset_index(inplace=True)

    df['source'] = 'yahoo.com'
    df['local_symbol'] = local_symbol
    try:
        min_date = df['date'].min()
        stmt = ep.delete().where(and_(ep.c.date >= min_date, ep.c.local_symbol == local_symbol))
        engine.execute(stmt)
        session.commit()

        df.to_sql('equity_price', engine, if_exists='append', index=False)
        logging.info('Inserted %s rows.' % len(df))
    except:
        tb = traceback.format_exc()
        logging.error(tb)
        pass

    engine.dispose()
    return


def query_spot_data(symbol):
    engine = create_engine(os.getenv('DATABASE_CONNECTION'))
    session_factory = sessionmaker(bind=engine)

    metadata = MetaData()
    metadata.reflect(bind=engine)
    ep = Table('equity_price', metadata, autoload=True, autoload_with=engine)
    session = scoped_session(session_factory)
    df_spot = query_equity_price_data(session, ep, symbol)
    session.remove()
    engine.dispose()

    spot = df_spot.loc[:, ['local_symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']].copy()
    spot.set_index('date', inplace=True)
    # this aligns dates between spot and df_p1
    spot['pct_ret'] = spot['adj_close'].pct_change().fillna(0)
    spot['pct_openclose_ret'] = ((spot['close'] - spot['open']) / spot['close'].shift(1)).fillna(0)
    spot['pct_closeopen_ret'] = spot['pct_ret']-spot['pct_openclose_ret']
    spot['pct_closeopen_nodiv_ret'] = spot['close'].pct_change().fillna(0)-spot['pct_openclose_ret']
    spot['pct_highlow_ret'] = ((spot['high'] - spot['low']) / spot['close'].shift(1)).fillna(0)
    spot['dol_ret'] = spot['adj_close'].diff().fillna(0)
    spot['dol_openclose_ret'] = ((spot['close'] - spot['open'])).fillna(0)
    spot['dol_closeopen_ret'] = (spot['adj_close'].diff() - spot['dol_openclose_ret']).fillna(0)
    spot['dol_closeopen_nodiv_ret'] = (spot['close'].diff() - spot['dol_openclose_ret']).fillna(0)
    spot['1yr_sd'] = spot['pct_ret'].rolling(256).std()
    spot['1mo_sd'] = spot['pct_ret'].rolling(21).std()
    spot['vol_err'] = spot['1mo_sd'] - spot['1yr_sd']
    spot.fillna(0, inplace=True)
    spot.columns = pd.MultiIndex.from_product((symbol, spot.columns), names= ['mkt', 'stage'])

    return spot

def query_equity_price_data(session, ep, symbols):

    s = session.query(ep). \
        filter(ep.c.local_symbol.in_((symbols))). \
        order_by(ep.c.date)

    df_out = pd.read_sql(s.statement, s.session.bind)
    df_out['date'] = pd.to_datetime(df_out['date'])

    return df_out


def query_impliedvol_data(session, iv, symbols, eod, days_to_expiry=91, strike_threshold=0.02):
    assert (strike_threshold > 0)
    s = session.query(iv.c.date, iv.c.symbol, iv.c.option_expiration, iv.c.strike, iv.c.call_put,
                      iv.c.stock_price_close.label('spot'),
                      iv.c.mean_price, iv.c.iv). \
        filter((func.abs(iv.c.stock_price_close - iv.c.strike) / iv.c.stock_price_close) < strike_threshold). \
        filter(iv.c.symbol.in_((symbols))). \
        filter(iv.c.eod == eod). \
        filter(iv.c.iv > 0). \
        order_by(iv.c.date, iv.c.option_expiration)

    df_price_sql = pd.read_sql(s.statement, s.session.bind)
    # filter to nearest 63 days
    df_price_sql['days_to_expiry'] = (df_price_sql['option_expiration'] - df_price_sql['date']) / np.timedelta64(1, 'D')
    df_price_sql = df_price_sql[df_price_sql['days_to_expiry'] < days_to_expiry]

    df_temp = df_price_sql.set_index(['date', 'symbol', 'option_expiration', 'strike', 'call_put']).sort_index()
    df_temp = df_temp.unstack(level='call_put')
    df_iv = df_temp['iv'].groupby('date').mean()
    df_iv['iv_diff'] = df_iv['C'] - df_iv['P']
    df_iv = df_iv.rename(columns={"C": "call_iv", "P": "put_iv"})

    return df_iv

def query_signal_data(session, sd, symbols, start_date='2000-01-01', end_date='2000-01-01'):

    s = session.query(sd.c.date, sd.c.option_expiration, sd.c.call_put, sd.c.delta,
                      func.sum(sd.c.volume).label('volume'),
                      func.sum(sd.c.vol_dol).label('vol_dol'),
                      func.sum(sd.c.oi_dol).label('oi_dol'),
                      func.sum(sd.c.delta_vol_dol).label('delta_vol_dol'),
                      func.sum(sd.c.delta_oi_dol).label('delta_oi_dol'),
                      func.sum(sd.c.gamma_vol_dol).label('gamma_vol_dol'),
                      func.sum(sd.c.gamma_oi_dol).label('gamma_oi_dol'),
                      func.sum(sd.c.theta_vol_dol).label('theta_vol_dol'),
                      func.sum(sd.c.theta_oi_dol).label('theta_oi_dol'),
                      func.sum(sd.c.delta_vol).label('delta_vol'),
                      func.sum(sd.c.delta_oi).label('delta_oi'),
                      func.sum(sd.c.gamma_vol).label('gamma_vol'),
                      func.sum(sd.c.gamma_oi).label('gamma_oi'),
                      func.sum(sd.c.theta_vol).label('theta_vol'),
                      func.sum(sd.c.theta_oi).label('theta_oi')
            ).\
        filter(sd.c.symbol.in_(symbols)).\
        filter(sd.c.eod == True).\
        filter(sd.c.date > start_date).\
        filter(sd.c.date <= end_date).\
        group_by(sd.c.date, sd.c.option_expiration, sd.c.call_put, sd.c.delta).\
        order_by(sd.c.date, sd.c.option_expiration)

    df_sql = pd.read_sql(s.statement, s.session.bind)
    df_sql['date'] = pd.to_datetime(df_sql['date'])
    df_sql['option_expiration'] = pd.to_datetime(df_sql['option_expiration'])
    df_sql['days_to_expiry'] = (df_sql['option_expiration'] - df_sql['date']) / np.timedelta64(1, 'D')
    df_sql['delta_region'] = True

    return df_sql


def query_delta_gamma_data_v2(session, iv, symbols, eod, delta, start_date, end_date):
    lower_abs_delta_threshold = delta
    if lower_abs_delta_threshold == 0.9:
        upper_abs_delta_threshold = 1.01
    else:
        # we are working in tenths and adding 0.1 to 0.3 will cause rounding issues
        # so we need to add 10 to 30 and divide by 100
        upper_abs_delta_threshold = ((100*delta) + 10)/100

    s = session.query(iv.c.date, iv.c.eod, iv.c.symbol, iv.c.option_expiration, iv.c.call_put,
                      func.sum(iv.c.mean_price * iv.c.volume).label('vol_dol'),
                      func.sum(iv.c.mean_price * iv.c.open_interest).label('oi_dol'),
                      func.sum(iv.c.delta * iv.c.mean_price * iv.c.volume).label('delta_vol_dol'),
                      func.sum(iv.c.delta * iv.c.mean_price * iv.c.open_interest).label('delta_oi_dol'),
                      func.sum(iv.c.gamma * iv.c.mean_price * iv.c.volume).label('gamma_vol_dol'),
                      func.sum(iv.c.gamma * iv.c.mean_price * iv.c.open_interest).label('gamma_oi_dol'),
                      func.sum(iv.c.theta * iv.c.mean_price * iv.c.volume).label('theta_vol_dol'),
                      func.sum(iv.c.theta * iv.c.mean_price * iv.c.open_interest).label('theta_oi_dol'),
                      func.sum(iv.c.delta * iv.c.volume).label('delta_vol'),
                      func.sum(iv.c.delta * iv.c.open_interest).label('delta_oi'),
                      func.sum(iv.c.gamma * iv.c.volume).label('gamma_vol'),
                      func.sum(iv.c.gamma * iv.c.open_interest).label('gamma_oi'),
                      func.sum(iv.c.theta * iv.c.volume).label('theta_vol'),
                      func.sum(iv.c.theta * iv.c.open_interest).label('theta_oi'),
                      func.sum(iv.c.volume).label('volume'),
                      func.sum(iv.c.open_interest).label('open_interest')
            ).\
        filter(iv.c.symbol.in_(symbols)).\
        filter(iv.c.eod == eod). \
        filter(iv.c.date > start_date).\
        filter(iv.c.date <= end_date).\
        filter(func.abs(iv.c.delta) >= lower_abs_delta_threshold). \
        filter(func.abs(iv.c.delta) < upper_abs_delta_threshold). \
        group_by(iv.c.date, iv.c.eod, iv.c.symbol, iv.c.option_expiration, iv.c.call_put).\
        order_by(iv.c.date, iv.c.option_expiration)

    df_sql = pd.read_sql(s.statement, s.session.bind)
    df_sql['date'] = pd.to_datetime(df_sql['date'])
    df_sql['option_expiration'] = pd.to_datetime(df_sql['option_expiration'])
    return df_sql

def query_delta_gamma_data(session, iv, symbols, eod,
                           strike_threshold=1,
                           lower_abs_delta_threshold=0.4,
                           upper_abs_delta_threshold=0.6,
                           include_delta_region=True):
    # pull and aggregate everything within strike_threshold of spot with a delta less
    # than the abs_delta threshold

    s = session.query(iv.c.date, iv.c.symbol, iv.c.option_expiration, iv.c.call_put,
                      iv.c.stock_price_close.label('spot'),
                      func.sum(iv.c.mean_price * iv.c.volume).label('vol_dol'),
                      func.sum(iv.c.mean_price * iv.c.open_interest).label('oi_dol'),
                      func.sum(iv.c.delta * iv.c.open_interest).label('delta_oi'),
                      func.sum(iv.c.gamma * iv.c.open_interest).label('gamma_oi'),
                      func.sum(iv.c.delta * iv.c.volume).label('delta_vol'),
                      func.sum(iv.c.gamma * iv.c.volume).label('gamma_vol'),
                      func.sum(iv.c.gamma * iv.c.mean_price * iv.c.volume).label('gamma_dol_vol'),
                      func.sum(iv.c.gamma * iv.c.mean_price * iv.c.open_interest).label('gamma_dol_oi'),
                      func.sum(iv.c.delta * iv.c.mean_price * iv.c.volume).label('delta_dol_vol'),
                      func.sum(iv.c.delta * iv.c.mean_price * iv.c.open_interest).label('delta_dol_oi'),
                      func.sum(iv.c.volume).label('volume'),
                      func.sum(iv.c.open_interest).label('open_interest'),
                      and_(func.abs(iv.c.delta) >= lower_abs_delta_threshold,
                            func.abs(iv.c.delta) <= upper_abs_delta_threshold).label('delta_region')
            ).\
        filter((func.abs(iv.c.stock_price_close - iv.c.strike) / iv.c.stock_price_close) < strike_threshold).\
        filter(iv.c.symbol.in_(symbols)).\
        filter(iv.c.eod == eod).\
        group_by(iv.c.date, iv.c.symbol, iv.c.option_expiration, iv.c.call_put, iv.c.stock_price_close,
                 and_(func.abs(iv.c.delta) >= lower_abs_delta_threshold,
                      func.abs(iv.c.delta) <= upper_abs_delta_threshold)
                 ).\
        order_by(iv.c.date, iv.c.option_expiration)

    df_sql = pd.read_sql(s.statement, s.session.bind)
    df_sql['date'] = pd.to_datetime(df_sql['date'])
    df_sql['option_expiration'] = pd.to_datetime(df_sql['option_expiration'])
    return df_sql


def query_putcall_data(session, iv, symbols, eod):

    s = session.query(iv.c.date, iv.c.symbol, iv.c.option_expiration, iv.c.call_put,
                      iv.c.stock_price_close.label('spot'),
                      iv.c.strike.label('strike'),
                      func.sum(iv.c.volume).label('volume'),
                      func.sum(iv.c.open_interest).label('open_interest'),
                      func.sum(iv.c.delta * iv.c.open_interest).label('eff_delta'),
                      func.sum(iv.c.gamma * iv.c.open_interest).label('eff_gamma')
                      ).\
        filter(iv.c.symbol.in_(symbols)).\
        filter(iv.c.eod == eod).\
        group_by(iv.c.date, iv.c.symbol, iv.c.option_expiration, iv.c.call_put, iv.c.stock_price_close, iv.c.strike).\
        order_by(iv.c.date, iv.c.option_expiration)
    df_sql = pd.read_sql(s.statement, s.session.bind)

    return df_sql



def query_putcall_ratio_data(session, iv, symbols, eod, strike_threshold=0.1):
    assert (strike_threshold > 0)
    s = session.query(iv.c.date, iv.c.symbol, iv.c.option_expiration, iv.c.call_put,
                      iv.c.stock_price_close.label('spot'),
                      iv.c.strike.label('strike'),
                      func.sum(iv.c.volume).label('volume'),
                      func.sum(iv.c.open_interest).label('open_interest'),
                      func.sum(iv.c.delta * iv.c.open_interest).label('eff_delta'),
                      func.sum(iv.c.gamma * iv.c.open_interest).label('eff_gamma')
                      ).\
        filter(iv.c.symbol.in_(symbols)).\
        filter(iv.c.eod == eod).\
        filter((func.abs(iv.c.stock_price_close - iv.c.strike) / iv.c.stock_price_close) <= strike_threshold).\
        group_by(iv.c.date, iv.c.symbol, iv.c.option_expiration, iv.c.call_put, iv.c.stock_price_close, iv.c.strike).\
        order_by(iv.c.date, iv.c.option_expiration)
    df_sql = pd.read_sql(s.statement, s.session.bind)
    return df_sql


def preprocess_data(df_sql, days_to_expiry=183):
    df = df_sql.set_index(['date', 'symbol', 'option_expiration', 'call_put']).sort_index()
    du = df.unstack(level='call_put')
    du['days_to_expiry'] = (du.index.get_level_values('option_expiration') - du.index.get_level_values(
        'date')) / np.timedelta64(1, 'D')
    du['time_weight'] = (du['days_to_expiry'] < days_to_expiry) * 1

    # extract only the nearest expiry dates
    temp = du[du['time_weight'] == 1]
    eff_delta = temp.loc[:, [('eff_delta', 'P'), ('eff_delta', 'C')]].sum(axis=1)
    eff_delta = eff_delta.groupby('date').sum()
    eff_delta.name = 'effDelta'

    # extract only the nearest expiry dates
    temp = du[du['time_weight'] == 1]
    gamma_ratio = temp.loc[:, [('eff_gamma', 'P'), ('eff_gamma', 'C')]].groupby('date').sum()
    gamma_ratio = gamma_ratio[('eff_gamma', 'C')] / gamma_ratio[('eff_gamma', 'P')]
    gamma_ratio = gamma_ratio.fillna(method='ffill')
    gamma_ratio.name = 'gammaRatio'

    # Build Spot Series
    spot = du.loc[:, [('spot', 'C')]].groupby(level='date').max()
    spot.columns = spot.columns.droplevel(level=1)

    spot['pct_ret'] = spot.pct_change().fillna(0)
    spot['ret_2'] = spot.loc[:, ['pct_ret']].apply(lambda x: x.rolling(2).sum())
    spot['ret_5'] = spot.loc[:, ['pct_ret']].apply(lambda x: x.rolling(5).sum())
    spot['ret_10'] = spot.loc[:, ['pct_ret']].apply(lambda x: x.rolling(10).sum())
    spot['ret_20'] = spot.loc[:, ['pct_ret']].apply(lambda x: x.rolling(20).sum())
    spot['ret_50'] = spot.loc[:, ['pct_ret']].apply(lambda x: x.rolling(50).sum())

    spot['ret_sign'] = np.sign(spot['pct_ret'])
    spot['ret_sign5'] = spot.loc[:, ['ret_sign']].apply(lambda x: x.rolling(5).sum())
    spot['ret_sign10'] = spot.loc[:, ['ret_sign']].apply(lambda x: x.rolling(10).sum())
    spot['ret_sign20'] = spot.loc[:, ['ret_sign']].apply(lambda x: x.rolling(20).sum())
    spot['ret_sign50'] = spot.loc[:, ['ret_sign']].apply(lambda x: x.rolling(50).sum())

    spot['1yr_sd'] = spot['pct_ret'].rolling(256).std()
    spot['1qtr_sd'] = spot['pct_ret'].rolling(63).std()

    spot['fut_1ret'] = spot[['pct_ret']].shift(-1)
    spot['fut_2ret'] = spot[['ret_2']].shift(-2)
    spot['fut_5ret'] = spot[['ret_5']].shift(-5)
    spot['fut_10ret'] = spot[['ret_10']].shift(-10)
    spot['fut_20ret'] = spot[['ret_20']].shift(-20)
    spot['fut_50ret'] = spot[['ret_50']].shift(-50)

    spot['fut_1ret_sign'] = spot[['ret_sign']].shift(-1)
    spot['fut_5ret_sign'] = spot[['ret_sign5']].shift(-5)
    spot['fut_10ret_sign'] = spot[['ret_sign10']].shift(-10)
    spot['fut_20ret_sign'] = spot[['ret_sign20']].shift(-20)
    spot['fut_50ret_sign'] = spot[['ret_sign50']].shift(-50)

    # negative means we over estimated risk
    spot['fut_sd1yr_err'] = spot['1yr_sd'] - np.abs(spot['fut_1ret'])
    spot['fut_sd1qtr_err'] = spot['1qtr_sd'] - np.abs(spot['fut_1ret'])
    rolling5daySd = spot['ret_5'].rolling(251).std()
    spot['fut_5sd1yr_err'] = rolling5daySd - np.abs(spot['fut_5ret'])

    # ## Assign Risk Categories
    spot['fut_risk_cat'] = 0
    idx_high = spot['fut_sd1yr_err'] < -0.002
    idx_low = spot['fut_sd1yr_err'] > 0.01
    spot.loc[idx_high, 'fut_risk_cat'] = -1
    spot.loc[idx_low, 'fut_risk_cat'] = 1

    spot['fut_risk_cat5'] = 0
    idx_high = spot['fut_5sd1yr_err'] < -0.002
    idx_low = spot['fut_5sd1yr_err'] > 0.01
    spot.loc[idx_high, 'fut_risk_cat5'] = -1
    spot.loc[idx_low, 'fut_risk_cat5'] = 1

    # Join all the data together
    all_data = spot.join(gamma_ratio)
    all_data = all_data.join(eff_delta)
    all_data = all_data.join(df['open_interest'].groupby('date').sum())

    all_data['gammaRatio_smooth_3'] = all_data['gammaRatio'].rolling(3).mean()
    all_data['deltaSignal_smooth_3'] = all_data['effDelta'].rolling(3).mean()
    all_data['gammaRatio_smooth_5'] = all_data['gammaRatio'].rolling(5).mean()
    all_data['deltaSignal_smooth_5'] = all_data['effDelta'].rolling(5).mean()
    all_data['gammaRatio_smooth_10'] = all_data['gammaRatio'].rolling(10).mean()
    all_data['deltaSignal_smooth_10'] = all_data['effDelta'].rolling(10).mean()
    all_data['gammaRatio_smooth_20'] = all_data['gammaRatio'].rolling(20).mean()
    all_data['deltaSignal_smooth_20'] = all_data['effDelta'].rolling(20).mean()
    all_data['gammaRatio_smooth_50'] = all_data['gammaRatio'].rolling(50).mean()
    all_data['deltaSignal_smooth_50'] = all_data['effDelta'].rolling(50).mean()

    all_data['d_open_interest'] = all_data['open_interest'].diff(1).fillna(0)
    col_mean = all_data['d_open_interest'].rolling(50).mean()
    col_std = all_data['d_open_interest'].rolling(50).std()
    all_data['d_open_interest'] = (all_data['d_open_interest'] - col_mean) / col_std

    all_data['d_gammaRatio'] = all_data['gammaRatio'].diff(1).fillna(0)
    col_mean = all_data['d_gammaRatio'].rolling(100).mean()
    col_std = all_data['d_gammaRatio'].rolling(100).std()
    all_data['d_gammaRatio'] = (all_data['d_gammaRatio'] - col_mean) / col_std

    all_data['sign_gammaRatio'] = np.sign(all_data['gammaRatio'] - 1)
    all_data['sign_gammaRatio'] = all_data['sign_gammaRatio'].rolling(15).sum()

    return all_data

#
# df_sql = query_delta_gamma_data()
# df_iv_temp = query_impliedvol_data(91, 0.02)
#
# df_iv = df_iv_temp.pct_change().fillna(0).add_prefix('pct_chg_')
# df_iv_temp_10 = df_iv_temp.pct_change().fillna(0).add_prefix('pct_chg_10').apply(lambda x: x.rolling(10).sum())
# df_iv_temp_20 = df_iv_temp.pct_change().fillna(0).add_prefix('pct_chg_20').apply(lambda x: x.rolling(20).sum())
# df_iv_temp_50 = df_iv_temp.pct_change().fillna(0).add_prefix('pct_chg_50').apply(lambda x: x.rolling(50).sum())
#
# df_iv = df_iv.join(df_iv_temp_10)
# df_iv = df_iv.join(df_iv_temp_20)
# df_iv = df_iv.join(df_iv_temp_50)
# df_iv.plot(figsize=(15,15),alpha=0.5)
#
# allData = preprocess_data(df_sql, 183)
#
# all_data = allData
# all_data = all_data.join(df_iv)
# #all_data = all_data.join(etf_data)
# #all_data = all_data.join(ted_spread)
#
# # this removes data both at the start and end due to NaNs from 'fut_' series
# training_data = all_data.dropna().copy()
# evaluation_data = all_data.dropna().copy()
#
# ## Pull the target columns from data
# targetcols = evaluation_data.filter(regex='^(fut)|spot|1yr_sd',axis=1).columns
#
# ## ML Starts Here
# scaler = StandardScaler()
# train_qry = 'date < "1-1-2015"'
# in_sample = training_data.query(train_qry)
# temp_in_sample = in_sample.drop(columns=targetcols)
# X = in_sample.drop(columns=targetcols)
#
# scaler.fit(X.values)
# X = scaler.transform(X.values)
#
# X_full = evaluation_data.drop(columns=targetcols)
# output = X_full['pct_ret'].to_frame().copy()
# output['spot'] = evaluation_data['spot']
#
# scaler.fit(X_full.values)
# X_full = scaler.transform(X_full.values)
#
# randomForest_pipe = make_pipeline(
#     StandardScaler(),
#     RandomForestClassifier(n_estimators=500)
# )
#
# adaBoost_pipe = make_pipeline(
#     StandardScaler(),
#     AdaBoostClassifier(n_estimators=500)
# )
#
# MLPRegressor_pipe = make_pipeline(
#     StandardScaler(),
#     MLPRegressor(hidden_layer_sizes = (100,10))
# )
#
# GradientBoostingRegressor_pipe = make_pipeline(
#     StandardScaler(),
#     GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, loss='ls')
# )

