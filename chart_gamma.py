import pandas as pd
import json
from datetime import datetime, timedelta
import os
from multiprocessing import Pool
import logging
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sqlalchemy import create_engine, Table, MetaData, or_, except_, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session
import IV_RiskMitigation_v1 as rm
import db_multiproc_calc_neutral2 as dmcn

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(filename='gamma_info.log', level=logging.INFO, format=FORMAT)

win_spike = lambda x: (np.median(x))

def plot_delta_gamma_results(pp, df_oi, df_greeks, chart_title, spot, gamma_text, delta_text, df_alt=""):
    fig = create_plot_delta_gamma_results(df_oi, df_greeks, chart_title, spot, gamma_text, delta_text, df_alt)
    pp.savefig(fig)
    plt.close(fig)

    return


def get_gamma_by_strike(root_symbol, trade_date):
    with open('data.config.json') as f:
        data_config = json.load(f)
    engine = create_engine(data_config['DATABASE_CONNECTION'])
    session_factory = sessionmaker(bind=engine)
    session = scoped_session(session_factory)

    metadata = MetaData()
    metadata.reflect(bind=engine)
    iv = Table(vol_table, metadata, autoload=True, autoload_with=engine)
    us = Table('us_treasury_yield', metadata, autoload=True, autoload_with=engine)

    df_iv = dmcn.query_ivolatility(session, iv, trade_date, root_symbol)

    df_us = dmcn.query_us_rfr(session, us, trade_date)
    rf_rate = df_us['rate'][0]

    s = session.query(iv.c.strike). \
        filter(iv.c.symbol == root_symbol).filter(iv.c.date == trade_date). \
        distinct()
    unique_strikes = sorted([value for (value,) in s.all()], key=float)
    spot_prices = np.array(dmcn.calc_spot_price_levels(unique_strikes), dtype=float)

    agg_neutral, df_greeks_by_strike, exp_neutral = dmcn.calc_daily_neutral_values(df_iv, trade_date, rf_rate,
                                                                                   root_symbol,
                                                                                   spot_prices,
                                                                                   calc_only_greeks=True)
    df = df_greeks_by_strike.copy().reset_index()
    session.remove()
    engine.dispose()

    return df


def prepare_gamma_dynamics_data(root_symbol, query_date, k_chart_pct_range_limit, session, ep, iv, nba):
    slc = pd.IndexSlice

    last_price = query_spot_data(session, ep, query_date, root_symbol)
    s = session.query(iv.c.date, iv.c.symbol, iv.c.call_put, iv.c.strike, iv.c.option_expiration, iv.c.open_interest). \
        filter(iv.c.date < iv.c.option_expiration). \
        filter(iv.c.date == query_date). \
        filter(iv.c.symbol == root_symbol)

    df_oi = pd.read_sql(s.statement, s.session.bind)
    df_oi.set_index(['date', 'symbol', 'call_put', 'strike', 'option_expiration'], inplace=True)
    df_oi.sort_index(axis=0, inplace=True)

    df = get_gamma_by_strike(root_symbol, query_date)
    df['last_price'] = last_price
    df['pct_diff'] = (df['spot'] / df['last_price']) - 1

    df_gamma = df.set_index(['date', 'symbol', 'call_put', 'spot'])
    # open_interest_gamma is sum(oi*gamma) and multiplying by last_price * 100 gives dollars
    df_gamma['open_interest_gamma'] = df_gamma['open_interest_gamma'] * last_price * 100
    far_expiry_idx = df_gamma['option_expiration'] != df_gamma['option_expiration'].min()
    spot_range_idx = np.abs(df_gamma['pct_diff']) < k_chart_pct_range_limit
    max_strike = df_gamma[spot_range_idx].reset_index()['spot'].max()
    min_strike = df_gamma[spot_range_idx].reset_index()['spot'].min()

    df_temp = df_oi.reset_index().set_index(['strike', 'call_put'])
    near_expiry_idx = df_temp['option_expiration'] == df_temp['option_expiration'].min()
    df_temp = df_temp[near_expiry_idx].unstack('call_put').loc[:, slc['open_interest', :]].droplevel(axis=1, level=0)
    df_temp.rename(columns={'C': 'Call Open Interest', 'P': 'Put Open Interest'}, inplace=True)
    strikes = df_temp.reset_index()['strike']
    idx_oi = (strikes >= min_strike) & (strikes <= max_strike)

    df_temp = df_temp.rolling(5, center=True).max()  # .rolling(5, center=True).mean()
    df_temp = df_temp.reset_index().loc[idx_oi, :].set_index('strike')

    df_gamma_all = df_gamma.loc[spot_range_idx, 'open_interest_gamma'].groupby(['call_put', 'spot']).sum().unstack(
        'call_put')
    df_gamma_all.rename(columns={'C': 'Total Call', 'P': 'Total Put'}, inplace=True)
    df_gamma_all['Total Net'] = df_gamma_all.sum(axis=1)

    df_gamma_nonear = df_gamma.loc[far_expiry_idx & spot_range_idx, 'open_interest_gamma'].groupby(
        ['call_put', 'spot']).sum().unstack('call_put')
    df_gamma_nonear.rename(columns={'C': 'Call-no near', 'P': 'Put-no near'}, inplace=True)
    df_gamma_nonear['Net-no near'] = df_gamma_nonear.sum(axis=1)

    s = session.query(nba). \
        filter(nba.c.date == query_date). \
        filter(nba.c.symbol == root_symbol). \
        filter(nba.c.expiry_group == -1)
    df_neutral = pd.read_sql(s.statement, s.session.bind)

    n = df_neutral
    neutral_value = n['gamma_neutral'].max()
    max_value = df_gamma_all['Total Net'].max()

    gamma_text = '\n'.join((
        f'max = {max_value:,.0f}',
        f'neutral = {neutral_value:,.2f}'))

    close_price_text = ''.join(r'close=%.2f' % (last_price,))
    date_str = query_date.strftime('%Y-%m-%d')
    chart_title = f'{root_symbol} Gamma Dynamics at close of {date_str}'

    data = {"df_gamma_all": df_gamma_all,
            "df_gamma_nonear": df_gamma_nonear,
            "df_temp": df_temp,
            "gamma_text": gamma_text,
            "chart_title": chart_title,
            "close_price_text": close_price_text,
            "last_price": last_price
            }
    return data


def create_gamma_dynamics_figure(data):
    df_gamma_all = data["df_gamma_all"]
    df_gamma_nonear = data["df_gamma_nonear"]
    df_temp = data["df_temp"]
    gamma_text = data["gamma_text"]
    chart_title = data["chart_title"]
    close_price_text = data["close_price_text"]
    last_price = data["last_price"]

    fsize = (15, 9)
    fig, ax = plt.subplots(1, figsize=fsize)

    min_x = df_temp.reset_index().loc[:, 'strike'].min()
    max_x = df_temp.reset_index().loc[:, 'strike'].max()
    plt.xlim(min_x, max_x)

    c = {'Total Net': 'k', 'Total Call': 'tab:blue', 'Total Put': 'tab:red', 'Net-no near': 'k', \
         'Call-no near': 'tab:blue', 'Put-no near': 'tab:red', 'Call Open Interest': 'tab:blue', \
         'Put Open Interest': 'tab:red'}
    df_gamma_all.plot(color=c, ax=ax)

    df_gamma_nonear.plot(ax=ax, alpha=.40, color=c)
    ax.axvline(last_price, color='gray', linestyle='dashed')
    ax.axhline(0, color='gray', linestyle='solid')

    ax.set_title(chart_title, fontsize=15)
    ax.set_ylabel('Gamma at Strike')
    ax.set_xlabel('Strike')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(b=True, color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='k', linestyle=':')

    ax1 = ax.twinx()

    df_temp.plot.area(ax=ax1, color=c, alpha=0.3, legend=False)

    ax1.grid(b=True, which='minor', color='lightgray', linestyle='--')

    lines, labels = ax.get_legend_handles_labels()
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax.legend(lines + lines1, labels + labels1, loc='upper right')

    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='wheat')
    ax1.text(0.05, 0.95, gamma_text, transform=ax.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    ax1.set_ylabel('Nearest Open Interest', fontsize=12)

    middleOfData = (np.min(df_gamma_all['Total Put']) * 0.95)
    t = ax.annotate(close_price_text, (last_price, middleOfData), (last_price * 1.01, middleOfData),
                    fontsize=12, arrowprops=dict(arrowstyle='->', mutation_scale=10, color='k'))
    t.set_bbox(dict(facecolor='white'))

    fig.text(0.9, 0.25, 'viking-analytics.com', fontsize=20,
             color='gray', ha='right', va='top',
             alpha=0.5)

    return fig


def create_plot_delta_gamma_results(df_oi, df_greeks, chart_title, spot, gamma_text, delta_text, df_alt=""):
    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    fig.set_size_inches(10, 10)

    y = df_greeks['spot_price']

    close_price_text = ''.join(r'close=%.2f' % (spot,))

    axs[0].set_title(chart_title)

    color = 'tab:red'
    axs[2].set_xlabel('Strike')
    axs[2].xaxis.set_minor_locator(AutoMinorLocator())
    axs[2].grid(b=True, which='minor', color='lightgray', linestyle='--')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)

    # place a text box in upper left in axes coords
    axs[0].text(0.05, 0.95, gamma_text, transform=axs[0].transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

    axs[0].set_ylabel('Gamma Details')
    color = 'tab:red'
    axs[0].plot(y, df_greeks['effective_gamma Put'], label='PUT gamma', color=color, linestyle='--')
    color = 'tab:blue'
    axs[0].plot(y, df_greeks['effective_gamma Call'], label='CALL gamma', color=color, linestyle='--')

    color = 'k'
    axs[0].plot(y, df_greeks['total_gamma'], label='effective_gamma_all', color=color)

    try:
        color = 'tab:green'
        axs[0].plot(y, df_alt['total_gamma'], label='effective_gamma_minus_near', color=color)
    except:
        pass

    axs[0].axhline(0, color='k', linestyle='dashed')
    axs[0].legend(loc='upper right')
    axs[0].axvline(spot, color='gray', linestyle='dashed')

    middleOfData = (np.min(df_greeks['effective_gamma Put']))

    axs[0].annotate(close_price_text, (spot, middleOfData), (spot * 1.1, middleOfData),
                    arrowprops=dict(arrowstyle='->'))
    axs[0].set_xlabel('Strike')
    axs[0].grid(True)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0].grid(b=True, which='minor', color='lightgray', linestyle='--')

    callmask = df_oi['option_type'] == 'Call'
    putmask = df_oi['option_type'] == 'Put'
    s = df_oi['strike']
    oi = df_oi['open_interest']
    w = (max(s) - min(s)) / 50
    axs[1].bar(s[putmask], oi[putmask], color='tab:red', width=w, align='center', label='Puts')
    axs[1].bar(s[callmask], oi[callmask], color='tab:blue', width=w, align='center', label='Calls', alpha=0.8)
    axs[1].set_ylabel('Open Interest')
    axs[1].grid(True)
    axs[1].legend(loc='upper right')
    axs[1].axvline(spot, color='gray', linestyle='dashed')
    axs[1].grid(b=True, which='minor', color='lightgray', linestyle='--')

    color = 'tab:red'
    # place a text box in upper left in axes coords
    axs[2].text(0.05, 0.95, delta_text, transform=axs[2].transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

    axs[2].set_ylabel('Delta Details')
    axs[2].plot(y, df_greeks['effective_delta Put'], label='PUT delta', color=color)
    color = 'tab:blue'
    axs[2].plot(y, df_greeks['effective_delta Call'], label='CALL delta', color=color)
    axs[2].tick_params(axis='y', labelcolor=color)
    axs[2].grid(True)

    color = 'tab:gray'
    axs[2].plot(y, df_greeks['total_delta'], label='effective_delta', color=color)
    axs[2].axhline(0, color='k', linestyle='dashed')

    axs[2].legend(loc='upper right')
    axs[2].axvline(spot, color='gray', linestyle='dashed')

    return fig


def query_gamma_data(session, gbs, trade_date, root_symbol):
    if True:

        import db_multiproc_calc_neutral2 as dmcn

        with open('data.config.json') as f:
            data_config = json.load(f)
        engine = create_engine(data_config['DATABASE_CONNECTION'])
        session_factory = sessionmaker(bind=engine)
        session = scoped_session(session_factory)

        metadata = MetaData()
        metadata.reflect(bind=engine)
        iv = Table(vol_table, metadata, autoload=True, autoload_with=engine)
        us = Table('us_treasury_yield', metadata, autoload=True, autoload_with=engine)

        df_iv = dmcn.query_ivolatility(session, iv, trade_date, root_symbol)

        df_us = dmcn.query_us_rfr(session, us, trade_date)
        rf_rate = df_us['rate'][0]

        s = session.query(iv.c.strike). \
            filter(iv.c.symbol == root_symbol).filter(iv.c.date == trade_date). \
            distinct()
        unique_strikes = sorted([value for (value,) in s.all()], key=float)
        spot_prices = np.array(dmcn.calc_spot_price_levels(unique_strikes), dtype=float)

        agg_neutral, df_greeks_by_strike, exp_neutral = dmcn.calc_daily_neutral_values(df_iv, trade_date, rf_rate,
                                                                                       root_symbol,
                                                                                       spot_prices,
                                                                                       calc_only_greeks=True)
        df = df_greeks_by_strike.copy().reset_index()
        session.remove()
        engine.dispose()

    else:
        s = session.query(gbs). \
            filter(gbs.c.date == trade_date). \
            filter(gbs.c.symbol == root_symbol)
        df = pd.read_sql(s.statement, s.session.bind)

    df = df.rename(columns={
        'date': 'trade_date',
        'symbol': 'root_symbol',
        'option_expiration': 'expiry',
        'open_interest_delta': 'effective_delta',
        'open_interest_gamma': 'effective_gamma',
        'call_put': 'option_type',
        'spot': 'spot_price'
    })

    idxCall = df['option_type'] == 'C'
    idxPut = df['option_type'] == 'P'
    df.loc[idxCall, 'option_type'] = 'Call'
    df.loc[idxPut, 'option_type'] = 'Put'

    return df


def query_open_interest(session, opm, trade_date, root_symbol):
    s = session.query(opm.c.open_interest, opm.c.option_type, opm.c.strike, opm.c.expiry). \
        filter(opm.c.trade_date < opm.c.expiry). \
        filter(opm.c.trade_date == trade_date). \
        filter(opm.c.root_symbol == root_symbol)

    df = pd.read_sql(s.statement, s.session.bind)
    return df


def query_spot_data(session, eq, trade_date, root_symbol):
    s = session.query(eq.c.close). \
        filter(eq.c.date == trade_date). \
        filter(eq.c.local_symbol == root_symbol)

    df = pd.read_sql(s.statement, s.session.bind)
    try:
        ret_value = df['close'].iloc[0]
    except:
        ret_value = 1

    return ret_value


def process_file(param):
    with open('data.config.json') as f:
        data_config = json.load(f)
    engine = create_engine(data_config['DATABASE_CONNECTION'])
    session_factory = sessionmaker(bind=engine)
    metadata = MetaData()
    metadata.reflect(bind=engine)
    opm = Table(option_price_market_table, metadata, autoload=True, autoload_with=engine)
    ivm = Table(vol_table, metadata, autoload=True, autoload_with=engine)
    eq = Table('equity_price', metadata, autoload=True, autoload_with=engine)
    nbe = Table(neutral_by_exp_table, metadata, autoload=True, autoload_with=engine)
    nba = Table(neutral_by_agg_table, metadata, autoload=True, autoload_with=engine)

    # gbs = Table('greeks_by_strike', metadata, autoload=True, autoload_with=engine)

    session = scoped_session(session_factory)

    query_date = param[0]
    run_date = (query_date + datetime.timedelta(days=1)).strftime('%Y%m%d')
    root_symbol = param[1]
    out_data_root = param[2]

    logging.info("dataprep_start:chart_multiproc:" + root_symbol + ":" + run_date)
    print('process id:', os.getpid())

    try:
        spot = query_spot_data(session, eq, query_date, root_symbol)

        full_path = os.path.join(out_data_root, root_symbol + "_" + run_date + "_report.pdf")

        df_oi = query_open_interest(session, opm, query_date, root_symbol)
        #df_oi = query_open_interest(session, ivm, query_date, root_symbol)

        logging.info("chart_start:chart_multiproc:" + root_symbol + ":" + run_date)

        if df_oi.empty:
            logging.error("Error with " + run_date + "in charting.")
        else:
            pp = PdfPages(full_path)
            # iterate over expiry...
            df_gamma = query_gamma_data(session, 0, query_date, root_symbol)

            df_greeks = df_gamma
            df_oi_temp = df_oi
            chart_title, delta_text, df, gamma_text = create_plot_details(True, df_greeks, "", nba, nbe, query_date,
                                                                          root_symbol, session)

            idx = df_greeks['expiry'] != df_greeks['expiry'].min()
            df_greeks = df_gamma.loc[idx]
            chart_title_dummy, delta_text_dummy, df_withoutNear, gamma_text_dummy = \
                create_plot_details(True, df_greeks, "", nba, nbe, query_date, root_symbol, session)

            plot_delta_gamma_results(pp, df_oi_temp, df, chart_title, spot, gamma_text, delta_text, df_withoutNear)

            i = 0
            for idx, df_greeks in df_gamma.groupby('expiry'):
                expiry = df_greeks['expiry'].iloc[0]

                if expiry.dayofweek == 4:
                    df_oi_temp = df_oi.loc[df_oi['expiry'] == expiry, :]

                    chart_title, delta_text, df, gamma_text = create_plot_details(False, df_greeks, expiry, nba, nbe,
                                                                                  query_date,
                                                                                  root_symbol, session)

                    plot_delta_gamma_results(pp, df_oi_temp, df, chart_title, spot, gamma_text, delta_text)

                    i = i + 1
                    if i > 20:
                        break

            pp.close()
    except:
        logging.info("ERROR chart_multiproc:" + root_symbol + ":" + run_date)
        pass

    engine.dispose()
    session.remove()

    return


def create_plot_details(plot_agg, df_greeks, expiry, nba, nbe, query_date, root_symbol, session):
    # calculate total delta and gammafactor
    df = df_greeks.copy()
    idx = df['effective_gamma'] == 0
    df.loc[idx, 'effective_gamma'] = 1e-8
    # pivot data to aggregate effective delta and gamma columns
    df = pd.pivot_table(df, values=['effective_delta', 'effective_gamma'], index=['spot_price'],
                        columns=['option_type'], aggfunc=np.sum)
    total_delta = df.loc[:, 'effective_delta']['Call'] + df.loc[:, 'effective_delta']['Put']
    df['total_delta'] = total_delta
    total_gamma = df.loc[:, 'effective_gamma']['Call'] + df.loc[:, 'effective_gamma']['Put']
    df['total_gamma'] = total_gamma
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    df.reset_index(inplace=True)

    if plot_agg:
        chart_title = 'Gamma Dynamics - ' + root_symbol + ' on ' + query_date.strftime('%Y%m%d') + ' All Dates'

        s = session.query(nba). \
            filter(nba.c.date == query_date). \
            filter(nba.c.symbol == root_symbol). \
            filter(nba.c.expiry_group == -1)
        df_neutral = pd.read_sql(s.statement, s.session.bind)
        try:
            n = df_neutral  # .loc[expiry]
            neutral_value = n['gamma_neutral']
            slope_value = n['gamma_slope']
            max_value = total_gamma.max()

            gamma_text = '\n'.join((
                r'max = %.2f' % (max_value,),
                r'neutral = %.2f' % (neutral_value,),
                r'slope = %.4f' % (slope_value,)))

            neutral_value = n['delta_neutral']
            slope_value = n['delta_slope']
            delta_text = '\n'.join((
                r'neutral = %.2f' % (neutral_value,),
                r'slope = %.4f' % (slope_value,)))

        except:
            logging.error('No neutral value')
            gamma_text = ''
            delta_text = ''
    else:
        chart_title = 'Gamma Dynamics - ' + root_symbol + ' on ' + query_date.strftime(
            '%Y%m%d') + ' Exp:' + expiry.strftime('%Y%m%d')

        s = session.query(nbe). \
            filter(nbe.c.date == query_date). \
            filter(nbe.c.symbol == root_symbol)
        df_neutral = pd.read_sql(s.statement, s.session.bind)
        df_neutral.set_index('option_expiration', inplace=True)

        try:
            n = df_neutral.loc[expiry]
            neutral_value = n['gamma_neutral']
            slope_value = n['gamma_slope']
            max_value = total_gamma.max()

            gamma_text = '\n'.join((
                r'max = %.2f' % (max_value,),
                r'neutral = %.2f' % (neutral_value,),
                r'slope = %.4f' % (slope_value,)))

            neutral_value = n['delta_neutral']
            slope_value = n['delta_slope']
            delta_text = '\n'.join((
                r'neutral = %.2f' % (neutral_value,),
                r'slope = %.4f' % (slope_value,)))

        except:
            logging.error('No neutral value')
            gamma_text = ''
            delta_text = ''
    return chart_title, delta_text, df, gamma_text

