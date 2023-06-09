# Calculate and insert greek sweep results

import pandas as pd
import numpy as np
import json
from multiprocessing import Pool
import logging
from sqlalchemy import create_engine, Table, MetaData, or_, except_, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session
import datetime as dt
import QuantLib as ql
import math

mapOptionType = dict()
mapOptionType[1] = ql.Option.Call
mapOptionType[-1] = ql.Option.Put
mapOptionType['Call'] = ql.Option.Call
mapOptionType['Put'] = ql.Option.Put
mapOptionType['C'] = ql.Option.Call
mapOptionType['P'] = ql.Option.Put

inverse_mapOptionType = dict()
inverse_mapOptionType[ql.Option.Call] = 'C'
inverse_mapOptionType[ql.Option.Put] = 'P'

# this is the map for aggregated expiry dates.
k_dateMap = [dt.timedelta(days=64), dt.timedelta(days=128), dt.timedelta(days=4000)]

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(filename='gamma_info.log', level=logging.INFO, format=FORMAT)
profile_me = False

def calc_spot_price_levels(strikes):
    num_strikes = len(strikes)
    strikes = np.sort(strikes)

    target_num = 50
    if num_strikes > target_num:
        sample_freq = int(num_strikes / target_num)
        spot_levels = strikes[1::sample_freq]
        # always include the last point
        if target_num % sample_freq:
            spot_levels = np.unique(np.append(spot_levels, strikes[len(strikes)-1]))

    else:
        spot_levels = strikes

    return spot_levels


def calc_neutral(df_in, grouping_column, data_column):
    r = df_in.groupby([grouping_column, 'spot'])[data_column].sum().to_frame()
    r['sign'] = np.sign(r)

    df_sign = r.groupby(grouping_column)['sign'].diff()
    idxchange = np.where(df_sign > 0)[0]
    rt = r.reset_index()

    if len(idxchange > 0):
        a = rt.iloc[idxchange, :]
        b = rt.iloc[idxchange - 1, :]

        slope = (a.loc[:, data_column] - b.loc[:, data_column].values) / (a.spot - b.spot.values)
        d = a.loc[:, data_column] - a.spot * slope
        x = -d / slope
    else:
        slope = pd.Series(math.nan)
        # all have same sign
        if r['sign'].values[0] > 0:
            x = pd.Series(math.nan)
        else:
            x = pd.Series(0)

    expiry_dates = rt.groupby(grouping_column).sum().reset_index()[grouping_column]
    df_out = pd.DataFrame([x for x in zip(expiry_dates.values, x.values, slope.values)],
                          columns=[grouping_column, 'neutral', 'slope'])
    return df_out


def calc_daily_neutral_values(df_iv, target_date, rf_rate, root_symbol, spot_prices, calc_only_greeks=False):

    df_iv_file = df_iv.copy()
    # this assumes all one trade day
    trade_date = pd.Timestamp(target_date)
    as_of_date = ql.Date(trade_date.day, trade_date.month, trade_date.year)
    # close = df_iv_file.SPOT[0]
    ql.Settings.instance().evaluationDate = as_of_date
    dividend_rate = 0.0 #RWM
    day_count = ql.Actual365Fixed()
    calendar = ql.UnitedStates()
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(as_of_date, rf_rate, day_count)
    )
    dividend_yield = ql.YieldTermStructureHandle(
        ql.FlatForward(as_of_date, dividend_rate, day_count)
    )
    try:
        df_greeks_by_strike, all_spot_gammas, all_spot_deltas = \
            calc_greeks_by_strike(calendar, day_count, df_iv_file,
                                dividend_yield, flat_ts, root_symbol,
                                spot_prices, trade_date)

        if calc_only_greeks:
            agg_neutral = 0
            exp_neutral = 0
        else:
            agg_neutral, exp_neutral = calc_all_neutrals(target_date, root_symbol, all_spot_gammas, trade_date, all_spot_deltas)

    except:
        logging.exception()
        logging.error('MAJOR PROBLEM ')

    return agg_neutral, df_greeks_by_strike, exp_neutral


def calc_all_neutrals(query_date, root_symbol, rr, trade_date, tt):
    # create a map to get date deltas. There are many duplicates and a map is faster
    uniquedate = np.unique(tt['option_expiration'].values)
    datemap = {x: x - trade_date for x in uniquedate}
    tt['daysToExpiry'] = [datemap[x] for x in tt['option_expiration'].values]

    def map_date_range(dateIn):
        idx = [dateIn < x for x in k_dateMap]
        return np.min(np.where(idx))

    uniquedelta = pd.to_timedelta(np.unique(tt['daysToExpiry']))
    expGroupMap = {x: map_date_range(x) for x in uniquedelta}
    tt['expiry_group'] = [expGroupMap[x] for x in tt['daysToExpiry']]
    rr['daysToExpiry'] = tt['daysToExpiry']
    rr['expiry_group'] = tt['expiry_group']
    rr.to_csv('delme.csv')
    eoig = calc_neutral(rr, 'option_expiration', 'oi_gamma').rename(columns={'exp': 'option_expiration'}). \
        set_index('option_expiration').add_prefix('gamma_')
    eoid = calc_neutral(tt, 'option_expiration', 'oi_delta').rename(columns={'exp': 'option_expiration'}). \
        set_index('option_expiration').add_prefix('delta_')
    evd = calc_neutral(tt, 'option_expiration', 'vol_delta').rename(columns={'exp': 'option_expiration'}). \
        set_index('option_expiration').add_prefix('delta_v_')
    oig = calc_neutral(rr, 'expiry_group', 'oi_gamma').set_index('expiry_group').add_prefix('gamma_')
    oid = calc_neutral(tt, 'expiry_group', 'oi_delta').set_index('expiry_group').add_prefix('delta_')
    vd = calc_neutral(tt, 'expiry_group', 'vol_delta').set_index('expiry_group').add_prefix('delta_v_')
    # set expiry_group to all one number so that we calculate results of all expiry dates
    tt['expiry_group'] = -1
    rr['expiry_group'] = tt['expiry_group']
    oig_all = calc_neutral(rr, 'expiry_group', 'oi_gamma').set_index('expiry_group').add_prefix('gamma_')
    oid_all = calc_neutral(tt, 'expiry_group', 'oi_delta').set_index('expiry_group').add_prefix('delta_')
    vd_all = calc_neutral(tt, 'expiry_group', 'vol_delta').set_index('expiry_group').add_prefix('delta_v_')
    exp_neutral = pd.concat([eoig, eoid, evd], axis=1)
    exp_neutral['date'] = query_date
    exp_neutral['symbol'] = root_symbol
    exp_neutral = exp_neutral.rename(columns={'exp': 'option_expiration'})
    quarterly_neutral = pd.concat([oig, oid, vd], axis=1)
    all_neutral = pd.concat([oig_all, oid_all, vd_all], axis=1)
    agg_neutral = pd.concat([quarterly_neutral, all_neutral], axis=0)
    agg_neutral['date'] = query_date
    agg_neutral['symbol'] = root_symbol
    return agg_neutral, exp_neutral


def calc_greeks_by_strike(calendar, day_count, df_iv_file, dividend_yield, flat_ts,
                          root_symbol, spot_prices, trade_date):

    as_of_date = ql.Date(trade_date.day, trade_date.month, trade_date.year)
    gammalist = []
    deltalist = []

    for idx_exp, df_exp in df_iv_file.groupby('option_expiration'):
        maturity_date = ql.Date(idx_exp.day, idx_exp.month, idx_exp.year)

        exercise = ql.EuropeanExercise(maturity_date)

        for idx_row, df_row in df_exp.iterrows():

            # construct the European Option
            option_type = mapOptionType[df_row.call_put]
            strike = df_row.strike

            # worth 0.1 to 0.2 seconds
            payoff = ql.PlainVanillaPayoff(option_type, strike)
            european_option = ql.VanillaOption(payoff, exercise)

            # worth 0.5 seconds
            volatility = df_row['iv']#RWM
            flat_vol_ts = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(as_of_date, calendar, volatility, day_count)
            )

            for spot in spot_prices:

                spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))

                process = ql.GeneralizedBlackScholesProcess(spot_handle,
                                                            dividend_yield,
                                                            flat_ts,
                                                            flat_vol_ts)


                european_option.setPricingEngine(ql.AnalyticEuropeanEngine(process))#RWM
                delta = european_option.delta()
                gamma = european_option.gamma()
                deltalist.append([idx_exp, spot, option_type, delta * df_row['open_interest'],
                                  delta * df_row['volume']])
                gammalist.append([idx_exp, spot, option_type, option_type * gamma * df_row['open_interest'],
                                gamma * df_row['volume']])

    all_spot_deltas = pd.DataFrame(deltalist, columns=['option_expiration', 'spot', 'call_put', 'oi_delta', 'vol_delta'])
    all_spot_gammas = pd.DataFrame(gammalist, columns=['option_expiration', 'spot', 'call_put', 'oi_gamma', 'vol_gamma'])

    temp = pd.concat(
        [all_spot_deltas.groupby(['option_expiration', 'spot', 'call_put']).sum(),
         all_spot_gammas.groupby(['option_expiration', 'spot', 'call_put']).sum()],
        axis=1)

    temp['date'] = pd.to_datetime(trade_date)
    temp['symbol'] = root_symbol

    temp = temp.reset_index().rename(columns={
        'exp': 'option_expiration',
        'oi_delta': 'open_interest_delta',
        'oi_gamma': 'open_interest_gamma',
        'vol_delta': 'volume_delta',
        'vol_gamma': 'volume_gamma',
    })

    temp['call_put'] = temp['call_put'].map(inverse_mapOptionType)

    df_greeks_by_strike = temp.copy()
    df_greeks_by_strike.set_index(['date', 'symbol', 'option_expiration', 'spot', 'call_put'], inplace=True)

    return df_greeks_by_strike, all_spot_gammas, all_spot_deltas

