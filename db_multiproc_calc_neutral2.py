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

vol_table = 'v_ivolatility_m'
neutral_by_agg_table = 'neutral_by_agg'
neutral_by_exp_table = 'neutral_by_exp'


def query_ivolatility(session, iv, trade_date, symbol):
    s = session.query(iv.c.symbol, iv.c.date, iv.c.call_put, iv.c.option_expiration, iv.c.strike,
                      iv.c.mean_price, iv.c.iv, iv.c.delta, iv.c.gamma, iv.c.stock_price_close, iv.c.open_interest,
                      iv.c.volume). \
        filter(iv.c.date < iv.c.option_expiration). \
        filter(iv.c.symbol == symbol). \
        filter(iv.c.date == trade_date). \
        filter(iv.c.eod == True)

    df = pd.read_sql(s.statement, s.session.bind, parse_dates=['date', 'option_expiration'])
    return df


def query_us_rfr(session, us, trade_date):
    # always pull a range, fill forward and then return the date because some us rate data in
    # 2018-10 is missing
    d0 = trade_date - dt.timedelta(days=5)
    d1 = trade_date + dt.timedelta(days=5)
    r = pd.date_range(start=d0, end=d1)

    s = session.query(us.c.date, us.c.tenor, us.c.rate).filter(us.c.date < d1). \
        filter(us.c.date > d0). \
        filter(us.c.tenor == '1y')
    df = pd.read_sql(s.statement, s.session.bind)
    df = df.set_index('date').reindex(r).ffill()
    # pandas transposes a series so I need to transpose back
    df = df.loc[trade_date.strftime('%Y-%m-%d')].to_frame().T
    return df


def query_spot_data(session, eq, trade_date, root_symbol):
    s = session.query(eq.c.close). \
        filter(eq.c.date == trade_date). \
        filter(eq.c.local_symbol == root_symbol)

    df = pd.read_sql(s.statement, s.session.bind)
    return df['close'].iloc[0]


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


def insert_neutral_data(p):

    query_date = p[0]
    root_symbol = p[1]
    status_str = query_date.strftime('%y-%m-%d') + "," + root_symbol
    print(status_str)

    with open('data.config.json') as f:
        data_config = json.load(f)
    engine = create_engine(data_config['DATABASE_CONNECTION'])
    session_factory = sessionmaker(bind=engine)
    session = scoped_session(session_factory)

    metadata = MetaData()
    metadata.reflect(bind=engine)
    iv = Table(vol_table, metadata, autoload=True, autoload_with=engine)
    us = Table('us_treasury_yield', metadata, autoload=True, autoload_with=engine)

    df_iv = query_ivolatility(session, iv, query_date, root_symbol)

    df_us = query_us_rfr(session, us, query_date)
    rf_rate = df_us['rate'][0]

    s = session.query(iv.c.strike). \
        filter(iv.c.symbol == root_symbol).filter(iv.c.date == query_date). \
        distinct()
    unique_strikes = sorted([value for (value,) in s.all()], key=float)
    spot_prices = np.array(calc_spot_price_levels(unique_strikes), dtype=float)

    agg_neutral, df_greeks_by_strike, exp_neutral = calc_daily_neutral_values(df_iv, query_date, rf_rate, root_symbol,
                                                                              spot_prices)

    try:
        agg_neutral.to_sql(neutral_by_agg_table, engine, if_exists='append', index=True, method='multi')
    except:
        logging.exception()
        logging.error('agg_neutral' + status_str)

    try:
        exp_neutral.to_sql(neutral_by_exp_table, engine, if_exists='append', index=True, method='multi')
    except:
        logging.exception()
        logging.error('exp_neutral ' + status_str)

    session.remove()
    engine.dispose()

    return


def calc_daily_neutral_values(df_iv, target_date, rf_rate, root_symbol, spot_prices, calc_only_greeks=False):

    df_iv_file = df_iv.copy()
    deltalist = []
    gammalist = []
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
        df_greeks_by_strike, rr, tt = calc_greeks_by_strike(calendar, day_count, deltalist, df_iv_file,
                                                            dividend_yield, flat_ts, gammalist, root_symbol,
                                                            spot_prices, trade_date)

        if calc_only_greeks:
            agg_neutral = 0
            exp_neutral = 0
        else:
            agg_neutral, exp_neutral = calc_all_neutrals(target_date, root_symbol, rr, trade_date, tt)

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


def calc_greeks_by_strike(calendar, day_count, deltalist, df_iv_file, dividend_yield, flat_ts, gammalist,
                          root_symbol, spot_prices, trade_date):
    as_of_date = ql.Date(trade_date.day, trade_date.month, trade_date.year)
    for idx_exp, df_exp in df_iv_file.groupby('option_expiration'):
        maturity_date = ql.Date(idx_exp.day, idx_exp.month, idx_exp.year)

        exercise = ql.AmericanExercise(as_of_date, maturity_date)#RWM

        for idx_row, df_row in df_exp.iterrows():

            # construct the European Option
            option_type = mapOptionType[df_row.call_put]
            strike = df_row.strike

            # worth 0.1 to 0.2 seconds
            payoff = ql.PlainVanillaPayoff(option_type, strike)
            european_option = ql.VanillaOption(payoff, exercise)

            # worth 0.5 seconds
            volatility = 0.1#RWM df_row['iv']
            flat_vol_ts = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(as_of_date, calendar, volatility, day_count)
            )

            for spot in spot_prices:

                spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))

                process = ql.GeneralizedBlackScholesProcess(spot_handle,
                                                            dividend_yield,
                                                            flat_ts,
                                                            flat_vol_ts)


                european_option.setPricingEngine(ql.AnalyticDigitalAmericanEngine(process))
                delta = european_option.delta()
                gamma = european_option.gamma()
                deltalist.append([idx_exp, spot, option_type, delta * df_row['open_interest'], delta * df_row['volume']])
                gammalist.append(
                    [idx_exp, spot, option_type, option_type * gamma * df_row['open_interest'],
                     gamma * df_row['volume']])
    tt = pd.DataFrame(deltalist, columns=['option_expiration', 'spot', 'call_put', 'oi_delta', 'vol_delta'])
    rr = pd.DataFrame(gammalist, columns=['option_expiration', 'spot', 'call_put', 'oi_gamma', 'vol_gamma'])
    temp = pd.concat(
        [tt.groupby(['option_expiration', 'spot', 'call_put']).sum(), rr.groupby(['option_expiration', 'spot', 'call_put']).sum()], axis=1)
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

    return df_greeks_by_strike, rr, tt


###########################################################################################
# Insert neutrals
def calc_and_insert_neutral_values(use_orats_data=False):
    # get the set of trade_date, root_symbol, expiry and option_type which needs calculation
    with open('data.config.json') as f:
        data_config = json.load(f)
    engine = create_engine(data_config['DATABASE_CONNECTION'])
    session_factory = sessionmaker(bind=engine)

    metadata = MetaData()
    metadata.reflect(bind=engine)
    session = scoped_session(session_factory)

    global vol_table
    global neutral_by_agg_table
    global neutral_by_exp_table

    if use_orats_data:
        vol_table = 'v_orats_as_ivolatility'
        neutral_by_agg_table = 'neutral_by_agg_orats'
        neutral_by_exp_table = 'neutral_by_exp_orats'
    else:
        vol_table = 'v_ivolatility_m'
        neutral_by_agg_table = 'neutral_by_agg'
        neutral_by_exp_table = 'neutral_by_exp'

    iv = Table(vol_table, metadata, autoload=True, autoload_with=engine)
    nbe = Table(neutral_by_exp_table, metadata, autoload=True, autoload_with=engine)

    with open('data.config.json') as f:
        data_config = json.load(f)
    start_trade_date = data_config['START_DATE']

    params = list()
    start_trade_date = dt.datetime(1990, 1, 1)

    # Find option values which are missing neutral_data table. Only look at expiry dates which are past today's
    # date and on Friday. Use the IV table as the golden source. This operates on a day by day basis.
    q1 = session.query(iv.c.date, iv.c.symbol, iv.c.option_expiration). \
        filter(iv.c.date > start_trade_date). \
        filter(or_(iv.c.open_interest > 0, iv.c.volume > 0)). \
        filter(iv.c.date < iv.c.option_expiration).distinct().subquery()
    q2 = session.query(nbe.c.date, nbe.c.symbol).filter(
        nbe.c.date > start_trade_date).subquery()
    s = session.query(q1.c.date, q1.c.symbol). \
        join(q2, (q2.c.symbol == q1.c.symbol) & (q2.c.date == q1.c.date), isouter=True). \
        filter(q2.c.date == None). \
        filter(q1.c.date > start_trade_date). \
        distinct()

    listOfResults = s.all()
    for op_data in listOfResults:
        params.append([op_data[0], op_data[1]])

    session.remove()
    engine.dispose()

    print("%s calc of num Items:%d" % (__file__, len(params)))

    if profile_me:
        for p in params:
            insert_neutral_data(p)
    else:

        p = Pool(10)
        p.map(insert_neutral_data, params)
        p.close()  # No more work
        p.join()  # Wait for completion



if __name__ == '__main__':
    calc_and_insert_neutral_values(use_orats_data=True)
