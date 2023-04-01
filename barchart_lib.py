from selenium import webdriver
import os
from selenium.webdriver.common.by import By
import time
import pandas as pd
from random import *

def bar_chart_get_driver(executable_path, download_path):
    chromeOptions = webdriver.ChromeOptions()
    chromeOptions.add_experimental_option("prefs", {
        "download.default_directory": download_path,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })

    driver = webdriver.Chrome(executable_path=executable_path, options=chromeOptions)
    return driver

def bar_chart_login(driver):
    driver.get("https://www.barchart.com/login")
    driver.find_element(By.NAME, "email").click()
    driver.find_element(By.NAME, "email").send_keys(os.getenv("BARCHART_USER"))
    driver.find_element(By.NAME, "password").click()
    driver.find_element(By.NAME, "password").send_keys(os.getenv("BARCHART_PASSWORD"))
    driver.find_element(By.CLASS_NAME, 'login-button').click()
    return


def get_futures_contract_specs(driver, sym):
    contract_root_url = f"https://www.barchart.com/futures/quotes/{sym}/overview"
    option_greek_url = f'https://www.barchart.com/futures/quotes/{sym}/volatility-greeks?moneyness=allRows'

    #-------------
    # open the contract root page
    driver.get(contract_root_url)

    # get the first notice and expiration date
    time.sleep(random()*3)
    # get the expiration of the contract from the option price page
    temp_name = "#main-content-column > div > div.barchart-content-block.commodity-profile > div.block-content > div > div:nth-child(2) > div:nth-child(4) > div.small-7.column.text-right > span > span"
    first_notice = driver.find_element(By.CSS_SELECTOR, temp_name).text
    first_notice = first_notice[0:8]

    temp_name = "#main-content-column > div > div.barchart-content-block.commodity-profile > div.block-content > div > div:nth-child(2) > div:nth-child(5) > div.small-7.column.text-right > span > span"
    futures_expiration = driver.find_element(By.CSS_SELECTOR, temp_name).text
    # make sure we only get the date portion
    futures_expiration = futures_expiration[0:8]

    #-------------
    # open the greeks page to get options contract details. We open each contract page and get the top 4 months
    driver.get(option_greek_url)
    time.sleep(random()+1)
    try:
        # Get the list of options in the dropdown
        driver.find_element(By.ID, "bc-options-toolbar__dropdown-month").click()
        dropdown = driver.find_element(By.ID, "bc-options-toolbar__dropdown-month")
        options = dropdown.text
        options = options.split('\n')
        options = [x.lstrip() for x in options]
        mon_expr=[]
        option_expiration=[]
        # iterate through the options and get the expiration
        for i in [0, 1, 2]:
            time.sleep(random()*3)
            try:
                driver.find_element(By.ID, "bc-options-toolbar__dropdown-month").click()
                dropdown = driver.find_element(By.ID, "bc-options-toolbar__dropdown-month")
                dropdown.find_element(By.XPATH, f"//option[. = '{options[i]}']").click()

                # Get the contract associated with the month
                temp_name = "#main-content-column > div > div.page-title.symbol-header-info.ng-scope > div.symbol-name > div > span:nth-child(2)"
                # note we remove ( and ) by removing the first and last character
                mon_expr.append(driver.find_element(By.CSS_SELECTOR, temp_name).text[1:-1])

                temp_name = "#main-content-column > div > div:nth-child(4) > div > div:nth-child(1) > div > strong:nth-child(2)"
                option_expiration.append(driver.find_element(By.CSS_SELECTOR, temp_name).text)
            except:
                print(options[i] + " not found")
    except:
        mon_expr = ['']
        option_expiration = ['']

    ret_data = [first_notice, futures_expiration, mon_expr,option_expiration]
    return ret_data


def bar_chart_download(driver, date_str, sym, download_path, download=False):

    def on_current_page_download_near_next(driver, download):
        # Get the list of options in the dropdown
        try:
            # attempt to select the contract, if there is no drop down all options on this contract are expired
            driver.find_element(By.ID, "bc-options-toolbar__dropdown-month").click()
        except:
            # select the next contract
            temp = "#main-content-column > div > div.error-page > ul > li:nth-child(1) > a"
            driver.find_element(By.CSS_SELECTOR, temp).click()
            time.sleep(1)
            driver.find_element(By.ID, "bc-options-toolbar__dropdown-month").click()

        dropdown = driver.find_element(By.ID, "bc-options-toolbar__dropdown-month")
        options = dropdown.text
        options = options.split('\n')
        options = [x.lstrip() for x in options]
        mon_expr=[]
        option_expiration=[]
        iv_all=[]

        # iterate through the options and get the expiration
        num_downloaded = 0
        check_loop_counter = 0
        while num_downloaded < 2:
            time.sleep(random()*3)

            try:
                driver.find_element(By.ID, "bc-options-toolbar__dropdown-month").click()
                time.sleep(random()*2)
                dropdown = driver.find_element(By.ID, "bc-options-toolbar__dropdown-month")
                dropdown.find_element(By.XPATH, f"//option[. = '{options[check_loop_counter]}']").click()

                #click all expiries and daily
                time.sleep(random()*0.5)
                driver.find_element(By.NAME, "moneyness").click()
                time.sleep(random()*0.5)
                dropdown = driver.find_element(By.NAME, "moneyness")
                dropdown.find_element(By.XPATH, "//option[. = 'Show All']").click()
                time.sleep(random()*0.5)
                driver.find_element(By.CSS_SELECTOR, ".bc-datatable-toolbar:nth-child(3)").click()
                time.sleep(random()*0.5)
                try:
                    driver.find_element(By.NAME, "futuresOptionsTime").click()
                    time.sleep(random()*0.5)
                    dropdown = driver.find_element(By.NAME, "futuresOptionsTime")
                    dropdown.find_element(By.XPATH, "//option[. = 'Daily']").click()
                except:
                    print('')

                # Get the option expiration date
                temp_name = "#main-content-column > div > div:nth-child(4) > div > div:nth-child(1) > div > strong:nth-child(2)"
                expiration_date = driver.find_element(By.CSS_SELECTOR, temp_name).text

                if pd.to_datetime(expiration_date) > pd.to_datetime(date_str):
                    option_expiration.append(expiration_date)
                    # Get the contract associated with the month
                    temp_name = "#main-content-column > div > div.page-title.symbol-header-info.ng-scope > div.symbol-name > div > span:nth-child(2)"
                    # note we remove ( and ) by removing the first and last character
                    mon_expr.append(driver.find_element(By.CSS_SELECTOR, temp_name).text[1:-1])
                    # Get the IV of all options
                    temp_name = "#main-content-column > div > div:nth-child(4) > div > div.column.small-12.medium-4.text-medium-up-center > div > strong"
                    iv_all.append(driver.find_element(By.CSS_SELECTOR, temp_name).text)


                    if download:
                        time.sleep(random()*3)
                        print(f'downloading {mon_expr[-1]}')
                        driver.find_element(By.CSS_SELECTOR, ".toolbar-button > span").click()
                        time.sleep(3)
                        num_downloaded = num_downloaded + 1

                # make sure we don't loop endlessly
                check_loop_counter = check_loop_counter + 1
                if check_loop_counter > 10:
                    break
            except:
                print(sym+":"+options[check_loop_counter] + " not found")

        return pd.DataFrame(zip(mon_expr, option_expiration, iv_all), columns=['futures_contract', 'option_expiration', 'iv'])

    option_price_url = f"https://www.barchart.com/futures/quotes/{sym}/options?futuresOptionsTime=daily&moneyness=allRows"
    option_greek_url = f'https://www.barchart.com/futures/quotes/{sym}/volatility-greeks?moneyness=allRows'

    #-------------------
    # download the options prices
    time.sleep(2*random())
    driver.get(option_price_url)
    time.sleep(2)

    df_price_data = on_current_page_download_near_next(driver, download)

    # Get the option greeks
    time.sleep(2*random())
    driver.get(option_greek_url)
    time.sleep(2)

    df_greek_data = on_current_page_download_near_next(driver, download)

    df_price_data['source'] = 'price_page'
    df_greek_data['source'] = 'greeks_page'
    df_all = pd.concat([df_price_data, df_greek_data], axis=0)
    filename=os.path.join(download_path, f"{sym}_{date_str}.txt")
    df_all.to_csv(filename)

    # download all pricing files needed
    for ctr in df_all['futures_contract'].unique():
        future_price_url = f'https://www.barchart.com/futures/quotes/{ctr}/historical-download'

        # download the futures prices
        driver.get(future_price_url)

        # download the price history
        time.sleep(2*random())
        driver.find_element(By.CSS_SELECTOR, ".bc-price-history-checkboxes .checkbox").click()
        time.sleep(2)

        if download:
            driver.find_element(By.CSS_SELECTOR, ".add").click()
            time.sleep(3)
    return