#pip3 install twython
#log in to https://apps.twitter.com/
#Create a new app
##app_name = sentijaytweet
##app_desc = sentiment analysis on twitter data
##app_site = http://127.0.0.1
#now get these two
#Consumer Key (API Key)    okKrp74oqGZ4y3jxSOFI9Dp3Z
#Consumer Secret (API Secret)    EQWBb5lSWoghYJGtDPa7mJq5vXukep35PKnqoNX8tgZh9ytblp


# from twython import Twython
# import requests
# APP_KEY = 'okKrp74oqGZ4y3jxSOFI9Dp3Z'
# APP_SECRET = 'EQWBb5lSWoghYJGtDPa7mJq5vXukep35PKnqoNX8tgZh9ytblp'
# 
# twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
# ACCESS_TOKEN = twitter.obtain_access_token()
# 
# twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)
# 
# results = twitter.cursor(twitter.search, q='Safety Check')
# for result in results:
#     print(result)

import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
import time

chromedriver = "/usr/local/bin/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver
driver = webdriver.Chrome(chromedriver)
driver.get("https://twitter.com/?lang=en")
t_usr_nm = "jaydeep.chakraborty.1988@gmail.com"
t_pwd = "jaytwitterdeep"
email_field_xpath = "//input[@class='text-input email-input js-signin-email']"
pwd_field_xpath = "//input[@class='text-input']"
home_login_btn_xpath = "//a[@href='/login']"
login_button_xpath = "//input[@class='submit btn primary-btn js-submit']"
tbsearch_box = "//input[@class='search-input']"
tsearch_btn = "//button[@class='Icon Icon--search nav-search']"
tweet_chck = "//li[@data-item-type='tweet']"

home_login_btn_elmnt = WebDriverWait(driver, 10).until(lambda driver: driver.find_element_by_xpath(home_login_btn_xpath))
home_login_btn_elmnt.click()

email_field_elmnt = WebDriverWait(driver, 10).until(lambda driver: driver.find_element_by_xpath(email_field_xpath))
pwd_field_elmnt = WebDriverWait(driver, 10).until(lambda driver: driver.find_element_by_xpath(pwd_field_xpath))
login_btn_elmnt = WebDriverWait(driver, 10).until(lambda driver: driver.find_element_by_xpath(login_button_xpath))

 
email_field_elmnt.clear()
email_field_elmnt.send_keys(t_usr_nm)
pwd_field_elmnt.clear()
pwd_field_elmnt.send_keys(t_pwd)
login_btn_elmnt.click()

search_box_elmnt = WebDriverWait(driver, 10).until(lambda driver: driver.find_element_by_xpath(tbsearch_box))


search_box_elmnt.clear()
#iamsafe
#safety
#safety check
search_box_elmnt.send_keys("marked safe")

search_box_btn_elmnt = WebDriverWait(driver, 30).until(lambda driver: driver.find_element_by_xpath(tsearch_btn))

search_box_btn_elmnt.click()

WebDriverWait(driver, 30)

old = len(driver.find_elements_by_xpath(tweet_chck))
while True:
    print(driver.execute_script("window.scrollTo(0, document.body.scrollHeight); return window.scrollY;"))
#     print(WebDriverWait(driver, 30).until(lambda driver: driver.find_elements_by_xpath("//p[@class = 'TweetTextSize  js-tweet-text tweet-text']")).__len__())
#     print(driver.execute_script("return window.scrollY"))
    time.sleep(5)
    new = len(driver.find_elements_by_xpath(tweet_chck))
    if(new == old):
        break
    old = new


search_vals = driver.find_elements_by_xpath("//p[@class = 'TweetTextSize  js-tweet-text tweet-text']")
file = open("twitter_posts.txt", "w", encoding='utf-8')
for search_val in search_vals:
    file.write(str(search_val.text)+"\n")
    file.write("--------------------------------------------------------------------------------------------\n")

file.close()
driver.close()