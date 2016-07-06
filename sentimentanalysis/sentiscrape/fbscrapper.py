import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait

chromedriver = "/usr/local/bin/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver
driver = webdriver.Chrome(chromedriver)
driver.get("https://facebook.com")
fb_usr_nm = ""
fb_pwd = ""
email_field_id = "email"
pwd_field_id = "pass"
login_button_xpath = "//input[@value='Log In']"
fblogo_xpath = "(//a[contains(@href, 'logo')])[1]"
fbsearch_box = "//div[@class='_586i']"
fbsearch_btn = "//button[@aria-label='Search']"
 
email_field_elmnt = WebDriverWait(driver, 10).until(lambda driver: driver.find_element_by_id(email_field_id))
pwd_field_elmnt = WebDriverWait(driver, 10).until(lambda driver: driver.find_element_by_id(pwd_field_id))
login_btn_elmnt = WebDriverWait(driver, 10).until(lambda driver: driver.find_element_by_xpath(login_button_xpath))

 
email_field_elmnt.clear()
email_field_elmnt.send_keys(fb_usr_nm)
pwd_field_elmnt.clear()
pwd_field_elmnt.send_keys(fb_pwd)
login_btn_elmnt.click()

search_box_elmnt = WebDriverWait(driver, 10).until(lambda driver: driver.find_element_by_xpath(fbsearch_box))


search_box_elmnt.clear()
#iamsafe
#safety
#safety check
search_box_elmnt.send_keys("marked safe")

search_box_btn_elmnt = WebDriverWait(driver, 30).until(lambda driver: driver.find_element_by_xpath(fbsearch_btn))

search_box_btn_elmnt.click()

WebDriverWait(driver, 30)




while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    try:
        if WebDriverWait(driver, 3).until(lambda driver: driver.find_element_by_xpath("//div[@class='phm _64f']")).text == 'End of Results':
            print("scrolling done")
            break
    except:
        pass



search_vals = driver.find_elements_by_xpath("//span[@class = 'highlightNode']/parent::p")

file = open("fb_posts.txt", "w", encoding='utf-8')

for search_val in search_vals:
    file.write(str(search_val.text)+"\n")
    file.write("--------------------------------------------------------------------------------------------\n")

file.close()