import os
from getpass import getpass


def connect():
    """
    Setting up connection to Production Clickhouse DB
    """

    result = None
    f = open("/Users/kchanana/PycharmProjects/pricingml/utils/credentials", "r")
    lines = f.readlines()
    result = ':'.join(line.strip() for line in lines)
    f.close()

    print("\n Authenticating User ... \n")
    proxy_url = "http://" + result + "###### PROXY #####"

    os.environ['http_proxy'] = str(proxy_url)
    os.environ['https_proxy'] = str(proxy_url)


def run_test_query():
    """
    Running a test query to check connection - ONLY FOR TESTING
    """
    res = os.popen("curl -u @username:@password -k "
                   "@host -d 'select * from @Database "
                   "limit "
                   "1'").read()
    print(res)
