import os
import schedule
import time


def job():
    """
    Sends a command line call to execute main script
    """
    os.system(
        "python3 main.py --connect 'y' --feecode 0000 --siteid 0 --multi 'y' --WoW y --hourly 12 --model_name Prophet-Multi --substier y")


schedule.every().day.at("3:00").do(job)
schedule.every().day.at("9:00").do(job)
schedule.every().day.at("15:00").do(job)
schedule.every().day.at("21:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
