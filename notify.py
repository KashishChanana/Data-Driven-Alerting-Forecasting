import smtplib
from email.mime.text import MIMEText


def send_email_notifications(fee_code, listing_site_id, alerts):
    """
    Sends out emails to mentioned recipients in the event an inconsistency is observed.

    Parameters
    ----------------
    :param fee_code: code of fee_type being analyzed
    :param listing_site_id : listing site id on eBay of the site being analyzed
    :param alerts : alert text
    """

    sender = 'pricingml@ebay.com'
    # Edit this to change recipients
    receiver = ['kchanana@ebay.com']

    smtp_server_name = '@hostname'
    port = '00'
    server = smtplib.SMTP('{}:{}'.format(smtp_server_name, port))
    server.starttls()

    for alert in alerts:
        msg = MIMEText(alert)
        msg['From'] = sender
        msg['To'] = ", ".join(receiver)
        msg['Subject'] = '[ML Alert] for fee ' + str(fee_code) + " for siteID " + str(listing_site_id)

        server.sendmail(msg['From'], receiver, msg.as_string())
    server.quit()
