import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(sender, app_password, recipient, subject, body, enable=True):
    if not enable:
        print("[DRY RUN] Email body:\n", body); return True
    if not (sender and app_password):
        print("Missing email creds; dry-run body:\n", body); return False
    msg = MIMEMultipart()
    msg["From"] = sender; msg["To"] = recipient; msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP("smtp.gmail.com", 587) as s:
        s.starttls(); s.login(sender, app_password); s.sendmail(sender, recipient, msg.as_string())
    return True
