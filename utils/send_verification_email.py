import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from dotenv import load_dotenv

load_dotenv()

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "maison.ipe@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "pplahmwvjeywknyw") 

def send_verification_email(to_email: str, code: str):    
    styles = """
        <style>
            body { font-family: Arial, sans-serif; color: #333; line-height: 1.6; font-weight: bold;}
            .container { max-width: 600px; font-weight: bold; margin: auto; padding: 20px; border-radius: 8px; }
            .header { font-size: 20px; color: #333; margin-bottom: 20px; }
            .code { font-size: 30px; color: #cf3a34; margin: 20px 0; text-align: center; }
            .warning { color: #333; margin-top: 10px; }
            .footer { font-size: 14px; color: #888; margin-top: 30px; }
        </style>
    """

    subject = "Your Verification Code"
    body = f"""
        <html><head>{styles}</head><body>
            <div class="container">
                <div class="header">Hello,</div>
                <p>Here is your verification code:</p>
                <div class="code">{code}</div>
                <p class="warning">This code is valid for 15 minutes. Never share this code with anyone.</p>
                <p>Thank you for trusting Skema Canada.</p>
                <div class="footer">Best regards,<br>The Skema Canada Security Team</div>
            </div>
        </body></html>
        """

    message = MIMEMultipart("alternative")
    message["From"] = formataddr(("Skema Security", SMTP_USERNAME))
    message["To"] = to_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "html", _charset="utf-8"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(message)
    except Exception as e:
        raise
