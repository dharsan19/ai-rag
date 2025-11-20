# insecure_requests.py
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

def enable_insecure_requests():
    """Call this in startup to globally disable SSL verification on requests-based libs."""
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    original_request = requests.Session.request

    def insecure_request(self, method, url, **kwargs):
        kwargs.setdefault("verify", False)
        return original_request(self, method, url, **kwargs)

    requests.Session.request = insecure_request