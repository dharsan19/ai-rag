import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

def enable_insecure_requests():
    # Disable warnings
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    # Store original request
    original_request = requests.Session.request

    # Override request method
    def insecure_request(self, method, url, **kwargs):
        kwargs["verify"] = False
        return original_request(self, method, url, **kwargs)

    requests.Session.request = insecure_request