"""
C.1.2.1 – Get Theme

This script:
Retreive all themes from OneMap Singapore.

Author: Zhang Wenyu
Date: 2025-12-14
"""

import requests
    
url = "https://www.onemap.gov.sg/api/public/themesvc/getAllThemesInfo"
    
headers = {"Authorization": YOUR_HEADERS}
    
response = requests.request("GET", url, headers=headers)
    
print(response.text)
