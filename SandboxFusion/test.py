import requests
import json

# Define the endpoint
url = "http://172.26.93.87:10080/run_code"

# Define the test code


new = """
#!/usr/bin/env python3
import sys
import os

def main():
    # Full path of current Python executable
    python_executable = sys.executable
    print("Python executable:", python_executable)

    # Directory containing the Python executable
    python_dir = os.path.dirname(python_executable)
    print("Python directory:", python_dir)

if __name__ == "__main__":
    main()

"""

payload = {
    "code": """
from z3 import *
""",
    "language": "python"
}

# Send the request
response = requests.post(url, json=payload)

# Parse and pretty-print the response
try:
    result = response.json()
    print(json.dumps(result, indent=2))
    
    # Optional: check if the output is correct
    if result.get("run_result", {}).get("stdout") == "Hello, world!\n":
        print("✅ Sandbox is working correctly.")
    else:
        print("❌ Sandbox responded but did not return expected output.")
except Exception as e:
    print("❌ Failed to parse response or connect to sandbox:", str(e))
