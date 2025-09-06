#!/usr/bin/env python3
"""Check if we need mDNS for Wyoming discovery."""
import json
import subprocess

# Check if avahi-browse is available
try:
    result = subprocess.run(["avahi-browse", "-a", "-t", "-r", "-p"], capture_output=True, text=True)
    print("=== mDNS Services ===")
    for line in result.stdout.splitlines():
        if "wyoming" in line.lower() or "_tcp" in line:
            print(line)
except FileNotFoundError:
    print("avahi-browse not found")

# Check HA addon config for discovery
print("\n=== Addon Config ===")
try:
    with open("/data/options.json", "r") as f:
        options = json.load(f)
        print(json.dumps(options, indent=2))
except Exception:
    print("Could not read addon options")

# Check if we're advertising on mDNS
print("\n=== Should we advertise? ===")
print("Home Assistant addons with 'discovery: wyoming' should advertise via mDNS")
print("Service type: _wyoming._tcp")
print("Port: 10300")
