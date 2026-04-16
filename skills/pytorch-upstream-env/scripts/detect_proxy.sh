#!/bin/bash
# Detect which Intel proxy works by testing connectivity
# Returns the first working proxy URL

set -e

PROXIES=(
    "http://proxy.ims.intel.com:911"
    "http://proxy-prc.intel.com:913"
    "http://child-ir.intel.com:912"
    "http://proxy-chain.intel.com:912"
    "http://proxy-dmz.intel.com:911"
    "http://proxy-us.intel.com:912"
)

TEST_URL="https://github.com"

echo "Testing proxy connectivity..."

for proxy in "${PROXIES[@]}"; do
    echo -n "  Testing $proxy ... "
    if curl -s -o /dev/null -w "%{http_code}" --proxy "$proxy" --connect-timeout 5 --max-time 10 "$TEST_URL" 2>/dev/null | grep -q "^[23]"; then
        echo "OK"
        echo "WORKING_PROXY=$proxy"
        exit 0
    else
        echo "FAILED"
    fi
done

echo "ERROR: No working proxy found. Please configure manually."
exit 1
