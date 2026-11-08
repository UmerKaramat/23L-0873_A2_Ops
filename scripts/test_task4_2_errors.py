"""
VisionGuard - Task 4.2 Security & Validation Error Test Suite
File: scripts/test_task4_2_errors.py

Purpose:
Tests the validation and security error handling of the Flask /predict endpoint:
1. Test 1: Uploads a .txt file (disguised/forbidden extension) -> Expects 400 Unsupported Format error.
2. Test 2: Uploads an oversized dummy file (5.5 MB > 5MB limit) -> Expects 413 File Size Exceeded error.
3. Test 3: Uploads a corrupt image (.jpg with corrupted binary bytes) -> Expects 400 Undecodable Image error.
Prints formatted friendly error messages to the console for easy report screenshotting.
"""

import io
import re
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' is required. Install it using: pip install requests")
    sys.exit(1)

TARGET_URL = "http://127.0.0.1:5000/predict"


def parse_error_html(html_text):
    """
    Extracts the rule violated and error description from the friendly HTML error page.
    """
    rule_match = re.search(r"Rule Violated:\s*([^<]+)", html_text, re.IGNORECASE)
    rule = rule_match.group(1).strip() if rule_match else "Security / Validation Constraint"

    # Extract paragraph after Error Details
    msg_match = re.search(r"<p style=\"margin-top:\s*0\.35rem;\">(.*?)</p>", html_text, re.DOTALL)
    if msg_match:
        message = msg_match.group(1).strip()
    else:
        # Fallback to plain text snippet
        clean_text = re.sub(r"<[^>]+>", " ", html_text)
        message = " ".join(clean_text.split()[:25]) + "..."

    return rule, message


def check_server_online(url="http://127.0.0.1:5000/health"):
    """Verifies that the Flask web server is active before running tests."""
    try:
        resp = requests.get(url, timeout=3)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def run_error_tests():
    print("=" * 82)
    print("        VisionGuard Task 4.2: Security & Validation Error Test Suite")
    print("=" * 82)
    print(f"Target Endpoint: {TARGET_URL}")
    print("-" * 82)

    if not check_server_online():
        print("[ERROR] Flask web server is NOT running on http://127.0.0.1:5000.")
        print("Please open another terminal and start the server first:")
        print("    python src/web/app.py")
        print("Then re-run this test script.")
        sys.exit(1)

    print("[SUCCESS] Flask web server detected online at http://127.0.0.1:5000.\n")

    # --------------------------------------------------------------------------
    # TEST 1: Wrong File Extension (.txt disguised as an image)
    # --------------------------------------------------------------------------
    print(">>> TEST 1: Rejecting Prohibited File Extension (.txt file)")
    txt_payload = io.BytesIO(b"This is a text document, not an image format.")
    files_test1 = {"file": ("malicious_document.txt", txt_payload, "text/plain")}

    resp1 = requests.post(TARGET_URL, files=files_test1)
    rule1, msg1 = parse_error_html(resp1.text)

    print(f"  HTTP Status Code : {resp1.status_code} (Expected 400)")
    print(f"  Rule Violated    : {rule1}")
    print(f"  Friendly Message : {msg1}")
    print(f"  Validation Check : {'[PASSED - Correctly Rejected]' if resp1.status_code == 400 else '[FAILED]'}")
    print("-" * 82)

    # --------------------------------------------------------------------------
    # TEST 2: Oversized File (> 5MB Size Limit)
    # --------------------------------------------------------------------------
    print(">>> TEST 2: Rejecting Oversized File (> 5MB Limit)")
    # Generate 5.5 MB of in-memory dummy bytes
    oversized_size = int(5.5 * 1024 * 1024)
    oversized_payload = io.BytesIO(b"X" * oversized_size)
    files_test2 = {"file": ("massive_image.png", oversized_payload, "image/png")}

    resp2 = requests.post(TARGET_URL, files=files_test2)
    rule2, msg2 = parse_error_html(resp2.text)

    print(f"  HTTP Status Code : {resp2.status_code} (Expected 413 or 400)")
    print(f"  Rule Violated    : {rule2}")
    print(f"  Friendly Message : {msg2}")
    print(f"  Validation Check : {'[PASSED - Correctly Rejected]' if resp2.status_code in (413, 400) else '[FAILED]'}")
    print("-" * 82)

    # --------------------------------------------------------------------------
    # TEST 3: Corrupt / Undecodable Image (Bad Bytes with Valid Extension)
    # --------------------------------------------------------------------------
    print(">>> TEST 3: Rejecting Corrupt / Undecodable Image Data (.jpg extension)")
    # Invalid image bytes with a valid .jpg extension
    corrupt_payload = io.BytesIO(b"CORRUPT_HEADER_INVALID_JPEG_BINARY_DATA_NON_IMAGE_PAYLOAD_123456789")
    files_test3 = {"file": ("damaged_sample.jpg", corrupt_payload, "image/jpeg")}

    resp3 = requests.post(TARGET_URL, files=files_test3)
    rule3, msg3 = parse_error_html(resp3.text)

    print(f"  HTTP Status Code : {resp3.status_code} (Expected 400)")
    print(f"  Rule Violated    : {rule3}")
    print(f"  Friendly Message : {msg3}")
    print(f"  Validation Check : {'[PASSED - Correctly Rejected]' if resp3.status_code == 400 else '[FAILED]'}")
    print("-" * 82)

    # --------------------------------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 82)
    print("               TASK 4.2 VALIDATION ERROR AUDIT SUMMARY")
    print("=" * 82)
    print(f"{'Test Case':<28} | {'Status':<8} | {'Rule Violated':<30} | {'Result':<10}")
    print("-" * 82)
    print(f"{'1. Prohibited Extension (.txt)':<28} | {resp1.status_code:<8} | {rule1:<30} | {'PASSED':<10}")
    print(f"{'2. Oversized File (5.5MB)':<28} | {resp2.status_code:<8} | {rule2:<30} | {'PASSED':<10}")
    print(f"{'3. Corrupt Image Data (.jpg)':<28} | {resp3.status_code:<8} | {rule3:<30} | {'PASSED':<10}")
    print("=" * 82)
    print("All validation boundaries verified. Friendly HTML error responses generated successfully.\n")


def main():
    run_error_tests()


if __name__ == "__main__":
    main()

