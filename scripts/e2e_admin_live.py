"""Live test of the admin dataset-upload and retraining flow."""
import re
import sys
import time
import urllib.parse

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from e2e_live import BASE, check, csrf, req, results, session  # noqa: E402  (runs the main e2e suite first)

adm = session()
tok = csrf(adm, "/login")
req(adm, "/login", {"csrf_token": tok, "email": "admin@healthpredict.local", "password": "admin12345"})


def upload(module, filename, content: bytes):
    tok = csrf(adm, "/admin/models")
    boundary = "----e2eboundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"csrf_token\"\r\n\r\n{tok}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"dataset\"; filename=\"{filename}\"\r\n"
            f"Content-Type: text/csv\r\n\r\n").encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return req(adm, f"/admin/models/{module}/upload", body,
               {"Content-Type": f"multipart/form-data; boundary={boundary}"})


code, html, _ = upload("diabetes", "bad.csv", b"a,b,c\n1,2,3\n")
check("admin: CSV with wrong columns is rejected", b"Dataset rejected - missing columns" in html, code)
code, html, _ = upload("diabetes", "notes.txt", b"hello")
check("admin: non-CSV file is rejected", b"Please choose a .csv file" in html, code)

# re-upload the real dataset (exercises backup + replace)
from pathlib import Path  # noqa: E402
data = (Path(__file__).resolve().parents[1] / "data" / "raw" / "diabetes.csv").read_bytes()
code, html, _ = upload("diabetes", "diabetes.csv", data)
check("admin: valid dataset accepted", b"Dataset for Diabetes updated" in html, code)

tok = csrf(adm, "/admin/models")
code, html, _ = req(adm, "/admin/models/diabetes/retrain", {"csrf_token": tok})
check("admin: retrain started", b"Retraining Diabetes started" in html, code)
tok = csrf(adm, "/admin/models")
code, html, _ = req(adm, "/admin/models/diabetes/retrain", {"csrf_token": tok})
check("admin: second retrain while running is refused", b"already running" in html, code)

run_id = None
for _ in range(120):
    _, html, _ = req(adm, "/admin/models")
    m = re.search(rb'/admin/runs/(\d+)', html)
    run_id = m.group(1).decode() if m else None
    if b">running<" not in html:
        break
    time.sleep(3)
_, log, _ = req(adm, f"/admin/runs/{run_id}")
check("admin: retrain finished successfully", b"Status: <b>success</b>" in log)
check("admin: training log shows all 12 classifiers", log.count(b"cv-f1=") == 12, log.count(b"cv-f1="))
check("admin: same data -> model promoted (score not lower)", b"PROMOTED" in log and b"NOT PROMOTED" not in log)
code, body, _ = req(adm, "/api/v1/models")
check("admin: app still serves predictions after retrain", b"diabetes" in body)

passed = sum(ok for _, ok in results)
print(f"\nTOTAL {passed}/{len(results)} live checks passed")
