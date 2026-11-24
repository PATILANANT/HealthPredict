"""Live end-to-end test (CSRF on). Start the server first:  python run.py   then:  python scripts/e2e_live.py"""
import http.cookiejar
import json
import re
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:5000"
results = []


def check(name, ok, info=""):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {info}" if info and not ok else ""))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def session(follow=True):
    jar = http.cookiejar.CookieJar()
    handlers = [urllib.request.HTTPCookieProcessor(jar)]
    if not follow:
        handlers.append(NoRedirect())
    return urllib.request.build_opener(*handlers)


def req(op, path, data=None, headers=None, method=None, raw=False):
    body = None
    if isinstance(data, dict) and not raw:
        body = urllib.parse.urlencode(data).encode()
    elif data is not None:
        body = data
    r = urllib.request.Request(BASE + path, data=body, headers=headers or {}, method=method)
    try:
        resp = op.open(r, timeout=30)
        return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def csrf(op, path):
    _, html, _ = req(op, path)
    m = re.search(rb'name="csrf_token" value="([^"]+)"', html)
    return m.group(1).decode() if m else ""


# ---------- registration / login ----------
s = session()
email = f"e2e_{secrets.token_hex(3)}@example.com"
name = "Ganesh <script>alert(1)</script>"
tok = csrf(s, "/register")
check("CSRF token present on register form", bool(tok))
code, html, _ = req(s, "/register", {"csrf_token": tok, "name": name, "email": email,
                                    "password": "ganesh12345", "confirm": "ganesh12345"})
check("register works", code == 200 and b"Hello," in html, code)
check("XSS in name is escaped", b"<script>alert(1)</script>" not in html and b"&lt;script&gt;" in html)

nocsrf = session()
req(nocsrf, "/login")
code, _, _ = req(nocsrf, "/login", {"email": email, "password": "ganesh12345"})
check("POST without CSRF token is rejected", code == 400, code)

bad = session()
tok = csrf(bad, "/login")
code, html, _ = req(bad, "/login", {"csrf_token": tok, "email": email, "password": "wrong"})
check("wrong password rejected", b"Invalid email or password" in html)

nr = session(follow=False)
tok = csrf(nr, "/login")
code, _, h = req(nr, "/login?next=//evil.com", {"csrf_token": tok, "email": email, "password": "ganesh12345"})
check("open redirect blocked (next=//evil.com)", "evil.com" not in h.get("Location", ""), h.get("Location"))

# ---------- predictions through real forms ----------
cases = {
    "diabetes": ({"Pregnancies": "2", "Glucose": "165", "BloodPressure": "80", "SkinThickness": "",
                  "Insulin": "", "BMI": "34.5", "DiabetesPedigreeFunction": "0.6", "Age": "50"}, b"Diabetic"),
    "stroke": ({"gender": "Male", "age": "72", "hypertension": "1", "heart_disease": "1",
                "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
                "avg_glucose_level": "210.5", "bmi": "31", "smoking_status": "formerly smoked"},
               b"High Stroke Risk"),
    "mental_health": ({"text": "i want to kill himm, I wont spare that guyy", "consent": "on"},
                      b"Anger / violent thoughts"),
}
pred_urls = []
for module, (form, expect) in cases.items():
    tok = csrf(s, f"/predict/{module}")
    form = dict(form, csrf_token=tok)
    code, html, _ = req(s, f"/predict/{module}", form)
    ok = code == 200 and expect in html
    check(f"web predict {module} -> {expect.decode()}", ok, code)
    m = re.search(rb'/report/pdf/(\d+)', html)
    if m:
        pred_urls.append(f"/result/{m.group(1).decode()}")
        code, pdf, h = req(s, f"/report/pdf/{m.group(1).decode()}")
        check(f"PDF for {module}", code == 200 and pdf[:4] == b"%PDF" and h.get("Content-Type") == "application/pdf")

tok = csrf(s, "/predict/mental_health")
code, html, _ = req(s, "/predict/mental_health", {"csrf_token": tok, "text": "hello there my friend, how are you doing"})
check("mental health without consent -> error", b"Please give consent" in html)
tok = csrf(s, "/predict/diabetes")
code, html, _ = req(s, "/predict/diabetes", {"csrf_token": tok, "Glucose": "-5", "BMI": "abc", "Age": "150"})
check("diabetes bad values -> 3+ field errors", html.count(b"invalid-feedback") >= 3)

code, pdf, _ = req(s, "/report/pdf")
check("combined PDF report", code == 200 and pdf[:4] == b"%PDF")
code, html, _ = req(s, "/report/")
check("combined report shows overall High", b"Overall" in html and b"High risk" in html)

# ---------- access control ----------
other = session()
tok = csrf(other, "/register")
req(other, "/register", {"csrf_token": tok, "name": "Other", "email": f"o_{secrets.token_hex(3)}@example.com",
                         "password": "other12345", "confirm": "other12345"})
code, _, _ = req(other, pred_urls[0])
check("other user cannot see Ganesh's result (403)", code == 403, code)
code, _, _ = req(other, "/report/pdf/" + pred_urls[0].split("/")[-1])
check("other user cannot download Ganesh's PDF (403)", code == 403, code)
code, _, _ = req(other, "/admin/")
check("normal user blocked from admin (403)", code == 403, code)
code, _, _ = req(other, "/patients/")
check("normal user blocked from patients (403)", code == 403, code)

doc = session()
tok = csrf(doc, "/login")
req(doc, "/login", {"csrf_token": tok, "email": "doctor@healthpredict.local", "password": "doctor12345"})
code, html, _ = req(doc, "/patients/")
check("doctor sees patients list incl. Ganesh", code == 200 and email.encode() in html)
code, _, _ = req(doc, pred_urls[0])
check("doctor can open Ganesh's result", code == 200)
code, _, _ = req(doc, "/admin/")
check("doctor blocked from admin (403)", code == 403)

adm = session()
tok = csrf(adm, "/login")
req(adm, "/login", {"csrf_token": tok, "email": "admin@healthpredict.local", "password": "admin12345"})
for p in ("/admin/", "/admin/users", "/admin/models", "/metrics/stroke", "/metrics/mental_health"):
    code, _, _ = req(adm, p)
    check(f"admin page {p}", code == 200, code)

# ---------- REST API ----------
_, html, _ = req(s, "/profile")
key = re.search(rb'font-monospace" readonly value="([0-9a-f]+)"', html).group(1).decode()
J = {"Content-Type": "application/json"}
code, body, _ = req(s, "/api/v1/predict/diabetes", json.dumps({"Glucose": 100}).encode(), J)
check("API without key -> 401", code == 401)
code, body, _ = req(s, "/api/v1/predict/diabetes", b"not json", dict(J, **{"X-API-Key": key}))
check("API bad JSON -> 400", code == 400, code)
code, body, _ = req(s, "/api/v1/predict/heart", b"{}", dict(J, **{"X-API-Key": key}))
check("API unknown module -> 404", code == 404)
code, body, _ = req(s, "/api/v1/predict/stroke", json.dumps({"age": 300}).encode(), dict(J, **{"X-API-Key": key}))
check("API invalid fields -> 422 with field list", code == 422 and "age" in json.loads(body)["fields"])
code, body, _ = req(s, "/api/v1/predict/mental_health",
                    json.dumps({"text": "I feel empty, hopeless and worthless every day", "consent": True}).encode(),
                    dict(J, **{"X-API-Key": key}))
check("API mental health predict", code == 200 and json.loads(body)["label"] == "Depression", body[:120])
code, body, _ = req(s, "/api/v1/predictions", headers={"X-API-Key": key})
check("API history lists predictions", code == 200 and len(json.loads(body)) >= 4)
code, body, _ = req(s, "/api/v1/modules")
check("API modules schema", code == 200 and set(json.loads(body)) == {"diabetes", "stroke", "mental_health"})

# ---------- privacy ----------
import sqlite3
from pathlib import Path
db = sqlite3.connect(str(Path(__file__).resolve().parents[1] / "instance" / "health.db"))
leak = db.execute("select count(*) from prediction where inputs_json like '%spare that%' "
                  "or result_json like '%I wont spare that guyy%'").fetchone()[0]
check("raw mental-health text not stored in DB", leak == 0, leak)
db.close()

passed = sum(ok for _, ok in results)
print(f"\n{passed}/{len(results)} checks passed")
if __name__ == "__main__":
    sys.exit(0 if passed == len(results) else 1)
