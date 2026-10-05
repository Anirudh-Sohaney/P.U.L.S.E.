"""Exercise the built browser forecast without uploading private CSV contents.

Requires Playwright and its Chromium browser. Run against an isolated test
deployment because this creates a disposable account in that deployment.
"""

from __future__ import annotations

import argparse
import csv
import json
import secrets
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import expect, sync_playwright


def fixture_files(directory: Path, marker: str) -> tuple[Path, Path]:
    last_day = datetime.now(timezone.utc).date()
    first_day = last_day - timedelta(days=199)
    sales = directory / "sales.csv"
    inventory = directory / "inventory.csv"
    with sales.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("date", "drug_name", "units_sold"))
        for index in range(200):
            day = first_day + timedelta(days=index)
            writer.writerow((day.isoformat(), marker, 4 + day.weekday() % 4))
    with inventory.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("date", "drug_name", "on_hand_units", "on_order_units"))
        for index in range(200):
            day = first_day + timedelta(days=index)
            writer.writerow((day.isoformat(), marker, 2 + index % 11, 0))
    return sales, inventory


def run(base_url: str, signal_years: int = 0,
        simulate_rate_limit: bool = False) -> None:
    username = "smoke_" + secrets.token_hex(5)
    password = "Browser-smoke-" + secrets.token_urlsafe(18)
    marker = "PrivateDrug" + secrets.token_hex(7)
    api_bodies: list[str] = []
    signal_history_statuses: list[int] = []
    history_queries: list[dict] = []
    injected_limit = [False]
    with tempfile.TemporaryDirectory(prefix="pulse-browser-smoke-") as directory:
        sales, inventory = fixture_files(Path(directory), marker)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
            context = browser.new_context(timezone_id="UTC")
            page = context.new_page()

            def record_api_request(request):
                if "/api/" in request.url:
                    api_bodies.append(request.post_data or "")
                if "/api/v1/signals/history" in request.url and request.post_data:
                    history_queries.append(json.loads(request.post_data))

            page.on("request", record_api_request)
            page.on("response", lambda response: signal_history_statuses.append(response.status)
                    if "/api/v1/signals/history" in response.url else None)
            if simulate_rate_limit:
                def throttle_once(route):
                    if not injected_limit[0]:
                        injected_limit[0] = True
                        route.fulfill(status=429, content_type="application/json",
                                      headers={"Retry-After": "1"},
                                      body='{"detail":"Public API request limit reached"}')
                    else:
                        route.continue_()
                page.route("**/api/v1/signals/history", throttle_once)
            try:
                docs = context.new_page()
                external_docs_requests: list[str] = []
                def same_origin_docs(route):
                    if route.request.url.startswith(base_url.rstrip("/") + "/"):
                        route.continue_()
                    else:
                        external_docs_requests.append(route.request.url)
                        route.abort()
                docs.route("**/*", same_origin_docs)
                docs.goto(base_url.rstrip("/") + "/docs", wait_until="domcontentloaded")
                expect(docs.locator(".swagger-ui .info .title")).to_contain_text(
                    "Pharmacy Demand Planner", timeout=30000)
                assert not external_docs_requests, external_docs_requests
                docs.close()

                page.goto(base_url.rstrip("/") + "/", wait_until="domcontentloaded")
                page.get_by_role("button", name="Sign up", exact=True).click()
                page.get_by_label("Username").fill(username)
                page.get_by_label("Password").fill(password)
                page.get_by_role("button", name="Create my workspace").click()
                expect(page.get_by_role("heading", name="Build your demand plan.")).to_be_visible(timeout=30000)
                files = page.locator('input[type="file"]')
                files.nth(0).set_input_files(str(sales))
                files.nth(1).set_input_files(str(inventory))
                page.locator("#signal-years").select_option(str(signal_years))
                page.get_by_role("button", name="Train on this device").click()
                expect(page.get_by_role("heading", name="Restock priorities")).to_be_visible(timeout=240000)
                expect(page.get_by_role("cell", name=marker, exact=True)).to_be_visible()
                if signal_years:
                    assert signal_history_statuses, "Training did not request public signal history"
                    catalog_response = context.request.get(
                        base_url.rstrip("/") + "/api/v1/signals/catalog?limit=1500")
                    assert catalog_response.status == 200
                    catalog = {row["id"]: row for row in catalog_response.json()["signals"]}
                    for query in history_queries:
                        for uid in query["ids"]:
                            row = catalog[uid]
                            assert row["latest_observation_date"] >= query["start_date"]
                            assert row["signal_origin"] != "model_news_output"
                            assert not row["signal_id"].startswith(
                                "arkansas_atc_demand_state::")
                    if simulate_rate_limit:
                        assert injected_limit[0] and signal_history_statuses.count(429) == 1
                    assert all(status in (200, 429) for status in signal_history_statuses), signal_history_statuses
                    assert signal_history_statuses[-1] == 200

                encrypted = page.evaluate("""async username => {
                  const db = await new Promise((resolve, reject) => {
                    const request = indexedDB.open('pulse-private-v1', 1)
                    request.onsuccess = () => resolve(request.result)
                    request.onerror = () => reject(request.error)
                  })
                  const record = await new Promise((resolve, reject) => {
                    const tx = db.transaction('plans', 'readonly')
                    const request = tx.objectStore('plans').get(username)
                    request.onsuccess = () => resolve(request.result)
                    request.onerror = () => reject(request.error)
                  })
                  db.close()
                  return record
                }""", username)
                assert encrypted and encrypted.get("iv") and encrypted.get("ciphertext")
                assert marker not in str(encrypted), "Saved plan contains plaintext private data"
                assert all(marker not in body for body in api_bodies), "Private CSV content reached the API"

                page.get_by_role("button", name="Train with new files").click()
                expect(page.get_by_text("Choose CSV file", exact=True)).to_have_count(2)
                assert page.locator('input[type="file"]').evaluate_all(
                    "inputs => inputs.every(input => input.files.length === 0)"
                ), "Private CSV files stayed selected after training"

                page.clock.install()
                page.reload(wait_until="domcontentloaded")
                expect(page.get_by_role("heading", name="Unlock your workspace.")).to_be_visible(timeout=30000)
                page.get_by_label("Password").fill(password)
                page.get_by_role("button", name="Unlock plan").click()
                expect(page.get_by_role("heading", name="Restock priorities")).to_be_visible(timeout=30000)
                expect(page.get_by_role("cell", name=marker, exact=True)).to_be_visible()
                assert all(marker not in body for body in api_bodies), "Private CSV content reached the API"

                page.get_by_role("button", name="Train with new files").click()
                files = page.locator('input[type="file"]')
                files.nth(0).set_input_files(str(sales))
                files.nth(1).set_input_files(str(inventory))
                page.locator("#signal-years").select_option("1")
                held_routes = []
                page.route("**/api/v1/signals/catalog*", lambda route: held_routes.append(route))
                with page.expect_request("**/api/v1/signals/catalog*", timeout=60000):
                    page.get_by_role("button", name="Train on this device").click()
                page.clock.fast_forward(16 * 60 * 1000)
                expect(page.get_by_role("heading", name="Unlock your workspace.")).to_be_visible(timeout=10000)
                for route in held_routes:
                    route.abort()

                page.get_by_label("Password").fill(password)
                page.get_by_role("button", name="Unlock plan").click()
                expect(page.get_by_role("heading", name="Restock priorities")).to_be_visible(timeout=30000)
                held_auth = []
                page.route("**/api/auth/me", lambda route: held_auth.append(route))
                with page.expect_request("**/api/auth/me", timeout=10000):
                    page.evaluate("window.dispatchEvent(new Event('focus'))")
                page.clock.fast_forward(11_000)
                expect(page.get_by_role("heading", name="Unlock your workspace.")).to_be_visible(timeout=10000)
                for route in held_auth:
                    try:
                        route.abort()
                    except Exception:
                        pass  # The browser may already have canceled the timed-out request.

                security_context = browser.new_context()
                try:
                    signed_in = security_context.request.post(
                        base_url.rstrip("/") + "/api/auth/session",
                        form={"username": username, "password": password})
                    assert signed_in.status == 200
                    security_page = security_context.new_page()
                    security_page.goto(base_url.rstrip("/") + "/dashboard",
                                       wait_until="domcontentloaded")
                    security_page.get_by_role("button", name="Sign out everywhere").wait_for()
                    security_page.get_by_role("button", name="Sign out everywhere").evaluate(
                        "button => button.click()")
                    expect(security_page).to_have_url(base_url.rstrip("/") + "/", timeout=10000)
                    assert security_context.request.get(
                        base_url.rstrip("/") + "/api/auth/me").status == 401
                finally:
                    security_context.close()
            except PlaywrightTimeoutError as exc:
                raise AssertionError(f"Browser smoke timed out: {page.url}; {page.locator('body').inner_text()[:1000]}") from exc
            finally:
                context.close()
                browser.close()
    print(f"Browser smoke passed ({signal_years} signal years): local training, "
          "encrypted reload, idle lock, all-session revocation, and no private CSV API upload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:3141")
    parser.add_argument("--signal-years", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--simulate-rate-limit", action="store_true")
    args = parser.parse_args()
    run(args.base_url, args.signal_years, args.simulate_rate_limit)
