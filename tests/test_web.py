"""Headless browser tests for InferX web UI using Playwright + asyncio."""

import asyncio
import pytest
from playwright.async_api import async_playwright

BASE = "http://localhost:8999"


async def _run_test(test_func, *args, **kwargs):
    """Helper to run an async test with a fresh browser page."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1400, "height": 900})
        try:
            await test_func(page, *args, **kwargs)
        finally:
            await browser.close()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

async def _test_sidebar_pages_exist(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    nav_items = await page.query_selector_all(".nav-item")
    assert len(nav_items) >= 7

async def _test_click_each_page(page):
    for pg in ["dashboard", "models", "download", "instances", "presets", "benchmark", "config"]:
        await page.goto(BASE, wait_until="networkidle", timeout=10000)
        await page.click(f'[data-page="{pg}"]')
        assert await page.is_visible(f"#page-{pg}"), f"Failed to show page: {pg}"


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

async def _test_dashboard_gpu(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    text = await page.text_content("#cv-gpumem")
    assert "GB" in text or "MB" in text or "N/A" in text

async def _test_dashboard_cpu(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    text = await page.text_content("#cv-cpu")
    assert "cores" in text.lower() or "CORES" in text

async def _test_dashboard_ram(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    text = await page.text_content("#cv-ram")
    assert "GB" in text or "MB" in text

async def _test_dashboard_instances_count(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    total = await page.text_content("#cv-inst-total")
    assert total.strip().isdigit()

async def _test_engine_status_card(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    heading = await page.query_selector("text=Engine Status")
    assert heading is not None

async def _test_engine_status_installed(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    installed = await page.query_selector_all("text=Installed")
    assert len(installed) >= 3

async def _test_engine_status_not_installed(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    not_installed = await page.query_selector_all("text=Not installed")
    assert len(not_installed) >= 1

async def _test_no_running_models(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    msg = await page.query_selector("text=No running models")
    assert msg is not None


# ---------------------------------------------------------------------------
# Models page
# ---------------------------------------------------------------------------

async def _test_models_page(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="models"]')
    await page.wait_for_timeout(800)
    heading = await page.evaluate("() => document.querySelector('#page-models')?.innerText?.substring(0, 100) || ''")
    assert "Models" in heading
    table = await page.query_selector("#page-models table")
    assert table is not None


# ---------------------------------------------------------------------------
# Instances page
# ---------------------------------------------------------------------------

async def _test_instances_page(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="instances"]')
    await page.wait_for_timeout(800)
    heading = await page.evaluate("() => document.querySelector('#page-instances')?.innerText?.substring(0, 100) || ''")
    assert "Running Instances" in heading
    btn = await page.evaluate("() => !!document.querySelector('#page-instances button')")
    assert btn


# ---------------------------------------------------------------------------
# Start Instance Modal
# ---------------------------------------------------------------------------

async def _test_modal_opens_and_closes(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="instances"]')
    await page.wait_for_timeout(500)
    await page.click("text=+ Start New")
    await page.wait_for_selector("#modal-start.show", timeout=5000)
    assert await page.is_visible("#modal-start.show")
    await page.click("text=Cancel")
    await page.wait_for_timeout(300)
    assert not await page.is_visible("#modal-start.show")

async def _test_backend_dropdown_all_options(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="instances"]')
    await page.wait_for_timeout(500)
    await page.click("text=+ Start New")
    await page.wait_for_selector("#modal-start.show", timeout=5000)
    options = await page.evaluate("""
        () => Array.from(document.getElementById('start-backend').options).map(o => o.value)
    """)
    assert len(options) == 8
    assert "llamacpp" in options
    assert "vllm" in options

async def _test_unavailable_backends_disabled(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    # Wait for cachedBackends to be populated (API call is slow)
    for _ in range(20):
        cached = await page.evaluate("() => window.cachedBackends")
        if cached is not None:
            break
        await page.wait_for_timeout(500)
    await page.click('[data-page="instances"]')
    await page.wait_for_timeout(500)
    await page.click("text=+ Start New")
    await page.wait_for_selector("#modal-start.show", timeout=5000)
    await page.wait_for_timeout(300)
    disabled = await page.evaluate("""
        () => Array.from(document.getElementById('start-backend').options)
            .filter(o => o.disabled).map(o => o.value)
    """)
    enabled = await page.evaluate("""
        () => Array.from(document.getElementById('start-backend').options)
            .filter(o => !o.disabled).map(o => o.value)
    """)
    # At least some backends should be disabled and some enabled
    assert len(disabled) >= 1, f"Expected some disabled backends, got: {disabled}"
    assert len(enabled) >= 1, f"Expected some enabled backends, got: {enabled}"
    # Verify the disabled ones have "(Not installed)" text
    for d in disabled:
        opt_text = await page.evaluate(f"""
            () => document.querySelector('#start-backend option[value="{d}"]')?.textContent || ''
        """)
        assert "Not installed" in opt_text, f"Disabled option {d} missing '(Not installed)': {opt_text}"

async def _test_model_dropdown_populated(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="instances"]')
    await page.wait_for_timeout(500)
    await page.click("text=+ Start New")
    await page.wait_for_selector("#modal-start.show", timeout=5000)
    count = await page.evaluate("() => document.getElementById('start-model').options.length")
    assert count >= 1

async def _test_backend_params_switch(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="instances"]')
    await page.wait_for_timeout(500)
    await page.click("text=+ Start New")
    await page.wait_for_selector("#modal-start.show", timeout=5000)
    assert await page.is_visible("#params-llamacpp")
    assert not await page.is_visible("#params-vllm")
    await page.select_option("#start-backend", "vllm")
    await page.wait_for_timeout(200)
    assert not await page.is_visible("#params-llamacpp")
    assert await page.is_visible("#params-vllm")

async def _test_all_backend_params_sections(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="instances"]')
    await page.wait_for_timeout(500)
    await page.click("text=+ Start New")
    await page.wait_for_selector("#modal-start.show", timeout=5000)
    for backend in ["llamacpp", "vllm", "sglang", "tgi", "ollama", "tensorrt_llm", "lmdeploy", "openvino"]:
        # Skip disabled backends
        is_disabled = await page.evaluate(f"""
            () => document.querySelector('#start-backend option[value="{backend}"]')?.disabled ?? true
        """)
        if is_disabled:
            continue
        await page.select_option("#start-backend", backend)
        await page.wait_for_timeout(100)
        assert await page.is_visible(f"#params-{backend}"), f"Params section for {backend} not visible"


# ---------------------------------------------------------------------------
# Config page
# ---------------------------------------------------------------------------

async def _test_config_page(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="config"]')
    await page.wait_for_timeout(800)
    heading = await page.evaluate("() => document.querySelector('#page-config')?.innerText?.substring(0, 100) || ''")
    assert "Configuration" in heading

async def _test_config_model_dir(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="config"]')
    await page.wait_for_timeout(500)
    val = await page.input_value("#cfg-model-dir")
    assert val

async def _test_config_port_range(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="config"]')
    await page.wait_for_timeout(500)
    start = await page.input_value("#cfg-port-start")
    end = await page.input_value("#cfg-port-end")
    assert start.isdigit()
    assert end.isdigit()

async def _test_config_engine_status_badges(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="config"]')
    await page.wait_for_timeout(500)
    installed = await page.query_selector_all("text=Installed")
    not_installed = await page.query_selector_all("text=Not installed")
    assert len(installed) >= 3
    assert len(not_installed) >= 1

async def _test_config_all_engine_binaries(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="config"]')
    await page.wait_for_timeout(500)
    for field_id in ["cfg-llama-bin", "cfg-vllm-bin", "cfg-sglang-bin",
                     "cfg-tgi-bin", "cfg-ollama-bin", "cfg-trt-bin",
                     "cfg-lmdeploy-bin", "cfg-openvino-bin"]:
        el = await page.query_selector(f"#{field_id}")
        assert el is not None, f"Missing: {field_id}"

async def _test_config_default_backend_options(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="config"]')
    await page.wait_for_timeout(500)
    options = await page.evaluate("""
        () => Array.from(document.getElementById('cfg-default-backend').options).map(o => o.value)
    """)
    assert len(options) == 8

async def _test_config_save_button(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="config"]')
    await page.wait_for_timeout(500)
    btn = await page.query_selector("text=Save Config")
    assert btn is not None


# ---------------------------------------------------------------------------
# Presets page
# ---------------------------------------------------------------------------

async def _test_presets_page(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="presets"]')
    await page.wait_for_timeout(800)
    heading = await page.evaluate("() => document.querySelector('#page-presets')?.innerText?.substring(0, 100) || ''")
    assert "Presets" in heading


# ---------------------------------------------------------------------------
# Benchmark page
# ---------------------------------------------------------------------------

async def _test_benchmark_page(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="benchmark"]')
    await page.wait_for_timeout(1000)
    heading = await page.evaluate("() => document.querySelector('#page-benchmark')?.innerText?.substring(0, 100) || ''")
    assert "Benchmark" in heading
    model_table = await page.query_selector("#bench-models-tbody")
    assert model_table is not None
    backend_table = await page.query_selector("#bench-backends-tbody")
    assert backend_table is not None


# ---------------------------------------------------------------------------
# Download page
# ---------------------------------------------------------------------------

async def _test_download_page(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="download"]')
    await page.wait_for_timeout(800)
    heading = await page.evaluate("() => document.querySelector('#page-download')?.innerText?.substring(0, 100) || ''")
    assert "Download" in heading
    options = await page.evaluate("""
        () => Array.from(document.getElementById('dl-page-source')?.options || []).map(o => o.value)
    """)
    assert "hf" in options


# ---------------------------------------------------------------------------
# Download Modal
# ---------------------------------------------------------------------------

async def _test_download_modal(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="models"]')
    await page.wait_for_timeout(500)
    await page.click("text=+ Download")
    await page.wait_for_selector("#modal-download.show", timeout=5000)
    assert await page.is_visible("#modal-download.show")
    await page.click("#modal-download .btn-ghost")
    await page.wait_for_timeout(300)
    assert not await page.is_visible("#modal-download.show")


# ---------------------------------------------------------------------------
# API Docs link
# ---------------------------------------------------------------------------

async def _test_docs_link(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    link = await page.query_selector('a[href="/docs"]')
    assert link is not None


# ---------------------------------------------------------------------------
# Auto refresh
# ---------------------------------------------------------------------------

async def _test_dashboard_auto_refresh(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    val1 = await page.text_content("#cv-inst-total")
    await page.wait_for_timeout(3000)
    val2 = await page.text_content("#cv-inst-total")
    assert val1.strip().isdigit()
    assert val2.strip().isdigit()


# ---------------------------------------------------------------------------
# Screenshots
# ---------------------------------------------------------------------------

async def _test_screenshot_dashboard(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.screenshot(path="/home/ec2-user/infer-x/tests/screenshot_dashboard.png")

async def _test_screenshot_instances(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="instances"]')
    await page.wait_for_timeout(500)
    await page.screenshot(path="/home/ec2-user/infer-x/tests/screenshot_instances.png")

async def _test_screenshot_config(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="config"]')
    await page.wait_for_timeout(500)
    await page.screenshot(path="/home/ec2-user/infer-x/tests/screenshot_config.png")

async def _test_screenshot_start_modal(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="instances"]')
    await page.wait_for_timeout(500)
    await page.click("text=+ Start New")
    await page.wait_for_selector("#modal-start.show", timeout=5000)
    await page.wait_for_timeout(300)
    await page.screenshot(path="/home/ec2-user/infer-x/tests/screenshot_start_modal.png")

async def _test_screenshot_benchmark(page):
    await page.goto(BASE, wait_until="networkidle", timeout=10000)
    await page.click('[data-page="benchmark"]')
    await page.wait_for_timeout(1000)
    await page.screenshot(path="/home/ec2-user/infer-x/tests/screenshot_benchmark.png")


# ---------------------------------------------------------------------------
# Test class (sync wrappers calling async helpers)
# ---------------------------------------------------------------------------

ALL_WEB_TESTS = [
    # Navigation
    ("nav_sidebar_pages", _test_sidebar_pages_exist),
    ("nav_click_all_pages", _test_click_each_page),
    # Dashboard
    ("dashboard_gpu", _test_dashboard_gpu),
    ("dashboard_cpu", _test_dashboard_cpu),
    ("dashboard_ram", _test_dashboard_ram),
    ("dashboard_instances_count", _test_dashboard_instances_count),
    ("dashboard_engine_status_card", _test_engine_status_card),
    ("dashboard_engine_installed", _test_engine_status_installed),
    ("dashboard_engine_not_installed", _test_engine_status_not_installed),
    ("dashboard_no_running_models", _test_no_running_models),
    # Models
    ("models_page", _test_models_page),
    # Instances
    ("instances_page", _test_instances_page),
    # Start Modal
    ("modal_opens_closes", _test_modal_opens_and_closes),
    ("modal_backend_all_options", _test_backend_dropdown_all_options),
    ("modal_unavailable_disabled", _test_unavailable_backends_disabled),
    ("modal_model_populated", _test_model_dropdown_populated),
    ("modal_params_switch", _test_backend_params_switch),
    ("modal_all_params_sections", _test_all_backend_params_sections),
    # Config
    ("config_page", _test_config_page),
    ("config_model_dir", _test_config_model_dir),
    ("config_port_range", _test_config_port_range),
    ("config_engine_badges", _test_config_engine_status_badges),
    ("config_all_binaries", _test_config_all_engine_binaries),
    ("config_default_backend", _test_config_default_backend_options),
    ("config_save_button", _test_config_save_button),
    # Presets
    ("presets_page", _test_presets_page),
    # Benchmark
    ("benchmark_page", _test_benchmark_page),
    # Download
    ("download_page", _test_download_page),
    ("download_modal", _test_download_modal),
    # API Docs
    ("api_docs_link", _test_docs_link),
    # Auto refresh
    ("auto_refresh", _test_dashboard_auto_refresh),
    # Screenshots
    ("screenshot_dashboard", _test_screenshot_dashboard),
    ("screenshot_instances", _test_screenshot_instances),
    ("screenshot_config", _test_screenshot_config),
    ("screenshot_start_modal", _test_screenshot_start_modal),
    ("screenshot_benchmark", _test_screenshot_benchmark),
]


class TestWebUI:
    pass


for _name, _func in ALL_WEB_TESTS:
    def _make_wrapper(fn):
        def wrapper(self):
            asyncio.run(_run_test(fn))
        return wrapper
    setattr(TestWebUI, f"test_{_name}", _make_wrapper(_func))
