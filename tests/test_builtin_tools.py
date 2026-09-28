from nexus.core.kernel import Status, ToolRequest
from nexus.system import info


def run(kernel, tool, **args):
    return kernel.submit(ToolRequest(tool, args))


def test_system_info_reads_real_hardware(make_kernel):
    out = run(make_kernel(), "get_system_info")
    assert out.status is Status.DONE and out.result.verified is True
    data = out.result.data
    assert data["memory"]["total_gb"] > 0
    assert data["cpu"]["threads"] >= 1
    assert data["storage"], "expected at least one drive"
    assert "Memory" in out.message


def test_system_info_single_section(make_kernel):
    out = run(make_kernel(), "get_system_info", section="gpu")
    assert set(out.result.data) == {"gpu"}


def test_gpu_report_handles_missing_values():
    text = info.format_report({"gpu": [{"name": "X", "vram_total_gb": None, "vram_used_gb": None,
                                        "usage_percent": None, "temperature_c": None}]})
    assert text == "GPU      X"
    assert "no NVIDIA GPU" in info.format_report({"gpu": []})


def test_open_url_hands_off_to_platform(make_kernel, fake_platform):
    out = run(make_kernel(), "open_url", url="https://github.com")
    assert out.status is Status.DONE
    assert out.result.verified is None  # honest: can't confirm a tab appeared
    assert fake_platform.opened == ["https://github.com"]


def test_open_url_refuses_local_files(make_kernel, fake_platform):
    out = run(make_kernel(), "open_url", url="file:///C:/Windows/System32/calc.exe")
    assert out.status is Status.INVALID and fake_platform.opened == []


def test_open_url_failure_reported(make_kernel, fake_platform):
    def broken(uri):
        raise OSError("no browser")
    fake_platform.open_uri = broken
    out = run(make_kernel(), "open_url", url="https://github.com")
    assert out.status is Status.FAILED and "no browser" in out.message


def test_search_web_encodes_query(make_kernel, fake_platform):
    run(make_kernel(), "search_web", query="gta v mods & scripts", engine="youtube")
    assert fake_platform.opened == [
        "https://www.youtube.com/results?search_query=gta+v+mods+%26+scripts"]


def test_create_folder_in_safe_root(make_kernel, sandbox):
    target = sandbox / "home" / "projects" / "nexus"
    out = run(make_kernel(), "create_folder", path=str(target))
    assert out.status is Status.DONE and out.result.verified is True
    assert target.is_dir()
    again = run(make_kernel(), "create_folder", path=str(target))
    assert again.status is Status.DONE and "already exists" in again.message


def test_create_folder_over_file_fails(make_kernel, sandbox):
    f = sandbox / "home" / "file.txt"
    f.write_text("x")
    assert run(make_kernel(), "create_folder", path=str(f)).status is Status.FAILED


def test_create_folder_outside_safe_root_asks_then_creates(make_kernel, sandbox):
    k = make_kernel()
    target = sandbox / "outside" / "games"
    out = run(k, "create_folder", path=str(target))
    assert out.status is Status.NEEDS_CONFIRMATION and not target.exists()
    assert "Create the folder" in out.message
    assert k.resolve(out.confirmation_id, approved=True).status is Status.DONE
    assert target.is_dir()


def test_create_folder_in_protected_refused(make_kernel, sandbox):
    out = run(make_kernel(), "create_folder", path=str(sandbox / "system" / "x"))
    assert out.status is Status.BLOCKED and not (sandbox / "system" / "x").exists()
