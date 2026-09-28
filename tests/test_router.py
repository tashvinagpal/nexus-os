import pytest

from nexus.ai.router import FastPathRouter, detect_language

router = FastPathRouter()


def step(text):
    plan = router.route(text)
    return None if plan is None else (plan.actions[0].tool, plan.actions[0].args)


@pytest.mark.parametrize("text,section", [
    ("GPU ka temperature kya hai?", "gpu"), ("what's my GPU temperature", "gpu"), ("What GPU do I have?", "gpu"),
    ("how much RAM am I using", "memory"), ("RAM kitni use ho rahi hai", "memory"),
    ("kitni storage bachi hai?", "storage"), ("C drive mein kitni jagah hai", "storage"),
    ("CPU usage batao", "cpu"), ("system ki halat kaisi hai", "all"), ("PC ka haal batao", "all"),
    ("gpu and cpu temperature", None),  # "and": a compound request goes to the planner
])
def test_hardware_readings(text, section):
    result = step(text)
    if section is None:
        assert result is None
    else:
        assert result == ("get_system_info", {"section": section})


@pytest.mark.parametrize("text", ["what time is it?", "What's the date today", "time kya hua hai", "kitne baje hain",
                                  "aaj kaunsi date hai", "कितने बजे हैं", "abhi time kya hai"])
def test_time_and_date(text):
    assert step(text) == ("get_datetime", {})


@pytest.mark.parametrize("text", ["what do you remember about me?", "Tumhe mere baare mein kya yaad hai?",
                                  "show my memories", "what do you know about me"])
def test_list_memories(text):
    assert step(text) == ("list_memories", {})


@pytest.mark.parametrize("text,fact,scope", [
    ("Remember that Chrome is my preferred browser.", "Chrome is my preferred browser", "device"),
    ("remember my favourite game is GTA V", "my favourite game is GTA V", "device"),
    ("yaad rakhna ki mera favourite game GTA V hai", "mera favourite game GTA V hai", "device"),
    ("mera favourite game GTA V hai, yaad rakhna", "mera favourite game GTA V hai", "device"),
    ("remember that for today I'm working from home", "I'm working from home", "temporary"),
    ("remember privately that my bank is ABC", "my bank is ABC", "private"),
])
def test_remember_extraction(text, fact, scope):
    assert step(text) == ("remember", {"text": fact, "scope": scope})


@pytest.mark.parametrize("text", [
    "open chrome", "Nexus bhai GTA chala de", "mera PC slow kyun chal raha hai?", "what is RAM?",
    "RAM kya hota hai", "which GPU should I buy", "search gpu prices", "update my gpu driver",
    "yaar I'm bored", "ke haal se bhai?", "tussi ki haal chaal?", "remember to buy milk",
    "do you remember my birthday?", "forget that", "CPU usage batao aur chrome khol",
    "kaise ho?", "difference between RAM and VRAM", "format my C drive",
])
def test_everything_else_goes_to_the_planner(text):
    assert router.route(text) is None


def test_fast_plans_skip_the_acknowledgement():
    plan = router.route("what time is it?")
    assert plan.acknowledge is False and plan.say == ""


@pytest.mark.parametrize("text,lang", [("GPU ka temperature kya hai?", "hinglish"), ("what time is it", "en"),
                                       ("कितने बजे हैं", "hi"), ("ਕਿੰਨੇ ਵਜੇ ਹਨ", "pa")])
def test_language_detection(text, lang):
    assert detect_language(text) == lang
