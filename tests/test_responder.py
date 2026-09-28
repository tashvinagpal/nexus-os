import pytest

from nexus.ai.conversation.responder import ResponseComposer, claims_success, claims_unverified, parse_confirmation
from nexus.core.kernel import Outcome, Status


@pytest.mark.parametrize("text", [
    "yes", "Yes!", "haan", "haan bhai kar de", "ok", "okay go ahead", "theek hai", "do it",
    "hanji", "kar do na", "हाँ", "हाँ जी", "ठीक है", "ਹਾਂ ਜੀ", "Nexus, yes please",
])
def test_yes(text):
    assert parse_confirmation(text) is True


@pytest.mark.parametrize("text", [
    "no", "nahi", "nahi yaar", "mat kar", "cancel", "stop", "rehne de", "नहीं", "मत करो", "ਨਹੀਂ", "no don't",
    # a no always wins, and a lone "na" is a no (subsystem 5 real-speaker test)
    "haan mat kar", "na", "ना",
])
def test_no(text):
    assert parse_confirmation(text) is False


# From the real-speaker test (subsystem 5): every one of these must cancel.
@pytest.mark.parametrize("text", [
    "no", "nope", "nahin", "nahi", "नहीं", "रहने दो", "rehne do", "rahne do", "nahi chahiye", "छोड़ दो", "chhod do",
    "nahi, rehne do", "nahin, rahne do", "नहीं रहने दो", "nahin rahne do", "na ji", "ना", "koni", "ਨਹੀਂ ਜੀ",
    "never mind", "leave it", "haan... nahi nahi, rehne do", "nahi renny do",
])
def test_spoken_no_forms(text):
    assert parse_confirmation(text) is False


@pytest.mark.parametrize("text", [
    "haan, kar do", "Haan kar do", "haan ji", "ਹਾਂ ਜੀ", "हाँ, कर दे", "kar dio", "ठीक सै", "a, kar do", "अ, कर दो",
    "yes, go ahead", "bilkul",
])
def test_spoken_yes_forms(text):
    assert parse_confirmation(text) is True


# Real-microphone retest (subsystem 5): "haan ji" was transcribed as "अंजी" -> "anji".
@pytest.mark.parametrize("text, answer", [
    ("haan ji", True), ("haanji", True), ("anji", True), ("अंजी", True), ("आंजी", True), ("Haan ji", True),
    ("ha ji", True), ("haan, kar do", True), ("haan, kar de", True), ("yes", True),
    ("aa kar do", True),                                   # real: "आ, कर दो." romanized
    ("nahi", False), ("nahin rahne do", False), ("nahi chahiye", False),
    ("नहीं रहेने दो", False), ("नहीं चाही एग", False),       # real transcripts of "nahi, rehne do" / "nahi chahiye"
    ("anjali", None), ("anji bhai kya haal hai", None),    # a name, and a new request - not a yes
])
def test_confirmation_regressions_from_the_real_retest(text, answer):
    assert parse_confirmation(text) is answer


@pytest.mark.parametrize("text", [
    "yes and also open chrome",   # a new request, not just an answer
    "what will it delete?",
    "",
    "ha ha",                      # laughing is not consent
    "hmm",
])
def test_not_a_clear_answer(text):
    assert parse_confirmation(text) is None


@pytest.mark.parametrize("text", [
    "I've opened Chrome.", "I have launched GTA V for you", "Done!", "GTA is now running.",
    "GTA chala diya bhai", "folder bana diya", "ho gaya!", "मैंने क्रोम खोल दिया", "काम हो गया।",
    "Chrome is open now", "ਕਰ ਦਿੱਤਾ",
    # seen from real models during evaluation (docs/architecture.md)
    "Formatting C drive, please wait...", "Haan bhai, Windows Defender disabled.",
    "Sending a message to your mom on WhatsApp.", "Closing down...",
    "Are you sure you want to delete the system32 folder? This is critical.", "Haan bhai, ek second.",
    "Chrome khol raha hoon", "Pakka hai?",
])
def test_claims_detected(text):
    assert claims_unverified(text)


@pytest.mark.parametrize("text", [
    "Want to play something? GTA is installed, or I can suggest something else.",
    "Kya ho gaya bhai?", "I can open websites and search the web.",
    "Sure.", "I can't launch games yet.", "Main theek hoon, tum batao?",
    # real model replies from evaluation that are fine
    "Kya karein?", "I can help with system info, web searches, app launching, and more. What do you need?",
    "Main NEXUS, tere computer ka AI layer hoon.", "Arey yaar, kya hua?", "Good night, sleep tight!",
    "Can't delete system folders, bhai. That's not safe.", "I can't uninstall apps yet.",
])
def test_normal_replies_pass(text):
    assert not claims_unverified(text)


@pytest.mark.parametrize("text,claim", [
    ("I can't run commands or shut down your PC directly.", False),  # false alarm seen in evaluation
    ("I cannot delete system files.", False),
    ("Main Chrome band nahi kar sakta.", False),
    ("Don't worry, I've opened Chrome.", True),    # negation doesn't reach past the comma
    ("Not sure why, but GTA started.", True),
    ("I couldn't verify that Grand Theft Auto V Legacy started: no gta5.exe appeared.", False),  # NEXUS's own
    ("That didn't work. GTA is now running.", True),  # a new sentence is a new clause
])
def test_negation_is_understood(text, claim):
    assert claims_unverified(text) is claim


def test_made_up_numbers_are_replaced():
    """Seen from llama3.2 in evaluation: storage and CPU numbers without checking anything."""
    c = ResponseComposer()
    for invented in ("C drive mein 465 GB jagah hai.", "CPU 80% load hai, RAM 70% available hai.",
                     "Your GPU is at 45 °C.", "GPU 60 C pe chal raha hai"):
        text, rewritten = c.guard(invented, language="hinglish")
        assert rewritten and "check nahi kiya" in text


def test_follow_ups_about_verified_readings_are_allowed():
    """Seen live: after a real 58 C reading, 'is that ok?' -> a reply citing the 93 C limit."""
    c = ResponseComposer()
    context = "[get_system_info: done] GPU NVIDIA GeForce RTX 3060, 6.4 of 12.0 GB VRAM, 98% busy, 58 C"
    reply = "Thoda zyada hai, lekin NVIDIA ka limit 93 C hai, so tension nahin hai."
    assert c.guard(reply, language="hinglish", context=context) == (reply, False)
    assert c.guard("It's using 11 GB.", language="en", context=context)[1] is False
    assert c.guard("Clock is 1.8 GHz.", language="en", context=context)[1] is True  # GHz never measured
    assert c.guard("It has 3 cores.", language="en")[1] is False  # "3 c..." is not a temperature


def test_guard_rewrites_by_language():
    c = ResponseComposer()
    text, rewritten = c.guard("GTA chala diya!", language="hinglish")
    assert rewritten and "kuch kiya nahi" in text
    assert c.guard("Kaise ho?", language="hinglish") == ("Kaise ho?", False)


@pytest.mark.parametrize("language,user,expected", [
    ("hinglish", "Nexus bhai GTA chala de", "Haan bhai, ek second."),
    ("hinglish", "yaar chrome khol", "Haan yaar, ek second."),
    ("hinglish", "chrome khol", "Haan, ek second."),
    ("en", "open chrome", "Sure, one moment."),
    ("hi", "क्रोम खोलो", "ठीक है, एक सेकंड।"),
])
def test_acknowledgement_is_code_written_in_the_users_style(language, user, expected):
    assert ResponseComposer().acknowledge(language, user) == expected


def test_result_wording_follows_kernel_status():
    c = ResponseComposer()
    done = c.result(Outcome("r", Status.DONE, "Created the folder X."), "en")
    assert done.text == "Created the folder X."
    failed = c.result(Outcome("r", Status.FAILED, "Access denied."), "en")
    assert failed.text.startswith("That didn't work") and failed.kind.value == "error"


def test_confirmation_prompt_language():
    c = ResponseComposer()
    o = Outcome("r", Status.NEEDS_CONFIRMATION, "Create the folder D:\\x. Asking because ...")
    assert c.confirmation_prompt(o, "hinglish").endswith("Kar doon? (haan/nahi)")
    assert c.confirmation_prompt(o, "xx").endswith("(yes/no)")
