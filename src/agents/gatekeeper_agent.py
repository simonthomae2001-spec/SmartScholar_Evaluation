from __future__ import annotations

import base64
import binascii
import html
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import unquote


@dataclass(frozen=True)
class Finding:
    category: str
    reason: str
    severity: int  # 1 = low, 2 = medium, 3 = high


class GatekeeperAgent:
    """
    Defensive first-line validation for SmartScholar.

    Important:
    - This reduces risk, but cannot guarantee full prompt-injection prevention.
    - Treat accepted user input as untrusted data in all downstream prompts/tools.
    """

    MIN_INPUT_CHARS = 3
    MAX_INPUT_CHARS = 2_000
    MAX_LINES = 35
    MAX_REPEAT_CHAR_RUN = 80
    MAX_BASE64_DECODED_CHARS = 1_500

    _ZERO_WIDTH = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]")
    _CONTROL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
    _WHITESPACE = re.compile(r"\s+")
    _LONG_B64 = re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b|\b[A-Za-z0-9_-]{40,}={0,2}\b")

    _RESEARCH_HINTS = re.compile(
        r"\b("
        r"paper|papers|literature|literatur|review|survey|study|studies|studie|studien|"
        r"research|forschung|wissenschaft|wissenschaftlich|academic|akademisch|"
        r"analyse|analysis|vergleich|compare|method|methoden|benchmark|evaluation|"
        r"rag|retrieval|semantic scholar|citation|citations|doi|publication|publikation|"
        r"systematic|systematisch|state of the art|stand der forschung"
        r")\b",
        re.IGNORECASE,
    )

    _ATTACK_ACTIONS = re.compile(
        r"\b("
        r"write|generate|create|build|execute|run|install|deploy|exploit|steal|"
        r"exfiltrate|bypass|hack|jailbreak|leak|dump|delete|drop|payload|backdoor|"
        r"schreibe|erstelle|erzeuge|führe|ausführen|umgehen|hacke|klaue|stehle|"
        r"lösche|extrahiere|exfiltriere|zeige|gib|enthülle"
        r")\b",
        re.IGNORECASE,
    )

    _SECRET_WORDS = (
        r"api[_ -]?key|token|secret|password|passwort|credential|credentials|"
        r"\.env|private key|ssh key|secrets?"
    )
    _SECRET_VERBS = (
        r"show|print|dump|read|steal|exfiltrate|send|retrieve|"
        r"zeige|lies|liest|ausliest|auslesen|stehle|sende|sendet|senden|"
        r"extrahiere|exfiltriere"
    )
    _ENDPOINT_WORDS = r"https?://|webhook|discord|telegram|pastebin|requestbin|ngrok"
    _EXFIL_VERBS = (
        r"send|sends|sendet|senden|post|upload|exfiltrate|curl|wget|sende|"
        r"lade\s+hoch|exfiltriere"
    )

    _DENY_RULES: tuple[tuple[str, int, str, re.Pattern[str]], ...] = (
        (
            "prompt_override",
            3,
            "Versuch, System-/Entwickleranweisungen zu überschreiben oder Sicherheitsregeln zu umgehen.",
            re.compile(
                r"(?is)\b(ignore|disregard|forget|override|bypass|disable|break|violate|"
                r"ignoriere|missachte|vergiss|überschreibe|umgehe|deaktiviere)\b"
                r".{0,120}\b(previous|above|earlier|system|developer|instruction|instructions|"
                r"prompt|policy|rules|guardrails|safety|security|"
                r"vorherigen|obigen|system|entwickler|anweisungen|regeln|richtlinien|sicherheit)\b"
            ),
        ),
        (
            "system_prompt_leakage",
            3,
            "Versuch, interne Prompts, Regeln oder versteckte Anweisungen offenzulegen.",
            re.compile(
                r"(?is)\b(reveal|show|print|dump|leak|exfiltrate|zeige|gib\s+.*aus|"
                r"enthülle|verrate|drucke)\b"
                r".{0,120}\b(system prompt|developer message|hidden instructions|internal rules|"
                r"secret rules|policy|systemanweisung|system-prompt|interne regeln|geheime regeln)\b"
            ),
        ),
        (
            "jailbreak_roleplay",
            3,
            "Jailbreak-/Roleplay-Muster erkannt.",
            re.compile(
                r"(?is)\b(jailbreak|DAN|developer mode|god mode|sudo mode|admin mode|"
                r"unrestricted mode|no restrictions|without restrictions|"
                r"du bist jetzt|you are now|act as an unrestricted|tu so als ob)\b"
            ),
        ),
        (
            "credential_exfiltration",
            3,
            "Versuch, Secrets, Tokens, Passwörter oder private Schlüssel auszulesen.",
            re.compile(
                rf"(?is)(?:\b({_SECRET_VERBS})\b.{{0,140}}\b({_SECRET_WORDS})\b|"
                rf"\b({_SECRET_WORDS})\b.{{0,140}}\b({_SECRET_VERBS})\b)"
            ),
        ),
        (
            "external_exfiltration",
            3,
            "Verdacht auf Datenabfluss an externe Endpunkte.",
            re.compile(
                rf"(?is)(?:\b({_EXFIL_VERBS})\b.{{0,180}}\b({_ENDPOINT_WORDS})\b|"
                rf"\b({_ENDPOINT_WORDS})\b.{{0,180}}\b({_EXFIL_VERBS})\b)"
            ),
        ),
        (
            "dangerous_shell",
            3,
            "Gefährliche Shell-/Systembefehle erkannt.",
            re.compile(
                r"(?is)(?:^|\s)("
                r"rm\s+-rf|mkfs\.|chmod\s+777|chown\s+root|sudo\s+|"
                r"powershell\s+-enc|invoke-expression|\biex\s*\(|"
                r"bash\s+-i|nc\s+-e|netcat\s+-e|/etc/passwd|/etc/shadow|"
                r"curl\s+[^|]{0,100}\|\s*(?:sh|bash)|wget\s+[^|]{0,100}\|\s*(?:sh|bash)"
                r")"
            ),
        ),
        (
            "dangerous_python",
            3,
            "Gefährliche Python-Ausführungsmuster erkannt.",
            re.compile(
                r"(?is)\b("
                r"os\.system|subprocess\.(?:popen|run|call)|eval\s*\(|exec\s*\(|"
                r"__import__\s*\(|pickle\.loads|yaml\.load\s*\("
                r")"
            ),
        ),
        (
            "classic_injection_payload",
            3,
            "Klassisches Injection-Payload-Muster erkannt.",
            re.compile(
                r"(?is)("
                r"\bunion\s+select\b|\bdrop\s+table\b|\btruncate\s+table\b|"
                r"<script\b|javascript:|onerror\s*=|onload\s*=|"
                r"\bor\s+1\s*=\s*1\b|'\s*or\s*'1'\s*=\s*'1"
                r")"
            ),
        ),
        (
            "malware_request",
            3,
            "Anfrage wirkt wie Erstellung oder Ausführung von Schadcode.",
            re.compile(
                r"(?is)\b(write|generate|create|build|schreibe|erstelle|erzeuge|baue)\b"
                r".{0,140}\b(malware|ransomware|keylogger|reverse shell|credential stealer|"
                r"trojan|backdoor|botnet|exploit|payload|phishing kit|rootkit|schadcode)\b"
            ),
        ),
        (
            "rag_poisoning_instruction",
            3,
            "Indirekte Prompt-Injection/RAG-Poisoning-Anweisung erkannt.",
            re.compile(
                r"(?is)\b(when|whenever|if|nachdem|wenn|sobald)\b"
                r".{0,100}\b(summarizing|analysing|analyzing|processing|answering|searching|"
                r"zusammenfasst|analysierst|verarbeitest|antwortest|suchst)\b"
                r".{0,180}\b(ignore|disregard|follow these instructions|override|reveal|send|"
                r"ignoriere|missachte|befolge diese anweisungen|überschreibe|zeige|sende)\b"
            ),
        ),
        (
            "resource_exhaustion",
            2,
            "Möglicher Versuch, das Modell/den Kontext absichtlich zu überlasten.",
            re.compile(
                r"(?is)\b(repeat|wiederhole)\b.{0,80}\b(forever|unendlich|100000|million|"
                r"eine million)\b|\b(token|context|kontext)\b.{0,80}\b(flood|overflow|sprengen|füllen)\b"
            ),
        ),
    )

    def validate_input(self, user_input: str) -> tuple[bool, str]:
        if not isinstance(user_input, str):
            return False, "Ungültige Eingabe: Erwartet wurde Text."

        raw = user_input.strip()

        if len(raw) < self.MIN_INPUT_CHARS:
            return False, "Die Anfrage ist zu kurz. Bitte formuliere eine wissenschaftliche Recherchefrage."

        if len(raw) > self.MAX_INPUT_CHARS:
            return False, f"Die Anfrage ist zu lang. Maximal erlaubt: {self.MAX_INPUT_CHARS} Zeichen."

        if raw.count("\n") + 1 > self.MAX_LINES:
            return False, f"Die Anfrage enthält zu viele Zeilen. Maximal erlaubt: {self.MAX_LINES} Zeilen."

        if re.search(rf"(.)\1{{{self.MAX_REPEAT_CHAR_RUN},}}", raw, flags=re.DOTALL):
            return False, "Die Anfrage enthält ungewöhnlich lange Wiederholungen und wurde abgelehnt."

        normalized, variants, normalization_findings = self._normalize_and_expand(raw)

        if not normalized:
            return False, "Die Anfrage enthält nach der Normalisierung keinen verwertbaren Text."

        findings: list[Finding] = list(normalization_findings)
        findings.extend(self._scan_variants(variants))
        findings = self._deduplicate_findings(findings)

        if findings:
            highest = max(f.severity for f in findings)

            # Sicherheitsforschung darf Begriffe wie "malware" oder "prompt injection"
            # enthalten, solange die Anfrage klar akademisch/analytisch und nicht
            # handlungsorientiert ist.
            if highest <= 2 and self._looks_like_research_query(normalized):
                return True, "Eingabe akzeptiert: Sicherheitsthema wirkt akademisch und nicht handlungsorientiert."

            reasons = "; ".join(f"{f.category}: {f.reason}" for f in findings[:3])
            return False, f"Anfrage aus Sicherheitsgründen abgelehnt. {reasons}"

        if not self._looks_like_research_query(normalized):
            return False, (
                "Die Anfrage passt nicht klar zum Zweck von SmartScholar. "
                "Bitte stelle eine wissenschaftliche Recherchefrage, z. B. nach Papers, Studien, Methoden oder Literatur."
            )

        return True, "Eingabe akzeptiert: keine offensichtliche Prompt-Injection oder Schadcode-Anfrage erkannt."

    def _normalize_and_expand(self, raw: str) -> tuple[str, list[str], list[Finding]]:
        findings: list[Finding] = []

        text = unicodedata.normalize("NFKC", raw)
        text = html.unescape(text)
        text = unquote(text)

        if self._ZERO_WIDTH.search(text):
            findings.append(
                Finding(
                    category="hidden_unicode",
                    reason="Unsichtbare Unicode-/Richtungssteuerzeichen erkannt.",
                    severity=2,
                )
            )
            text = self._ZERO_WIDTH.sub("", text)

        if self._CONTROL.search(text):
            findings.append(
                Finding(
                    category="control_characters",
                    reason="Kontrollzeichen erkannt.",
                    severity=2,
                )
            )
            text = self._CONTROL.sub(" ", text)

        normalized = self._WHITESPACE.sub(" ", text).strip()
        variants = {normalized, normalized.lower()}

        for token in self._LONG_B64.findall(text):
            decoded = self._try_decode_base64(token)
            if decoded:
                decoded = decoded[: self.MAX_BASE64_DECODED_CHARS]
                decoded_norm = self._WHITESPACE.sub(" ", decoded).strip()
                if decoded_norm:
                    variants.add(decoded_norm)
                    variants.add(decoded_norm.lower())
                    findings.append(
                        Finding(
                            category="encoded_content",
                            reason="Base64/base64url-artiger Inhalt wurde erkannt und geprüft.",
                            severity=1,
                        )
                    )

        return normalized, sorted(variants), findings

    def _try_decode_base64(self, token: str) -> str | None:
        candidate = token.strip().replace("-", "+").replace("_", "/")
        candidate += "=" * ((-len(candidate)) % 4)

        try:
            decoded = base64.b64decode(candidate, validate=True)
        except (binascii.Error, ValueError):
            return None

        if not decoded:
            return None

        printable = sum(32 <= b <= 126 or b in (9, 10, 13) for b in decoded)
        if printable / len(decoded) < 0.85:
            return None

        return decoded.decode("utf-8", errors="ignore")

    def _scan_variants(self, variants: list[str]) -> list[Finding]:
        findings: list[Finding] = []

        for text in variants:
            for category, severity, reason, pattern in self._DENY_RULES:
                if pattern.search(text):
                    if severity <= 2 and self._looks_like_research_query(text):
                        continue
                    findings.append(Finding(category=category, reason=reason, severity=severity))

        return findings

    def _looks_like_research_query(self, text: str) -> bool:
        has_research_hint = bool(self._RESEARCH_HINTS.search(text))
        has_question_shape = bool(re.search(r"\?$|\b(what|which|how|why|welche|was|wie|warum)\b", text, re.I))
        has_actionable_attack = bool(self._ATTACK_ACTIONS.search(text))

        if has_research_hint and not has_actionable_attack:
            return True

        return has_question_shape and len(text.split()) >= 5 and not has_actionable_attack

    @staticmethod
    def _deduplicate_findings(findings: list[Finding]) -> list[Finding]:
        seen: set[str] = set()
        unique: list[Finding] = []

        for finding in sorted(findings, key=lambda f: f.severity, reverse=True):
            if finding.category in seen:
                continue
            seen.add(finding.category)
            unique.append(finding)

        return unique