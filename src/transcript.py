"""PDF transcript generation for saved model runs.

The transcript intentionally uses only the Python standard library so batch
workers do not need an additional PDF dependency.
"""

import json
import textwrap
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PdfLine = Tuple[str, bool, float]


def write_pdf_transcript(state_or_sim_data: Any, pdf_path: str, source_path: Optional[str] = None) -> str:
    """Write a complete run transcript PDF.

    Args:
        state_or_sim_data: Either a saved-state dict or SimulationData-like
            object with to_state().
        pdf_path: Destination PDF path.
        source_path: Optional source JSON path to print in the transcript.

    Returns:
        The PDF path that was written.
    """
    state = _as_state(state_or_sim_data)
    pdf = _SimplePdf()
    _build_transcript(pdf, state, source_path=source_path)
    pdf.write(pdf_path)
    return pdf_path


def write_pdf_transcript_from_state_file(state_path: str, pdf_path: Optional[str] = None) -> str:
    """Generate a transcript PDF from an existing saved state JSON file."""
    with open(state_path, "r", encoding="utf-8") as handle:
        state = json.load(handle)

    if pdf_path is None:
        if state_path.endswith(".json"):
            pdf_path = state_path[:-5] + ".transcript.pdf"
        else:
            pdf_path = state_path + ".transcript.pdf"

    return write_pdf_transcript(state, pdf_path, source_path=state_path)


def _as_state(state_or_sim_data: Any) -> Dict[str, Any]:
    if isinstance(state_or_sim_data, dict):
        return state_or_sim_data
    if hasattr(state_or_sim_data, "to_state"):
        return state_or_sim_data.to_state(include_agent_histories=True)
    raise TypeError("Expected a saved state dict or SimulationData-like object.")


def _build_transcript(pdf: "_SimplePdf", state: Dict[str, Any], source_path: Optional[str]) -> None:
    agents = state.get("agents") or {}
    show_agent_names = _show_agent_names_in_transcript(state)
    interactions = _collect_interactions(state)
    interactions_by_round: Dict[int, List[Dict[str, Any]]] = {}
    unscoped_interactions: List[Dict[str, Any]] = []
    for event in interactions:
        round_number = _int_or_none((event.get("metadata") or {}).get("round"))
        if round_number is None:
            unscoped_interactions.append(event)
        else:
            interactions_by_round.setdefault(round_number, []).append(event)

    pdf.add_heading("MODEL RUN TRANSCRIPT")
    pdf.add_kv("Generated", datetime.now().isoformat(timespec="seconds"))
    if source_path:
        pdf.add_kv("Source", source_path)
    _add_run_metadata(pdf, state, show_agent_names)

    if interactions:
        pdf.add_wrapped(
            "Coverage: exact chat message snapshots sent to the model are included for every recorded LLM call."
        )
    else:
        pdf.add_wrapped(
            "Coverage: this state has no per-call interaction_history. It was likely produced before full transcript logging; only saved round data and final agent messages can be shown."
        )

    _add_agent_roster(pdf, agents, show_agent_names)

    conversation_history = state.get("conversation_history") or []
    for entry in conversation_history:
        round_number = entry.get("round", "?")
        pdf.add_section(f"ROUND {round_number}")
        _add_round_summary(pdf, entry, agents, show_agent_names)

        round_events = interactions_by_round.get(_int_or_none(round_number), [])
        if round_events:
            pdf.add_heading("LLM CALLS")
            for event in round_events:
                _add_interaction(pdf, event, show_agent_names)
        else:
            _add_legacy_round_responses(pdf, entry)

        _add_written_myths(pdf, entry, agents, show_agent_names)

    if unscoped_interactions:
        pdf.add_section("UNSCOPED LLM CALLS")
        for event in unscoped_interactions:
            _add_interaction(pdf, event, show_agent_names)

    if not interactions:
        _add_final_agent_messages(pdf, agents)


def _show_agent_names_in_transcript(state: Dict[str, Any]) -> bool:
    value = (state.get("run_metadata") or {}).get("show_agent_names", True)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return True


def _add_run_metadata(pdf: "_SimplePdf", state: Dict[str, Any], show_agent_names: bool) -> None:
    metadata = state.get("run_metadata") or {}
    game_data = state.get("game_data") or {}
    task_order = state.get("task_order")

    pdf.add_heading("RUN METADATA")
    for key in [
        "model",
        "temperature",
        "num_turns",
        "num_agents",
        "memory_capacity",
        "myth_topic_id",
        "myth_topic",
        "game_params_name",
        "noise_config",
        "other_player_names",
        "replicate_id",
        "myth_prompt_arm_id",
        "myth_default_prompt_key",
        "myth_later_prompt_key",
        "system_addition_key",
        "history_policy",
        "self_history_window",
        "coplayer_history_window",
        "show_agent_names",
        "defector_ratio_requested",
        "defector_ratio_actual",
        "defector_count",
        "defector_agent_ids",
        "defector_seed",
    ]:
        if key in metadata:
            pdf.add_kv(key, metadata.get(key))
    if task_order is not None:
        pdf.add_kv("task_order", task_order)
    if game_data.get("agent_names") and show_agent_names:
        pdf.add_kv("agent_names", game_data.get("agent_names"))
    elif game_data.get("agent_names"):
        pdf.add_kv(
            "agent_names",
            "hidden in transcript labels because show_agent_names is false",
        )
    if game_data.get("agent_types"):
        pdf.add_kv("agent_types", game_data.get("agent_types"))
    pairing_schedule = game_data.get("dyadic_pairing_schedule")
    if pairing_schedule:
        pdf.add_kv(
            "dyadic_pairing_schedule",
            {
                "strategy": pairing_schedule.get("strategy"),
                "num_turns": pairing_schedule.get("num_turns"),
                "role_targets": pairing_schedule.get("role_targets"),
            },
        )


def _add_agent_roster(pdf: "_SimplePdf", agents: Dict[str, Any], show_agent_names: bool) -> None:
    if not agents:
        return

    pdf.add_heading("AGENTS")
    for agent_id, agent in sorted(agents.items()):
        display_name = agent.get("display_name") or agent_id
        label = str(agent_id)
        if show_agent_names and display_name != agent_id:
            label = f"{agent_id} ({display_name})"
        parts = [
            label,
            f"model={agent.get('model', 'unknown')}",
            f"memory_capacity={agent.get('memory_capacity', 'unknown')}",
        ]
        if agent.get("initial_bias"):
            parts.append(f"initial_bias={agent.get('initial_bias')}")
        pdf.add_wrapped("; ".join(parts))

        system_prompt = agent.get("system_prompt")
        if system_prompt:
            pdf.add_wrapped("System prompt:", indent="  ", bold=True)
            pdf.add_preformatted(system_prompt, indent="    ")


def _add_round_summary(
    pdf: "_SimplePdf",
    entry: Dict[str, Any],
    agents: Dict[str, Any],
    show_agent_names: bool,
) -> None:
    roles = entry.get("roles") or {}
    if roles:
        role_text = ", ".join(
            f"{_agent_label(agent_id, agents, show_agent_names)}={_role_label(role)}"
            for agent_id, role in sorted(roles.items())
        )
        pdf.add_wrapped(f"Roles: {role_text}")

    pairings = entry.get("pairings") or []
    if pairings:
        pdf.add_wrapped("Pairings:", bold=True)
        for pairing in pairings:
            investor = pairing.get("investor")
            trustee = pairing.get("trustee")
            pdf.add_wrapped(
                f"{pairing.get('dyad_id', 'dyad')}: "
                f"{_agent_label(investor, agents, show_agent_names)}=Sender, "
                f"{_agent_label(trustee, agents, show_agent_names)}=Receiver",
                indent="  ",
            )

    dyads = _iter_dyads(entry)
    if dyads:
        pdf.add_wrapped("Parsed game decisions:", bold=True)
        for dyad in dyads:
            pdf.add_wrapped(
                _format_dyad_decision(dyad, agents, show_agent_names),
                indent="  ",
            )
            action_validation = dyad.get("action_validation")
            if action_validation:
                pdf.add_wrapped(
                    "Action validation: " + _compact_json(action_validation),
                    indent="    ",
                )

    balances = entry.get("balances")
    if balances:
        pdf.add_kv("Balances after round", balances)
    balances_communicated = entry.get("balances_communicated")
    if balances_communicated:
        pdf.add_kv("Visible balances after round", balances_communicated)


def _add_interaction(pdf: "_SimplePdf", event: Dict[str, Any], show_agent_names: bool) -> None:
    metadata = _metadata_for_transcript(event.get("metadata") or {}, show_agent_names)
    task = str(metadata.get("task") or "interaction").upper()
    role = metadata.get("role_label") or _role_label(metadata.get("role"))
    agent_id = event.get("agent_id", "unknown")
    title_parts = [task, str(agent_id)]
    if role:
        title_parts.append(str(role))
    if metadata.get("dyad_id"):
        title_parts.append(str(metadata["dyad_id"]))
    pdf.add_heading(" - ".join(title_parts))

    pdf.add_kv("Timestamp", event.get("timestamp"))
    pdf.add_kv("Model", event.get("model"))
    pdf.add_kv("Temperature", event.get("temperature"))
    if metadata:
        pdf.add_kv("Metadata", metadata)

    messages = event.get("messages_sent") or []
    if messages:
        pdf.add_wrapped("Exact input messages sent to model:", bold=True)
        _add_messages(pdf, messages)
    elif event.get("prompt"):
        pdf.add_wrapped("Prompt:", bold=True)
        pdf.add_preformatted(event.get("prompt"), indent="  ")

    if event.get("error"):
        pdf.add_wrapped("LLM call failed:", bold=True)
        pdf.add_preformatted(_compact_json(event["error"]), indent="  ")
        return

    response = event.get("response") or {}
    pdf.add_wrapped("Assistant output:", bold=True)
    pdf.add_preformatted(response.get("content", ""), indent="  ")

    reasoning = response.get("reasoning")
    if reasoning:
        pdf.add_wrapped("Reasoning:", bold=True)
        pdf.add_preformatted(reasoning, indent="  ")

    usage = response.get("usage")
    if usage:
        pdf.add_kv("Usage", usage)


def _add_messages(pdf: "_SimplePdf", messages: List[Dict[str, Any]]) -> None:
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "unknown")
        pdf.add_wrapped(f"Message {index} role={role}", indent="  ", bold=True)

        if "content" in message:
            pdf.add_wrapped("content:", indent="    ", bold=True)
            pdf.add_preformatted(message.get("content", ""), indent="      ")

        for key, value in message.items():
            if key in {"role", "content"} or value is None:
                continue
            pdf.add_wrapped(f"{key}:", indent="    ", bold=True)
            pdf.add_preformatted(_stringify(value), indent="      ")


def _add_legacy_round_responses(pdf: "_SimplePdf", entry: Dict[str, Any]) -> None:
    for response_key, label in [
        ("game_responses", "Saved game responses"),
        ("myth_responses", "Saved myth responses"),
    ]:
        responses = entry.get(response_key) or {}
        if not responses:
            continue
        pdf.add_wrapped(label + ":", bold=True)
        for agent_id, response in sorted(responses.items()):
            pdf.add_wrapped(str(agent_id), indent="  ", bold=True)
            pdf.add_preformatted(_stringify(response), indent="    ")


def _add_written_myths(
    pdf: "_SimplePdf",
    entry: Dict[str, Any],
    agents: Dict[str, Any],
    show_agent_names: bool,
) -> None:
    myths = entry.get("myths") or {}
    if not myths:
        return

    pdf.add_wrapped("Written myths:", bold=True)
    for agent_id, myth in sorted(myths.items()):
        pdf.add_wrapped(
            _agent_label(agent_id, agents, show_agent_names),
            indent="  ",
            bold=True,
        )
        pdf.add_preformatted(myth, indent="    ")


def _add_final_agent_messages(pdf: "_SimplePdf", agents: Dict[str, Any]) -> None:
    if not agents:
        return

    pdf.add_section("FINAL SAVED AGENT MESSAGE HISTORIES")
    for agent_id, agent in sorted(agents.items()):
        pdf.add_heading(str(agent_id))
        _add_messages(pdf, agent.get("messages") or [])


def _collect_interactions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = []
    for agent_id, agent in sorted((state.get("agents") or {}).items()):
        for event in agent.get("interaction_history") or []:
            event = dict(event)
            event.setdefault("agent_id", agent_id)
            events.append(event)
    return sorted(events, key=_interaction_sort_key)


def _interaction_sort_key(event: Dict[str, Any]) -> Tuple[Any, ...]:
    metadata = event.get("metadata") or {}
    round_number = _int_or_none(metadata.get("round"))
    return (
        round_number if round_number is not None else 10**9,
        _int_or_default(metadata.get("task_index"), 10**6),
        _int_or_default(metadata.get("move_index"), 10**6),
        str(metadata.get("task", "")),
        str(event.get("agent_id", "")),
        _int_or_default(event.get("interaction_index"), 10**6),
        str(event.get("timestamp", "")),
    )


def _iter_dyads(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    dyads = entry.get("dyads") or []
    if dyads:
        return dyads

    if entry.get("sent") is None and entry.get("returned") is None:
        return []

    roles = entry.get("roles") or {}
    investor = next((agent_id for agent_id, role in roles.items() if role == "investor"), None)
    trustee = next((agent_id for agent_id, role in roles.items() if role == "trustee"), None)
    return [
        {
            "dyad_id": "dyad_1",
            "investor": investor,
            "trustee": trustee,
            "sent": entry.get("sent"),
            "sent_communicated": entry.get("sent_communicated"),
            "received": entry.get("received"),
            "received_communicated": entry.get("received_communicated"),
            "returned": entry.get("returned"),
            "returned_communicated": entry.get("returned_communicated"),
            "investor_payoff": entry.get("investor_payoff"),
            "trustee_payoff": entry.get("trustee_payoff"),
            "investor_payoff_communicated": entry.get("investor_payoff_communicated"),
            "trustee_payoff_communicated": entry.get("trustee_payoff_communicated"),
            "action_validation": entry.get("action_validation"),
        }
    ]


def _format_dyad_decision(
    dyad: Dict[str, Any],
    agents: Dict[str, Any],
    show_agent_names: bool,
) -> str:
    investor = _agent_label(dyad.get("investor"), agents, show_agent_names)
    trustee = _agent_label(dyad.get("trustee"), agents, show_agent_names)
    parts = [
        f"{dyad.get('dyad_id', 'dyad')}: {investor} sent {_money(dyad.get('sent'))}",
        f"received {_money(dyad.get('received'))}",
        f"{trustee} returned {_money(dyad.get('returned'))}",
        f"payoffs {investor}={_money(dyad.get('investor_payoff'))}, {trustee}={_money(dyad.get('trustee_payoff'))}",
    ]

    communicated = []
    for actual_key, communicated_key, label in [
        ("sent", "sent_communicated", "sent"),
        ("received", "received_communicated", "received"),
        ("returned", "returned_communicated", "returned"),
    ]:
        if communicated_key in dyad and dyad.get(communicated_key) != dyad.get(actual_key):
            communicated.append(f"{label} visible={_money(dyad.get(communicated_key))}")
    if communicated:
        parts.append("noise/visible: " + ", ".join(communicated))

    return "; ".join(parts)


def _metadata_for_transcript(metadata: Dict[str, Any], show_agent_names: bool) -> Dict[str, Any]:
    if show_agent_names:
        return metadata
    return _strip_display_names(metadata)


def _strip_display_names(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_display_names(item)
            for key, item in value.items()
            if key not in {"agent_names", "display_name"}
        }
    if isinstance(value, list):
        return [_strip_display_names(item) for item in value]
    return value


def _agent_label(agent_id: Any, agents: Dict[str, Any], show_agent_names: bool = True) -> str:
    if not agent_id:
        return "unknown"
    agent_id = str(agent_id)
    display_name = (agents.get(agent_id) or {}).get("display_name")
    if show_agent_names and display_name and display_name != agent_id:
        return f"{agent_id} ({display_name})"
    return agent_id


def _role_label(role: Any) -> str:
    if role == "investor":
        return "Sender"
    if role == "trustee":
        return "Receiver"
    if role is None:
        return ""
    return str(role)


def _money(value: Any) -> str:
    if value is None:
        return "$unknown"
    if isinstance(value, int):
        return f"${value}"
    if isinstance(value, float):
        if value.is_integer():
            return f"${int(value)}"
        return f"${value:.2f}".rstrip("0").rstrip(".")
    return "$" + str(value)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_or_default(value: Any, default: int) -> int:
    parsed = _int_or_none(value)
    return parsed if parsed is not None else default


class _SimplePdf:
    page_width = 612
    page_height = 792
    margin_x = 48
    top_y = 748
    bottom_y = 44
    line_height = 10
    wrap_width = 100
    code_width = 96

    def __init__(self) -> None:
        self.lines: List[PdfLine] = []

    def add_section(self, text: str) -> None:
        self.add_line()
        self.add_line("=" * self.wrap_width, bold=True, size=9)
        self.add_line(str(text), bold=True, size=9)
        self.add_line("=" * self.wrap_width, bold=True, size=9)

    def add_heading(self, text: str) -> None:
        self.add_line()
        self.add_line(str(text), bold=True, size=8.5)

    def add_kv(self, key: str, value: Any) -> None:
        if value is None or value == "":
            return
        self.add_wrapped(f"{key}: {_stringify(value)}")

    def add_wrapped(self, text: Any, indent: str = "", bold: bool = False, width: Optional[int] = None) -> None:
        text = _pdf_safe_text(_stringify(text))
        width = width or self.wrap_width
        if text == "":
            self.add_line(indent, bold=bold)
            return

        for raw_line in text.splitlines() or [""]:
            if raw_line == "":
                self.add_line(indent, bold=bold)
                continue
            wrapped = textwrap.wrap(
                raw_line,
                width=max(10, width - len(indent)),
                break_long_words=True,
                break_on_hyphens=False,
            )
            for line in wrapped or [""]:
                self.add_line(indent + line, bold=bold)

    def add_preformatted(self, text: Any, indent: str = "  ") -> None:
        text = _pdf_safe_text(_stringify(text)).replace("\t", "  ")
        lines = text.splitlines()
        if not lines:
            self.add_line(indent)
            return

        chunk_width = max(10, self.code_width - len(indent))
        continuation_indent = indent + "  "
        continuation_width = max(10, self.code_width - len(continuation_indent))
        for raw_line in lines:
            if raw_line == "":
                self.add_line(indent)
                continue
            line = raw_line
            first = True
            while len(line) > (chunk_width if first else continuation_width):
                width = chunk_width if first else continuation_width
                current_indent = indent if first else continuation_indent
                self.add_line(current_indent + line[:width])
                line = line[width:]
                first = False
            current_indent = indent if first else continuation_indent
            self.add_line(current_indent + line)

    def add_line(self, text: str = "", bold: bool = False, size: float = 8.0) -> None:
        self.lines.append((_pdf_safe_text(text), bold, size))

    def write(self, filepath: str) -> None:
        path = Path(filepath)
        if path.parent != Path(""):
            path.parent.mkdir(parents=True, exist_ok=True)

        pages = self._paginate()
        objects: List[bytes] = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")

        page_object_numbers = []
        next_object = 5
        for _ in pages:
            page_object_numbers.append(next_object)
            next_object += 2

        kids = " ".join(f"{number} 0 R" for number in page_object_numbers)
        objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"))
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Bold >>")

        total_pages = len(pages)
        for page_index, page_lines in enumerate(pages, start=1):
            page_obj_num = page_object_numbers[page_index - 1]
            content_obj_num = page_obj_num + 1
            content = self._page_content(page_lines, page_index, total_pages)
            page_obj = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.page_width} {self.page_height}] "
                f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
                f"/Contents {content_obj_num} 0 R >>"
            ).encode("ascii")
            content_obj = b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream"
            objects.append(page_obj)
            objects.append(content_obj)

        _write_pdf_objects(path, objects)

    def _paginate(self) -> List[List[PdfLine]]:
        max_lines = max(1, int((self.top_y - self.bottom_y) / self.line_height))
        if not self.lines:
            return [[("", False, 8.0)]]
        return [
            self.lines[index : index + max_lines]
            for index in range(0, len(self.lines), max_lines)
        ]

    def _page_content(self, page_lines: List[PdfLine], page_index: int, total_pages: int) -> bytes:
        commands = ["q"]
        y = self.top_y
        for text, bold, size in page_lines:
            if text:
                font = "F2" if bold else "F1"
                escaped = _escape_pdf_text(text)
                commands.append(
                    f"BT /{font} {size:.1f} Tf {self.margin_x:.1f} {y:.1f} Td ({escaped}) Tj ET"
                )
            y -= self.line_height

        footer = _escape_pdf_text(f"Page {page_index} of {total_pages}")
        commands.append(f"BT /F1 7.0 Tf {self.margin_x:.1f} 24.0 Td ({footer}) Tj ET")
        commands.append("Q")
        return "\n".join(commands).encode("latin-1", "replace")


def _write_pdf_objects(path: Path, objects: List[bytes]) -> None:
    pdf = bytearray()
    pdf.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )

    with open(path, "wb") as handle:
        handle.write(pdf)


def _pdf_safe_text(text: Any) -> str:
    text = "" if text is None else str(text)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.encode("latin-1", "replace").decode("latin-1")


def _escape_pdf_text(text: str) -> str:
    safe = _pdf_safe_text(text).encode("latin-1", "replace")
    escaped = bytearray()
    for byte in safe:
        if byte in (40, 41, 92):
            escaped.extend(b"\\")
            escaped.append(byte)
        elif byte < 32 or byte >= 127:
            escaped.extend(f"\\{byte:03o}".encode("ascii"))
        else:
            escaped.append(byte)
    return escaped.decode("ascii")
