"""
7 intent-specific prompt templates for 3GPP Q&A.
Each template slots in {context} (retrieved chunks) and {question}.

All templates share the same hard grounding rules: every claim must cite a
chunk from the context, and named entities (service operations, interfaces,
parameters) that don't appear verbatim in the context must NOT be invented.
"""

# Hard grounding rules prepended to every intent template. These are the
# anti-hallucination guard rails — they fire on EVERY claim, every name, every
# spec reference. Tightened after observing repeated service-name fabrications
# (e.g. "Nnwdaf_EventSubscription" instead of "Nnwdaf_EventsSubscription") and
# uncited generic prose in compare-style answers.
_GROUNDING_RULES = """# Grounding rules — read carefully

1. **Cite every claim.** Every factual sentence MUST end with `[spec_id §section]`
   matching a chunk from the context above. Format example:
       "NWDAF collects data from 5GC NFs [ts_23.288 §4.1]."
   If a sentence has no supporting chunk, DO NOT WRITE IT.

2. **Never invent named entities.** Service operations (Nxxx_Yyy_Zzz), interface
   names (Nxxx), parameter names (S-NSSAI, DNN…), and procedure names appear in
   the context EXACTLY or NOT AT ALL. If you want a name you cannot find in the
   context, write: "the context does not specify the exact <service operation |
   interface | parameter> name." Never guess a plausible-looking variant.

3. **No synthesis from training knowledge.** If the context lacks evidence for a
   sub-question, write one line: "Context does not cover <sub-question>." Then move
   on. Do NOT fill the gap with general 5G knowledge. (The ONE exception is Rule 8:
   a named service operation between the two asked-about entities may be stated
   generally, with its source context noted.)
   (An attempt to forbid redundant completeness disclaimers here BACKFIRED and was
   reverted: spelling out "do not write 'the context does not specify any other X'"
   raised the hedge count from 12 to 17 per 100 answers and the median answer from
   56 to 62 words, costing 0.030 Cover-EM — naming the unwanted phrase makes it more
   salient, not less. Retrieval was byte-identical, so the comparison was clean.
   If this is retried, do it by REWARDING brevity, never by naming the phrase.)

4. **Prefer concrete over generic.** When two chunks support a claim, cite the one
   with the most specific section (e.g. §6.2.18 over §4.1), and prefer a
   procedure-flow chunk over a bullet-list / definition chunk.

5. **Lead with the core, label the branches.** BEFORE listing details, identify
   the interaction/information the asked-about procedure is really built around and
   present it FIRST. Secondary, conditional, or special-branch interactions (SMS,
   disaster roaming, localized services, interworking, re-allocation) go AFTER and
   MUST be explicitly labelled as a side-branch/conditional case — never mixed in
   as peers of the main flow.

6. **Role and direction of every interaction.** When the question asks how X
   interacts with Y, EVERY point MUST have BOTH X and Y as the real interacting
   parties. For each service operation: identify the CONSUMER (who invokes it) and
   the PRODUCER (whose name-prefix it carries); keep it only if the pair is exactly
   X↔Y with the right roles; never reverse the direction (X→Y ≠ Y→X). A chunk that
   describes X↔Z (e.g. AMF↔PCF when the question is AMF↔UDM) is NOT an X↔Y
   interaction — drop it, or label it explicitly as a related-but-different one. A
   chunk merely *mentioning* Y in passing is not enough — Y must be a party to the
   interaction the point describes.

7. **Scan fully, but do not claim completeness.** FIRST scan EVERY chunk for
   service operations exchanged between the two asked-about entities and list each
   one you find — including operations inside a narrow-scenario chunk (interworking,
   roaming, re-allocation): do not omit one merely because its chunk's scenario is
   narrow (apply Rule 8). BUT do NOT conclude "there are no other interactions" —
   that is a strong negative the context (a partial slice of the spec) can't
   support. Instead scope it: "within the provided context, the recorded
   interactions are…".

8. **Controlled generalisation, with source context noted.** A service operation
   between EXACTLY the two asked-about entities is a stable interface: if a chunk
   shows X invoking `Nyyy_…` on Y — even in a narrow scenario — you MAY state it as
   a general X↔Y interaction, PROVIDED you note the source context in the same
   sentence, e.g. "the AMF registers with the UDM via `Nudm_UECM_Registration`
   [ts_23.632 §5.3.4] (shown here in the interworking flow, but the operation is
   the general registration interface)." Applies ONLY to named operations between
   the two asked-about entities — never to invent behaviour or generalise a
   third-party interaction.
   HARD FILTER: the operation's producer prefix MUST be one of the two entities'
   own prefixes. For an AMF↔UDM question keep ONLY `Nudm_…` and `Namf_…`. An
   `Nsmsf_…`, `Nsmf_…`, `Nnef_…` operation is produced by a THIRD entity — drop it
   even if the AMF invokes it (it is AMF↔that-third-entity). Likewise drop any point
   whose parties are the AMF and the UE, or a service-related entity.

9. **Faithfulness and gaps.** Use only information in the context. If the context
   lacks the CORE of the question, state the gap plainly (Rule 3) rather than
   substituting an available side-branch as if it were the answer.

10. **Check the grammatical SUBJECT of the very sentence you paraphrase.** An
    operation's prefix gives the PRODUCER, NOT who the consumer is in a given
    sentence. The SAME operation (e.g. `Nudm_UECM_Registration`) can appear in one
    chunk invoked by DIFFERENT actors — the AMF in one step, the SMSF in another.
    Before assigning a sentence to the X↔Y interaction, read the actual subject of
    THAT sentence:
      - "the SMSF registers with the UDM using Nudm_UECM_Registration" is SMSF↔UDM,
        NOT AMF↔UDM — even though the operation is `Nudm_…` and the chunk is a
        Registration procedure. Drop it or label the correct actor.
      - Never describe a sentence whose subject is Z (SMSF, HSS, NEF, SMF…) and then
        label it as X. The grammatical subject MUST be one of the two asked entities.
      - When the same operation appears with several subjects in one chunk, pick the
        sentence whose subject is the asked entity; if none exists, that operation is
        NOT an X↔Y interaction in this chunk.
    This is NARROWER than Rule 8: Rule 8 checks producer/consumer are the right
    pair; Rule 10 checks the subject of the specific sentence you paraphrase is the
    asked entity, not another actor that happens to use the same operation nearby."""


# Per-intent role + structure instructions. The full prompt is assembled by
# concatenating: grounding rules → intent-specific instructions → context →
# question. Keeping rules first means the LLM reads them before any context.
_INTENT_INSTRUCTIONS: dict[str, str] = {
    "definition": (
        "You are a 3GPP technical expert. Define the term or network function asked about,"
        " using ONLY the provided context. Cover: full name, purpose, key interfaces, and"
        " relevant spec references — but ONLY include each item if a chunk supports it."
        " If the context does not specify an item, write \"context does not specify\" for"
        " that item rather than skipping or fabricating."
    ),
    "procedure": (
        "You are a 3GPP technical expert. Describe the procedure step by step using ONLY"
        " the provided context. Number each step and end each step with the supporting"
        " citation `[spec_id §section]`. If a step cannot be supported by any chunk, do"
        " not include it; instead, after listing the supported steps, add: \"Context does"
        " not specify subsequent steps.\""
    ),
    "comparison": (
        "You are a 3GPP technical expert. Compare the entities asked about using ONLY the"
        " provided context. Structure as a markdown table with one row per attribute and"
        " one column per entity. Every cell must contain either a cited fact"
        " `[spec_id §section]` OR the literal string \"context does not specify\". Do not"
        " leave cells blank and do not fabricate values to make rows symmetric."
    ),
    "reference": (
        "You are a 3GPP technical expert. List the specification references mentioned in"
        " the context. For each reference: spec id, full title (only if a chunk provides"
        " it), and the citation. If the chunk does not give the title, write the spec id"
        " alone — do not guess the title."
    ),
    "network_function": (
        "You are a 3GPP technical expert. Describe the network function using ONLY the"
        " provided context. Cover three areas with separate paragraphs: (1) role in 5G"
        " architecture, (2) interfaces / service operations, (3) key procedures. For (2),"
        " ONLY name an interface or service operation that appears verbatim in a chunk;"
        " if no chunk names them, write \"Context does not specify the exact interface"
        " names.\" For (3), only list procedures the context explicitly describes."
    ),
    "relationship": (
        "You are a 3GPP technical expert. Explain how the two asked-about entities relate,"
        " using ONLY the provided context. Follow this order (goal.md Part C):\n"
        "1. IDENTIFY THE CORE. Name the main procedure/relationship the question is about"
        " and the interaction it is built around; present that FIRST (Rule 5).\n"
        "2. SCAN service operations (Rule 7). Go through EVERY chunk and list each"
        " `Nxxx_Yyy_Zzz` exchanged between the two entities. For each one, apply BEFORE"
        " listing:\n"
        "   - Rule 6/8 (role + direction): consumer and producer must be exactly the two"
        " entities; state the direction explicitly ('X invokes Nyyy_… on Y'); HARD-DROP any"
        " operation whose producer prefix is a THIRD entity (Nsmsf_/Nsmf_/Nnef_… for an"
        " AMF↔UDM question), or whose parties are an entity and the UE / a service entity.\n"
        "   - Rule 10 (sentence subject): the grammatical SUBJECT of the sentence you"
        " paraphrase must be one of the two entities. 'the SMSF registers with the UDM via"
        " Nudm_UECM_Registration' is SMSF↔UDM — DROP it, do not relabel it AMF↔UDM.\n"
        "   - Rule 8 (generalisation): keep an operation from a narrow-scenario chunk"
        " (interworking, roaming, SMS-over-NAS), stating the source context; do not omit it"
        " just because the scenario is narrow.\n"
        "3. ORDER: main flow first, then side-branches EXPLICITLY LABELLED as conditional/"
        "special-case (Rule 5). Do NOT pad with tangential flows (hosting-network / localized"
        " services) that only incidentally contain both entities.\n"
        "4. Cite every point (Rule 1). End by SCOPING, not by a completeness claim: 'within"
        " the provided context, the recorded interactions are…' (Rule 7) — never 'there are"
        " no other interactions'."
    ),
    "general": (
        "You are a 3GPP technical expert. Answer the question using ONLY the provided"
        " context from 3GPP specifications. If the context does not contain enough"
        " information, say so clearly per missing aspect."
    ),
}


PROMPT_TEMPLATES: dict[str, str] = {
    intent: f"""{_GROUNDING_RULES}

# Task
{instructions}

Context:
{{context}}

Question: {{question}}

Answer:"""
    for intent, instructions in _INTENT_INSTRUCTIONS.items()
}


# Appended (not prepended) under force_answer so it is the LAST instruction the model
# reads and overrides grounding Rule 3. Needed for forced-choice benchmarks such as
# TeleQnA: without it the grounding rules make the model reply "Context does not
# cover X" on general-IT / research questions the 3GPP corpus never covers, which the
# MCQ scorer counts as an extraction failure (measured 3/6 on a gemma4 trial).
_FORCE_ANSWER_NOTICE = """

# FORCED ANSWER — this overrides Rule 3 above
You MUST answer; refusing or writing "Context does not cover…" is not allowed.
Use the context when it is relevant. When it is not, answer from your own
telecommunications knowledge instead. If the question lists options, pick exactly
ONE of them, never "none of the above" unless it is itself one of the options.

Answer:"""


def build_prompt(
    intent: str, context: str, question: str,
    scope_warnings: list[str] | None = None,
    force_answer: bool = False,
) -> str:
    """Fill in a prompt template for the given intent. When `scope_warnings` is
    non-empty (out-of-scope / fabricated entity detected, see pipeline.scope_check),
    prepend an explicit ABSTAIN directive so the model refuses the premise instead
    of hallucinating an answer around a plausible-but-nonexistent entity.
    `force_answer` does the opposite: drop the abstain directive and require an
    answer, falling back to parametric knowledge when the context is silent."""
    template = PROMPT_TEMPLATES.get(intent, PROMPT_TEMPLATES["general"])
    body = template.format(context=context, question=question)
    if force_answer:
        # Every template ends with "Answer:"; replace it so the notice sits last.
        if body.rstrip().endswith("Answer:"):
            body = body.rstrip()[: -len("Answer:")].rstrip()
        return body + _FORCE_ANSWER_NOTICE
    if scope_warnings:
        notice = (
            "# SCOPE CHECK — READ FIRST\n"
            "The question introduces one or more entities/topics that DO NOT exist "
            "in the 3GPP corpus:\n"
            + "\n".join(f"  - {w}" for w in scope_warnings)
            + "\nDo NOT invent facts about a non-existent entity. State plainly that "
            "the corpus does not define it, briefly say what the corpus DOES cover "
            "for the closest real entity if relevant, and stop. Do not fabricate a "
            "plausible-sounding answer.\n\n"
        )
        return notice + body
    return body


# Parametric (no-retrieval) prompt for the LLM-only ablation config: the model
# answers from its own knowledge with NO context. Deliberately omits the grounding
# rules ("use ONLY context") — with empty context they'd force a refusal, defeating
# the LLM-only setup.
_LLM_ONLY_PROMPT = """You are a 3GPP / telecommunications expert. Answer the question below using your own knowledge. Be concise.

Question: {question}

Answer:"""


def build_llm_only_prompt(question: str) -> str:
    """Prompt for the LLM-only config (no retrieval) — parametric answer."""
    return _LLM_ONLY_PROMPT.format(question=question)
