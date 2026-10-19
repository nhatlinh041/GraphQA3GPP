import React from 'react';

/** Rater instructions. Shown on the sign-in card and from the "?" button; the
 *  same text lives nowhere else so the rubric cannot drift from the form. */
export function Guide({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`text-[13px] leading-relaxed text-gray-700 space-y-3 ${compact ? '' : 'max-w-2xl'}`}>
      <p>
        You will validate questions sampled from <b>3GPP-ClauseQA</b>. Each item shows the question, the full text
        of the 3GPP clauses labelled as <b>gold</b> (the evidence), and, in step 2, a <b>reference answer</b>.
        Every question is rated by two people independently — please <b>do not discuss</b> items with the other rater
        while the review is running.
      </p>
      <ol className="list-decimal pl-5 space-y-2">
        <li>
          <b>Step 1 (before seeing the reference answer).</b> Read the question and the gold clauses, then answer:
          <ul className="list-disc pl-5 mt-1 space-y-1">
            <li><b>C1 · Is the question valid?</b> Pass / Fail / Unsure. Fail = technically wrong, ambiguous, false premise, or not defined in the specifications.</li>
            <li><b>C2 · Do the gold clauses contain the answer?</b> Yes, fully / Partially / No. If a clause does <i>not</i> help answer the question, tick "not relevant" on that clause.</li>
            <li><b>Your own answer</b> (required, brief): answer from the gold clauses. If they do not answer the question, write "cannot be answered from gold" and why.</li>
          </ul>
        </li>
        <li>
          <b>Step 2 (after clicking "Show reference answer").</b>
          <ul className="list-disc pl-5 mt-1 space-y-1">
            <li><b>C3 · Is the reference answer correct?</b> Correct / Partially correct / Wrong — judged against the gold clauses, not against memory.</li>
            <li><b>Verdict:</b> Keep / Fix / Discard. Fix and Discard require a comment; if you know the fix, put it in "Suggested fix".</li>
          </ul>
        </li>
      </ol>
      <p>
        Time per step is recorded automatically. You can save a draft and come back; a completed item can still be
        edited, but the system keeps the answers you gave <i>before</i> seeing the reference answer. Expect roughly
        3–6 minutes per question; long clauses are collapsed, click to expand.
      </p>
    </div>
  );
}
