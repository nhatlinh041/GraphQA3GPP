import { useEffect, useRef, useState } from 'react';

// Reveal text gradually every time `text` changes — for static payloads (JSON
// input/output, prompt, a finished cypher) so they "type in" smoothly instead of
// popping in whole. Unlike useTypewriter, this always resets when the text changes.
export function useGradualReveal(text: string, charsPerTick = 6): string {
  const [revealed, setRevealed] = useState(0);

  useEffect(() => {
    setRevealed(0);
    if (!text) return;
    const id = window.setInterval(() => {
      setRevealed((cur) => {
        if (cur >= text.length) return cur;
        const backlog = text.length - cur;
        // Even reveal: at least charsPerTick, scaled by backlog so long text does not
        // drag on. Catches up within ~24 ticks (~400ms).
        const step = Math.max(charsPerTick, Math.ceil(backlog / 24));
        return Math.min(text.length, cur + step);
      });
    }, 16);
    return () => clearInterval(id);
  }, [text, charsPerTick]);

  return text.slice(0, revealed);
}

// Reveal `text` character by character at a steady rate — the backend (Ollama, SSE)
// tends to emit tokens in big bursts, so rendering them directly looks jerky. This
// hook buffers the text and drains it by backlog: paced while the stream runs, then
// flushed quickly once it ends.
export function useTypewriter(text: string, streaming: boolean): string {
  // A finished message renders in full immediately; only animate while streaming
  const [revealed, setRevealed] = useState(() => (streaming ? 0 : text.length));
  const textRef = useRef(text);
  const streamingRef = useRef(streaming);
  textRef.current = text;
  streamingRef.current = streaming;

  // When the turn ends, snap to the full text (drop the remaining animation)
  useEffect(() => {
    if (!streaming) setRevealed(text.length);
  }, [streaming, text.length]);

  // Continuous reveal loop — reads text/streaming through refs, so the interval
  // never needs restarting
  useEffect(() => {
    const id = window.setInterval(() => {
      setRevealed((cur) => {
        const total = textRef.current.length;
        if (cur >= total) return cur;
        const backlog = total - cur;
        // Streaming: catch up within ~24 ticks (~400ms), at least 2 chars/tick (~125 c/s)
        // Done: flush trong ~6 ticks
        const step = streamingRef.current
          ? Math.max(2, Math.ceil(backlog / 24))
          : Math.max(20, Math.ceil(backlog / 6));
        return Math.min(total, cur + step);
      });
    }, 16);
    return () => clearInterval(id);
  }, []);

  return text.slice(0, revealed);
}
