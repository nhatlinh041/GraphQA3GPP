import { useState, useCallback } from 'react';

export interface PipelineStage {
  stage: string;
  data: unknown;
  timestamp: number;
}

export interface Source {
  spec_id: string;
  section: string;
  chunk_id: string;
  // Full chunk content cited in the answer (added to the SSE payload by orchestrator.py)
  content?: string;
  score?: number;
}

export interface SSEState {
  stages: PipelineStage[];
  thinking: string;
  answer: string;
  sources: Source[];
  loading: boolean;
  error: string | null;
}

const INITIAL_STATE: SSEState = {
  stages: [],
  thinking: '',
  answer: '',
  sources: [],
  loading: false,
  error: null,
};

export function useSSE(apiUrl: string) {
  const [state, setState] = useState<SSEState>(INITIAL_STATE);

  const send = useCallback(
    async (body: {
      question: string;
      mode?: string;
      model?: string;
      think?: boolean;
      context_length?: number;
    }) => {
      setState({ ...INITIAL_STATE, loading: true });

      try {
        const response = await fetch(apiUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });

        if (!response.ok || !response.body) {
          throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() ?? '';

          for (const line of lines) {
            if (!line.startsWith('data:')) continue;
            try {
              const event = JSON.parse(line.slice(5).trim());
              const { stage, data } = event;

              setState((prev) => {
                const newStage: PipelineStage = { stage, data, timestamp: Date.now() };

                if (stage === 'thinking') {
                  // Thinking tokens stream in BEFORE answer for reasoning models;
                  // we accumulate them in their own buffer so the UI can render them
                  // separately from the final answer.
                  return {
                    ...prev,
                    thinking: prev.thinking + (data as string),
                    stages: [...prev.stages, newStage],
                  };
                }
                if (stage === 'answer') {
                  return {
                    ...prev,
                    answer: prev.answer + (data as string),
                    stages: [...prev.stages, newStage],
                  };
                }
                if (stage === 'sources') {
                  return {
                    ...prev,
                    sources: data as Source[],
                    loading: false,
                    stages: [...prev.stages, newStage],
                  };
                }
                return { ...prev, stages: [...prev.stages, newStage] };
              });
            } catch {
              // skip malformed SSE lines
            }
          }
        }
      } catch (err) {
        setState((prev) => ({ ...prev, loading: false, error: String(err) }));
      } finally {
        setState((prev) => ({ ...prev, loading: false }));
      }
    },
    [apiUrl],
  );

  const reset = useCallback(() => setState(INITIAL_STATE), []);

  return { ...state, send, reset };
}
