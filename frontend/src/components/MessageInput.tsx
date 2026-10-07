import React, { useLayoutEffect, useRef, useState, KeyboardEvent } from 'react';
import { ModeChip, type Mode } from './ModeToggle';
import { ModelChip, type ModelOption } from './ModelChip';
import { ThinkChip } from './ThinkChip';
import { ContextChip, type ContextLength } from './ContextChip';
import { UsageChip } from './UsageChip';

interface MessageInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  mode: Mode;
  onModeChange: (m: Mode) => void;
  model: string;
  onModelChange: (m: string) => void;
  models: readonly (string | ModelOption)[];
  think: boolean;
  /** Set when the chosen model has no reasoning channel. */
  thinkDisabledReason?: string;
  onThinkChange: (v: boolean) => void;
  contextLength: ContextLength;
  /** Set when the chosen model ignores num_ctx (hosted models do). */
  contextDisabledReason?: string;
  onContextLengthChange: (v: ContextLength) => void;
}

const MAX_HEIGHT = 240;

export function MessageInput({
  onSend,
  disabled = false,
  mode,
  onModeChange,
  model,
  onModelChange,
  models,
  think,
  thinkDisabledReason,
  onThinkChange,
  contextLength,
  contextDisabledReason,
  onContextLengthChange,
}: MessageInputProps) {
  const [value, setValue] = useState('');
  const taRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow: reset to auto, then snap to scrollHeight (capped)
  useLayoutEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const next = Math.min(el.scrollHeight, MAX_HEIGHT);
    el.style.height = next + 'px';
    el.style.overflowY = el.scrollHeight > MAX_HEIGHT ? 'auto' : 'hidden';
  }, [value]);

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const canSend = !disabled && value.trim().length > 0;

  return (
    <div className="px-4 pb-4 pt-2 bg-white">
      <div className="max-w-3xl mx-auto">
        <div
          className="bg-gray-50 border border-gray-200 rounded-2xl p-3
                     focus-within:border-gray-300 focus-within:bg-white
                     transition-colors shadow-sm"
        >
          {/* Auto-growing textarea */}
          <textarea
            ref={taRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            rows={1}
            placeholder="Ask a 3GPP question..."
            className="w-full resize-none bg-transparent border-0 outline-none
                       text-[15px] leading-relaxed text-gray-900 placeholder-gray-400
                       disabled:opacity-50"
            style={{ minHeight: '24px', maxHeight: `${MAX_HEIGHT}px` }}
          />

          {/* Bottom row: mode + model chips on the left, send button on the right */}
          <div className="flex items-center gap-2 mt-2 flex-wrap">
            <ModeChip value={mode} onChange={onModeChange} />
            <ModelChip models={models} value={model} onChange={onModelChange} />
            <ThinkChip value={think} onChange={onThinkChange}
                       disabledReason={thinkDisabledReason} />
            <UsageChip />
            <ContextChip value={contextLength} onChange={onContextLengthChange}
                         disabledReason={contextDisabledReason} />
            <div className="flex-1" />
            <button
              type="button"
              onClick={handleSend}
              disabled={!canSend}
              aria-label="Send message"
              className={`w-8 h-8 rounded-full flex items-center justify-center transition-all
                          ${
                            canSend
                              ? 'bg-gray-900 text-white hover:bg-gray-800'
                              : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                          }`}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M12 19V5M5 12l7-7 7 7" />
              </svg>
            </button>
          </div>
        </div>
        <div className="text-[11px] text-gray-400 text-center mt-2">
          Enter to send · Shift+Enter for newline
        </div>
      </div>
    </div>
  );
}
