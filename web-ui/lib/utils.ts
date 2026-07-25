import type {
  CoreAssistantMessage,
  CoreToolMessage,
  UIMessage,
  UIMessagePart,
} from 'ai';
import { type ClassValue, clsx } from 'clsx';
import { formatISO } from 'date-fns';
import { twMerge } from 'tailwind-merge';
import type { DBMessage, Document } from '@/lib/db/schema';
import { ChatSDKError, type ErrorCode } from './errors';
import type { ChatMessage, ChatTools, CustomUIDataTypes } from './types';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const fetcher = async (url: string) => {
  const response = await fetch(url);

  if (!response.ok) {
    const { code, cause } = await response.json();
    throw new ChatSDKError(code as ErrorCode, cause);
  }

  return response.json();
};

export async function fetchWithErrorHandlers(
  input: RequestInfo | URL,
  init?: RequestInit,
) {
  try {
    const response = await fetch(input, init);

    if (!response.ok) {
      const { code, cause } = await response.json();
      throw new ChatSDKError(code as ErrorCode, cause);
    }

    return response;
  } catch (error: unknown) {
    if (typeof navigator !== 'undefined' && !navigator.onLine) {
      throw new ChatSDKError('offline:chat');
    }

    throw error;
  }
}

export function getLocalStorage(key: string) {
  if (typeof window !== 'undefined') {
    return JSON.parse(localStorage.getItem(key) || '[]');
  }
  return [];
}

export function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

type ResponseMessageWithoutId = CoreToolMessage | CoreAssistantMessage;
type ResponseMessage = ResponseMessageWithoutId & { id: string };

export function getMostRecentUserMessage(messages: UIMessage[]) {
  const userMessages = messages.filter((message) => message.role === 'user');
  return userMessages.at(-1);
}

export function getDocumentTimestampByIndex(
  documents: Document[],
  index: number,
) {
  if (!documents) { return new Date(); }
  if (index > documents.length) { return new Date(); }

  return documents[index].createdAt;
}

export function getTrailingMessageId({
  messages,
}: {
  messages: ResponseMessage[];
}): string | null {
  const trailingMessage = messages.at(-1);

  if (!trailingMessage) { return null; }

  return trailingMessage.id;
}

export function sanitizeText(text: string) {
  return (
    text
      .replace('<has_function_call>', '')
      // Strip leading spaces from every line before markdown rendering.
      // CommonMark treats 4+ leading spaces as an indented code block —
      // meaningless in this app's line-by-line OCR/translation output,
      // where such spacing is just an OCR/formatting artifact — and mixing
      // a code block into what the surrounding markdown treats as a single
      // paragraph (via remark-breaks) produced invalid HTML (a <div> block
      // nested inside a <p>), causing a hydration error.
      .replace(/^[ \t]+/gm, '')
      // Escape stray backticks. A single backtick opens a markdown inline
      // code span that runs until the NEXT backtick anywhere in the text —
      // with remark-breaks keeping everything in one paragraph, an OCR
      // artifact backtick several lines away from an unrelated one can
      // wrap a huge multi-line stretch as "code", which Streamdown then
      // promotes to a block-level element — the same invalid <div>-in-<p>
      // nesting as the indented-code-block case above. This content is
      // OCR/translation output, never intentional markdown, so backticks
      // are always literal here.
      .replace(/`/g, '\\`')
  );
}

export function convertToUIMessages(messages: DBMessage[]): ChatMessage[] {
  return messages.map((message) => {
    // Parse parts from JSON string if needed, fallback to content field
    let parts = message.parts;

    if (!parts && (message as any).content) {
      // Fallback to content field for legacy messages
      parts = [{ type: 'text', text: (message as any).content }];
    } else if (typeof parts === 'string') {
      try {
        parts = JSON.parse(parts);
      } catch {
        // If parsing fails, wrap as text part
        parts = [{ type: 'text', text: parts }];
      }
    }

    // Ensure parts is always an array
    if (!parts || !Array.isArray(parts)) {
      parts = [];
    }

    return {
      id: message.id,
      role: message.role as 'user' | 'assistant' | 'system',
      parts: parts as UIMessagePart<CustomUIDataTypes, ChatTools>[],
      metadata: {
        createdAt: formatISO(message.createdAt),
      },
    };
  });
}

export function getTextFromMessage(message: ChatMessage | UIMessage): string {
  return message.parts
    .filter((part) => part.type === 'text')
    .map((part) => (part as { type: 'text'; text: string}).text)
    .join('');
}
