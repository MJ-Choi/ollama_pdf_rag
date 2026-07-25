import { ollamaChat } from "@/lib/ai/provider";
import { getDefaultChatModel } from "@/lib/ai/models";
import { createUIMessageStream, createUIMessageStreamResponse } from "ai";
import { after } from "next/server";
import {
  type ResumableStreamContext,
  createResumableStreamContext,
} from "resumable-stream";
import { auth } from "@/app/(auth)/auth";
import {
  deleteChatById,
  getChatById,
  saveChat,
  saveMessages,
} from "@/lib/db/queries";
import { ChatSDKError } from "@/lib/errors";
import { generateUUID } from "@/lib/utils";

// Only enforced on Vercel deployments (no effect on local `next dev`). A
// full-document translation walks the PDF page by page on the backend and
// can take well past 60s — raised so a Vercel deployment doesn't kill the
// function mid-translation. Actual ceiling depends on the Vercel plan tier.
export const maxDuration = 1800;

// Backs "resume an in-progress response after a page refresh" (see
// [id]/stream/route.ts), via Redis (REDIS_URL). This app doesn't set up a
// real Redis instance for local dev (.env.local's REDIS_URL is an unset
// placeholder) — any failure to construct the context is treated as
// "feature unavailable" and falls back to null rather than throwing, same
// as the upstream template's intent when Redis isn't configured.
let globalStreamContext: ResumableStreamContext | null = null;
let loggedStreamContextUnavailable = false;

export function getStreamContext() {
  if (!globalStreamContext) {
    try {
      globalStreamContext = createResumableStreamContext({
        waitUntil: after,
      });
    } catch {
      if (!loggedStreamContextUnavailable) {
        console.log(
          " > Resumable streams are disabled (no working REDIS_URL)"
        );
        loggedStreamContextUnavailable = true;
      }
    }
  }
  return globalStreamContext;
}

interface MessagePart {
  type: string;
  text?: string;
  url?: string;
  name?: string;
  mediaType?: string;
  data?: unknown;
}

interface PostRequestBody {
  id: string;
  message: {
    id?: string;
    role: string;
    parts: MessagePart[];
  };
  selectedChatModel?: string;
  selectedVisibilityType?: string;
  selectedPdfIds?: string[]; // PDFs selected by user
}

// Simple question classifier - checks if question likely needs document
// context. Only reached when the user selected ZERO PDFs (see
// noPdfsButNeedsContext below) — the point is purely to decide whether to
// show a "please select a PDF" warning instead of just answering as
// general chat. Precision is prioritized over recall: only strong,
// unambiguous document-reference terms are matched, since a false
// positive here blocks an otherwise-normal conversation with an
// unnecessary warning, while a false negative just falls through to
// general chat — a much smaller UX cost. (Previously this list included
// generic words like "this", "explain", "summary", "content", "text",
// "describe", "about the", "in the", "from the", which matched huge
// swaths of ordinary conversation.)
const ENGLISH_DOCUMENT_KEYWORDS = [
  "document", "pdf", "file", "page", "pages", "section", "chapter",
  "uploaded", "attachment", "attached",
];
// Word-boundary matching (not substring) — the old `.includes()` check
// matched "file" inside "profile" and "text" inside "context"/"textbook".
const ENGLISH_DOCUMENT_KEYWORDS_RE = new RegExp(
  `\\b(${ENGLISH_DOCUMENT_KEYWORDS.join("|")})\\b`,
  "i"
);
// The app's primary users ask questions in Korean (see root CLAUDE.md) —
// the English-only keyword list above never matched Korean questions at
// all, so this safety net effectively never fired for real usage.
const KOREAN_DOCUMENT_KEYWORDS = [
  "문서", "파일", "피디에프", "페이지", "업로드", "첨부", "도안",
];

function needsDocumentContext(question: string): boolean {
  if (ENGLISH_DOCUMENT_KEYWORDS_RE.test(question)) {
    return true;
  }
  return KOREAN_DOCUMENT_KEYWORDS.some((keyword) => question.includes(keyword));
}

export async function POST(request: Request) {
  try {
    const body: PostRequestBody = await request.json();
    const chatId = body.id;

    const session = await auth();
    if (!session?.user?.id) {
      return new Response(JSON.stringify({ error: "Unauthorized" }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      });
    }
    const userId = session.user.id;

    const selectedChatModel =
      body.selectedChatModel || (await getDefaultChatModel());
    console.log("Using model:", selectedChatModel);
    console.log("Received message:", JSON.stringify(body.message, null, 2));

    const textPart = body.message.parts.find((p) => p.type === "text");
    const textContent = textPart?.text || "";

    if (!textContent) {
      return new Response(
        JSON.stringify({ error: "No text content found in message" }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    const chat = await getChatById({ id: chatId });

    if (!chat) {
      const title =
        textContent.slice(0, 100) + (textContent.length > 100 ? "..." : "");
      await saveChat({
        id: chatId,
        userId,
        title,
        visibility:
          (body.selectedVisibilityType as "private" | "public") || "private",
      });
      console.log("Created new chat:", chatId);
    }

    const selectedPdfIds = body.selectedPdfIds || [];
    console.log("Selected PDF IDs from frontend:", selectedPdfIds);
    const questionNeedsContext = needsDocumentContext(textContent);
    console.log("Question needs document context:", questionNeedsContext);

    const userMessageId = body.message.id || generateUUID();
    await saveMessages({
      messages: [{
        id: userMessageId,
        chatId,
        role: "user",
        content: textContent, // Legacy field
        parts: JSON.stringify(body.message.parts),
        createdAt: new Date(),
      }],
    });
    console.log("Saved user message:", userMessageId);

    const useRAG = selectedPdfIds.length > 0;
    const noPdfsButNeedsContext =
      selectedPdfIds.length === 0 && questionNeedsContext;
    console.log("Mode:", useRAG ? "RAG" : "General Chat");
    console.log("No PDFs but needs context:", noPdfsButNeedsContext);

    // Only the current message is sent — there's no conversation history in
    // this call. RAGService.query_multi_pdf() is itself stateless/single-turn
    // (see root CLAUDE.md), so passing prior turns here wouldn't do anything yet.
    const messages = [{ role: body.message.role, content: textContent }];

    let result;
    let formattedResponse: string;
    let reasoningSteps: string[] = [];

    if (noPdfsButNeedsContext) {
      formattedResponse = `⚠️ **No documents selected**

It looks like your question might be about a document, but you haven't selected any PDFs to search.

**To get answers from your documents:**
1. Look at the sidebar on the left
2. Check the boxes next to the PDFs you want to use
3. Then ask your question again

**If you just want to chat without documents**, feel free to ask general questions and I'll respond based on my knowledge!

---

*Your question was: "${textContent}"*`;
      console.log("Warning: Question seems to need document context but no PDFs selected");

      result = {
        answer: formattedResponse,
        sources: [],
        metadata: { reasoning_steps: ["⚠️ No PDFs selected for document query"] }
      };
    } else if (useRAG) {
      console.log("Sending to backend:", {
        question: textContent,
        model: selectedChatModel,
        pdfIds: selectedPdfIds,
      });
      console.log("Calling ollamaChat with RAG...");
      result = await ollamaChat(messages, selectedChatModel, selectedPdfIds);
      // Sources are rendered separately by SourcesPanel (clickable, opens
      // the PDF viewer) via the data-sources part below — not appended as
      // flat text here, so the answer text doesn't duplicate them.
      formattedResponse = result.answer;
      reasoningSteps = result.metadata.reasoning_steps || [];
    } else {
      console.log("Calling ollamaChat for general chat (no RAG)...");
      result = await ollamaChat(messages, selectedChatModel, undefined);
      formattedResponse = result.answer;
      reasoningSteps = result.metadata.reasoning_steps || [];
    }
    console.log("Received result from backend:", {
      answerLength: result.answer.length,
      sourcesCount: result.sources?.length || 0,
      hasReasoningSteps: !!result.metadata?.reasoning_steps,
      reasoningStepsCount: result.metadata?.reasoning_steps?.length || 0,
    });
    console.log("Reasoning steps:", reasoningSteps);
    console.log("Formatted response length:", formattedResponse.length);
    console.log("First 200 chars of response:", formattedResponse.substring(0, 200));

    const assistantMessageId = generateUUID();
    const assistantParts: MessagePart[] = [{ type: "text", text: formattedResponse }];

    // Persisted alongside the text so SourcesPanel (clickable citations
    // that open the PDF viewer) still renders after a reload — the live
    // stream's data-sources part is transient and isn't saved on its own.
    if (result.sources && result.sources.length > 0) {
      assistantParts.unshift({
        type: "data-sources",
        data: result.sources,
      });
    }

    if (reasoningSteps.length > 0) {
      assistantParts.unshift({
        type: "reasoning",
        text: reasoningSteps.join("\n"),
      });
    }

    await saveMessages({
      messages: [{
        id: assistantMessageId,
        chatId,
        role: "assistant",
        content: formattedResponse, // Legacy field
        parts: JSON.stringify(assistantParts),
        createdAt: new Date(),
      }],
    });
    console.log("Saved assistant message:", assistantMessageId);

    console.log("Creating UI message stream...");
    const textId = "text-1";

    const messageStream = createUIMessageStream({
      execute: async ({ writer }) => {
        writer.write({
          type: "message-metadata",
          messageMetadata: { createdAt: new Date().toISOString() },
        });

        // Streamed progressively (with small delays below) purely for a
        // streaming visual effect — the full text is already known at this
        // point, unlike a real token-by-token model stream.
        if (reasoningSteps && reasoningSteps.length > 0) {
          console.log("Writing reasoning steps progressively:", reasoningSteps.length);
          const reasoningId = "reasoning-1";

          writer.write({ type: "reasoning-start", id: reasoningId });

          for (const step of reasoningSteps) {
            writer.write({
              type: "reasoning-delta",
              id: reasoningId,
              delta: step + "\n",
            });
            await new Promise(resolve => setTimeout(resolve, 150));
          }

          writer.write({ type: "reasoning-end", id: reasoningId });
        }

        const sources = result.sources || [];
        if (sources.length > 0) {
          console.log("Writing sources:", sources.length);
          writer.write({
            type: "data-sources",
            data: sources,
          });
        }

        console.log("Writing text content progressively...");
        writer.write({ type: "text-start", id: textId });

        const words = formattedResponse.split(" ");
        for (let i = 0; i < words.length; i++) {
          writer.write({
            type: "text-delta",
            id: textId,
            delta: words[i] + " ",
          });
          if (i % 3 === 0) {
            await new Promise(resolve => setTimeout(resolve, 30));
          }
        }

        writer.write({ type: "text-end", id: textId });
        console.log("Stream write complete");
      },
    });

    return createUIMessageStreamResponse({ stream: messageStream });
  } catch (error) {
    console.error("Chat error:", error);

    // Extract error message
    const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
    const errorText = `❌ **Error**: ${errorMessage}\n\nPlease check:\n- Model is installed and running\n- PDF documents are uploaded\n- Backend service is accessible`;

    // Return error as a UI message stream
    const errorStream = createUIMessageStream({
      execute: async ({ writer }) => {
        writer.write({
          type: "message-metadata",
          messageMetadata: { createdAt: new Date().toISOString() },
        });

        // Write error chunk
        writer.write({
          type: "error",
          errorText: errorMessage,
        });

        // Write error text progressively
        const textId = "error-text-1";
        writer.write({ type: "text-start", id: textId });

        const words = errorText.split(" ");
        for (let i = 0; i < words.length; i++) {
          writer.write({
            type: "text-delta",
            id: textId,
            delta: words[i] + " ",
          });
          // Delay every few words
          if (i % 3 === 0) {
            await new Promise(resolve => setTimeout(resolve, 30));
          }
        }

        writer.write({ type: "text-end", id: textId });
      },
    });

    return createUIMessageStreamResponse({ stream: errorStream });
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get("id");

  console.log("DELETE /api/chat requested for id:", id);

  if (!id) {
    console.log("DELETE /api/chat rejected: no id parameter");
    return new ChatSDKError(
      "bad_request:api",
      "Parameter id is required."
    ).toResponse();
  }

  const session = await auth();

  if (!session?.user?.id) {
    console.log("DELETE /api/chat rejected: no authenticated session");
    return new ChatSDKError("unauthorized:chat").toResponse();
  }

  try {
    const chat = await getChatById({ id });

    if (!chat) {
      console.log(`DELETE /api/chat rejected: chat ${id} not found`);
      return new ChatSDKError("not_found:chat").toResponse();
    }

    if (chat.userId !== session.user.id) {
      console.log(
        `DELETE /api/chat rejected: chat ${id} owned by ${chat.userId}, requested by ${session.user.id}`
      );
      return new ChatSDKError("forbidden:chat").toResponse();
    }

    const deletedChat = await deleteChatById({ id });
    console.log(`DELETE /api/chat succeeded for id: ${id}`);

    return Response.json(deletedChat, { status: 200 });
  } catch (error) {
    console.error(`DELETE /api/chat failed unexpectedly for id: ${id}`, error);

    if (error instanceof ChatSDKError) {
      return error.toResponse();
    }

    return new ChatSDKError("bad_request:database", "Failed to delete chat").toResponse();
  }
}
