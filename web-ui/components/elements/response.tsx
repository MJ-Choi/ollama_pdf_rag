"use client";

import { type ComponentProps, memo } from "react";
import remarkBreaks from "remark-breaks";
import { defaultRemarkPlugins, Streamdown } from "streamdown";
import { cn } from "@/lib/utils";

type ResponseProps = ComponentProps<typeof Streamdown>;

// Backend answers (raw OCR'd page text, translations, etc.) use single "\n"
// between lines. Standard CommonMark/GFM treats a lone newline as a soft
// break rendered as a space, not a visible line break, so without
// remark-breaks all that line structure was getting silently collapsed.
// Setting `remarkPlugins` overrides Streamdown's own defaults entirely
// (it doesn't merge), so its defaults are spread back in alongside it.
const remarkPlugins = [...Object.values(defaultRemarkPlugins), remarkBreaks];

export const Response = memo(
  ({ className, ...props }: ResponseProps) => (
    <Streamdown
      className={cn(
        "size-full [&>*:first-child]:mt-0 [&>*:last-child]:mb-0 [&_code]:whitespace-pre-wrap [&_code]:break-words [&_pre]:max-w-full [&_pre]:overflow-x-auto",
        className
      )}
      remarkPlugins={remarkPlugins}
      {...props}
    />
  ),
  (prevProps, nextProps) => prevProps.children === nextProps.children
);

Response.displayName = "Response";
