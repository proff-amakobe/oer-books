-- R1 print-only glyph safety. This preserves Pandoc CodeBlock semantics and
-- changes neither environment nor language class; HTML/EPUB source is untouched.
local replacements = {
  ["─"] = "-", ["═"] = "=", ["│"] = "|", ["┃"] = "|", ["┊"] = "|",
  ["┌"] = "+", ["┐"] = "+", ["└"] = "+", ["┘"] = "+",
  ["├"] = "+", ["┤"] = "+", ["┬"] = "+", ["┴"] = "+", ["┼"] = "+",
  ["╭"] = "+", ["╮"] = "+", ["╯"] = "+", ["╰"] = "+",
  ["╔"] = "+", ["╗"] = "+", ["╚"] = "+", ["╝"] = "+",
  ["╠"] = "+", ["╣"] = "+", ["╦"] = "+", ["╩"] = "+", ["╬"] = "+",
  ["╱"] = "/", ["╲"] = "\\",
  ["→"] = "->", ["←"] = "<-", ["►"] = ">", ["▶"] = ">",
  ["▷"] = ">", ["◄"] = "<", ["◀"] = "<", ["▼"] = "v",
  ["▲"] = "^", ["△"] = "^", ["●"] = "o", ["○"] = "o",
  ["◉"] = "o", ["◐"] = "o", ["◇"] = "<>", ["◆"] = "<>",
  ["□"] = "[ ]", ["▢"] = "[ ]", ["█"] = "#", ["░"] = ".",
  ["☰"] = "menu", ["⚠"] = "Warning", ["ℹ"] = "Info",
  ["📘"] = "Guide", ["🔔"] = "Alert", ["👤"] = "User",
  ["👁"] = "View", ["🚧"] = "In progress", ["🏝"] = "Island",
  ["🐛"] = "Bug", ["🔥"] = "High priority", ["📝"] = "Story",
  ["🚫"] = "Blocked", ["✕"] = "Close"
}

function CodeBlock(block)
  if not FORMAT:match("latex") then return nil end
  for source, replacement in pairs(replacements) do
    block.text = block.text:gsub(source, replacement)
  end
  return block
end
