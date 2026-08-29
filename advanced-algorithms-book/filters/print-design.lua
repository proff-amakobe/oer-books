-- Phase 6 print-only structural styling. Canonical manuscript remains unchanged.

local function stringify(block)
  return pandoc.utils.stringify(block)
end

local function latex_escape(value)
  return value:gsub("\\", "\\textbackslash{}")
              :gsub("([%%#&{}_])", "\\%1")
              :gsub("%$", "\\$")
end

local function first_strong(block)
  if block.t ~= "Para" and block.t ~= "Plain" then return nil end
  for _, inline in ipairs(block.content) do
    if inline.t == "Strong" then return stringify(inline) end
    if inline.t == "RawInline" then goto continue end
    if inline.t ~= "Space" then break end
    ::continue::
  end
  return nil
end

function Para(el)
  if FORMAT:match("latex") then
    table.insert(el.content, 1, pandoc.RawInline("latex", "\\color{ACAPrintInk}"))
  end
  return el
end

local function callout_kind(block)
  local label = first_strong(block)
  if not label then return nil end
  local low = label:lower():gsub(":$", "")
  if low:match("^theorem") then return "ACATheorem", "THEOREM" end
  if low:match("^proof sketch") then return "ACAProof", "Proof Sketch" end
  if low:match("^proof") or low:match("^correctness") then return "ACAProof", "Proof" end
  if low:match("^intuition") or low:match("^key insight") then return "ACAInsight", "INTUITION" end
  if low:match("^complexity") or low:match("^time complexity") or low:match("^space complexity") then return "ACAComplexity", "COMPLEXITY" end
  if low:match("^common pitfall") or low:match("^pitfall") or low:match("^warning") then return "ACAPitfall", "COMMON PITFALL" end
  if low:match("^implementation note") then return "ACAComplexity", "IMPLEMENTATION NOTE" end
  if low:match("^real%-world connection") then return "ACAInsight", "REAL-WORLD CONNECTION" end
  return nil
end

function Header(el)
  if FORMAT:match("latex") and el.level == 1 then
    local title = stringify(el)
    local number, chapter_title = title:match("^Chapter%s+(%d+):%s*(.+)$")
    if number then
      local display_number = tonumber(number) < 10 and ("0" .. number) or number
      return pandoc.RawBlock("latex", "\\ACAChapter{" .. display_number .. "}{" .. number .. "}{" .. latex_escape(chapter_title) .. "}")
    end
    local part_number, part_title = title:match("^Part%s+([IVX]+):%s*(.+)$")
    if part_title then
      return pandoc.RawBlock("latex", "\\ACAPart{" .. part_number .. "}{" .. latex_escape(part_title) .. "}")
    end
  end
end

function Pandoc(doc)
  if not FORMAT:match("latex") then return doc end
  local out = {}
  local i = 1
  while i <= #doc.blocks do
    local block = doc.blocks[i]
    local marker = (block.t == "Para" or block.t == "Plain") and stringify(block):lower() or ""
    if marker:match("^(python|bash|java|javascript|yaml|json)$") then
      i = i + 1
    elseif block.t == "Header" and block.level >= 2 then
      local title = stringify(block):lower()
      if title:match("learning objectives") or title:match("what you.?ll learn") then
        local content = {}
        i = i + 1
        while i <= #doc.blocks do
          local next_block = doc.blocks[i]
          if next_block.t == "Header" and next_block.level <= block.level then break end
          table.insert(content, next_block)
          i = i + 1
        end
        table.insert(out, pandoc.RawBlock("latex", "\\begin{ACAObjectives}"))
        for _, item in ipairs(content) do table.insert(out, item) end
        table.insert(out, pandoc.RawBlock("latex", "\\end{ACAObjectives}"))
      else
        table.insert(out, block)
        i = i + 1
      end
    else
      local env, label = callout_kind(block)
      if env then
        table.insert(out, pandoc.RawBlock("latex", "\\begin{" .. env .. "}{" .. label .. "}"))
        table.insert(out, block)
        table.insert(out, pandoc.RawBlock("latex", "\\end{" .. env .. "}"))
      else
        table.insert(out, block)
      end
      i = i + 1
    end
  end
  doc.blocks = out
  return doc
end
