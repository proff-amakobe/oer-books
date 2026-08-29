-- Quarto 1.7 disables Pandoc's native section numbering for book PDF/EPUB.
-- Generate labels from the normalized semantic heading tree for those formats.

local chapter = 0
local sections = {0, 0, 0, 0, 0}
local sectioning_started = false

local function has_class(element, name)
  for _, class in ipairs(element.classes) do
    if class == name then return true end
  end
  return false
end

local function prefix(element, label, class_name)
  table.insert(element.content, 1, pandoc.Space())
  table.insert(element.content, 1, pandoc.Span({pandoc.Str(label)}, pandoc.Attr("", {class_name})))
  return element
end

function Div(element)
  if has_class(element, "web-only") then
    element.content = {}
    return element
  end
end

function Header(element)
  if has_class(element, "unnumbered") then return element end
  if element.level == 1 and pandoc.utils.stringify(element):match("^Part [IVX]+:") then
    return element
  end

  if element.level == 1 then
    chapter = chapter + 1
    sections = {0, 0, 0, 0, 0}
    sectioning_started = false
    local label = tostring(chapter)
    if FORMAT:match("latex") then label = "Chapter " .. label .. ":" end
    return prefix(element, label, "chapter-number")
  end

  if chapter == 0 then return element end
  local index = element.level - 1
  if index == 1 then sectioning_started = true end
  if not sectioning_started then return element end

  sections[index] = sections[index] + 1
  for i = index + 1, #sections do sections[i] = 0 end
  local labels = {tostring(chapter)}
  for i = 1, index do table.insert(labels, tostring(sections[i])) end
  return prefix(element, table.concat(labels, "."), "header-section-number")
end
