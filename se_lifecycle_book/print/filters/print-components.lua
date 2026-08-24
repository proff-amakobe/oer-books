local function stringify(block)
  return pandoc.utils.stringify(block)
end

function Pandoc(doc)
  if not FORMAT:match("latex") then return doc end
  -- The designed title pages below replace Pandoc's automatic \maketitle.
  doc.meta.title = nil
  doc.meta.subtitle = nil
  doc.meta.author = nil
  doc.meta.date = nil
  -- Quarto combines every book input into one Pandoc document. Replace the
  -- landing-page blocks before the first chapter H1 with print front matter.
  local handle = assert(io.open("print/frontmatter.qmd", "r"))
  local text = handle:read("*a")
  handle:close()
  local frontmatter = pandoc.read(text, "markdown").blocks
  local manuscript = pandoc.List()
  local found_first_chapter = false
  for _, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.level == 1 and
       stringify(block) == "Introduction to Software Engineering" then
      found_first_chapter = true
    end
    if found_first_chapter then manuscript:insert(block) end
  end
  frontmatter:extend(manuscript)
  doc.blocks = frontmatter
  local out = pandoc.List()
  local blocks = doc.blocks
  local i = 1
  while i <= #blocks do
    local block = blocks[i]
    if block.t == "Header" and block.level == 2 and stringify(block) == "Learning Objectives" then
      local content = pandoc.List()
      i = i + 1
      while i <= #blocks and not (blocks[i].t == "Header" and blocks[i].level <= 2) do
        if blocks[i].t ~= "HorizontalRule" then content:insert(blocks[i]) end
        i = i + 1
      end
      out:insert(pandoc.RawBlock("latex", "\\begin{SEObjectives}"))
      out:extend(content)
      out:insert(pandoc.RawBlock("latex", "\\end{SEObjectives}"))
    else
      out:insert(block)
      i = i + 1
    end
  end
  return pandoc.Pandoc(out, doc.meta)
end
