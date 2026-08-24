local function stringify(block)
  return pandoc.utils.stringify(block)
end

local listing_languages = {
  bash = "bash", sh = "bash", shell = "bash", console = "bash",
  python = "Python", py = "Python", java = "Java", javascript = "",
  js = "", sql = "SQL", c = "C", cpp = "C++", json = "",
  yaml = "", yml = "", xml = "XML", html = "HTML", css = "",
  dockerfile = "", terraform = "", hcl = ""
}

local toc_exclusions = {
  ["Chapter Summary"] = true,
  ["Key Terms"] = true,
  ["Review Questions"] = true,
  ["Hands-On Exercises"] = true,
  ["Further Reading"] = true,
  ["References"] = true,
  ["Known Issues"] = true,
  ["Development"] = true
}

function CodeBlock(block)
  if not FORMAT:match("latex") then return nil end
  local language = listing_languages[block.classes[1] or ""]
  local option = language and language ~= "" and ("language=" .. language) or ""
  return pandoc.RawBlock("latex", "\\begin{SETerminal}[" .. option .. "]\n" .. block.text .. "\n\\end{SETerminal}")
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
  local chapter_title = ""
  local section_count = 0
  while i <= #blocks do
    local block = blocks[i]
    if block.t == "Header" and block.level == 1 then
      chapter_title = stringify(block)
      section_count = 0
    end
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
    elseif block.t == "Header" and block.level == 2 then
      section_count = section_count + 1
      local include_in_toc = chapter_title ~= "Glossary" and section_count <= 7 and not toc_exclusions[stringify(block)]
      local depth = include_in_toc and "1" or "0"
      out:insert(pandoc.RawBlock("latex", "\\addtocontents{toc}{\\protect\\setcounter{tocdepth}{" .. depth .. "}}"))
      out:insert(block)
      i = i + 1
    else
      out:insert(block)
      i = i + 1
    end
  end
  return pandoc.Pandoc(out, doc.meta)
end
