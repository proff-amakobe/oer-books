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

-- Figure policy for print: every instructional image gets one stable Quarto
-- label and a content-driven width. The LaTeX preamble supplies the final hard
-- maximum, so a source can never escape the text block or page-height limit.
local compact_figures = {
  ch1_fig1_waterfall_model = true,
  ch1_fig5_repo_structure = true,
  ch4_fig17_singleton_structure = true,
  ch4_fig18_factory_hierarchy = true,
  ch4_fig19_builder_pattern = true,
  ch4_fig20_adapter_structure = true,
  ch4_fig22_facade_structure = true,
  ch4_fig23_template_method = true
}

local complex_figures = {
  git_merge_before_after = true,
  git_three_way_merge = true,
  git_rebase_diagram = true,
  trunk_based_development = true,
  testing_pyramid = true,
  ch3_fig3_usecase_example = true,
  ch3_fig5_activity_example = true,
  ch3_fig7_sequence_example = true,
  ch3_fig10_class_example = true,
  ch3_fig11_activity_algorithm = true,
  ch3_fig14_diagram_selection = true,
  ch4_fig3_microservices = true,
  ch4_fig10_architecture_comparison = true,
  ch4_fig24_four_plus_one = true,
  ch5_fig4_nielsens_heuristics = true,
  ch5_fig10_design_system = true,
  ch5_fig11_journey_map = true,
  ch5_fig13_wireframe_examples = true,
  ch5_fig14_research_methods = true,
  ch6_fig2_scrum_framework = true,
  ch6_fig3_kanban_board = true
}

function Image(image)
  if not FORMAT:match("latex") then return nil end
  local stem = image.src:match("([^/]+)%.%w+$")
  if not stem or stem == "cover" then return nil end
  stem = stem:gsub("[^%w_-]", "-")
  if image.identifier == "" then image.identifier = "fig-" .. stem end
  if not image.attributes.width then
    if compact_figures[stem] then
      image.attributes.width = "62%"
    elseif complex_figures[stem] then
      image.attributes.width = "90%"
    else
      image.attributes.width = "74%"
    end
  end
  return image
end

function CodeBlock(block)
  if not FORMAT:match("latex") then return nil end
  -- Latin Modern Mono does not cover the Unicode box-drawing block. Preserve
  -- the diagrams' technical meaning in print with dependable ASCII glyphs;
  -- HTML and EPUB retain the canonical Unicode source unchanged.
  local print_text = block.text
  local replacements = {
    ["─"] = "-", ["═"] = "=", ["│"] = "|", ["┃"] = "|", ["┊"] = "|",
    ["┌"] = "+", ["┐"] = "+", ["└"] = "+", ["┘"] = "+",
    ["├"] = "+", ["┤"] = "+", ["┬"] = "+", ["┴"] = "+", ["┼"] = "+",
    ["╭"] = "+", ["╮"] = "+", ["╯"] = "+", ["╰"] = "+",
    ["╱"] = "/", ["╲"] = "\\", ["→"] = "->", ["←"] = "<-",
    ["✓"] = "PASS", ["✗"] = "FAIL"
  }
  for source, replacement in pairs(replacements) do
    print_text = print_text:gsub(source, replacement)
  end
  local language = listing_languages[block.classes[1] or ""]
  local option = language and language ~= "" and ("language=" .. language) or ""
  return pandoc.RawBlock("latex", "\\begin{SETerminal}[" .. option .. "]\n" .. print_text .. "\n\\end{SETerminal}")
end

function Pandoc(doc)
  if not FORMAT:match("latex") then return doc end
  local volume = doc.meta["volume-number"] and
    tonumber(pandoc.utils.stringify(doc.meta["volume-number"])) or nil
  -- The designed title pages below replace Pandoc's automatic \maketitle.
  doc.meta.title = nil
  doc.meta.subtitle = nil
  doc.meta.author = nil
  doc.meta.date = nil
  if volume then
    -- Volume profiles provide their own front matter. Discard only the shared
    -- website landing page and preserve the first volume-specific \frontmatter
    -- block onward.
    local volume_blocks = pandoc.List()
    local found_volume_frontmatter = false
    for _, block in ipairs(doc.blocks) do
      if block.t == "RawBlock" and block.format:match("latex") and
         block.text:match("\\frontmatter") then
        found_volume_frontmatter = true
      end
      if found_volume_frontmatter then volume_blocks:insert(block) end
    end
    doc.blocks = volume_blocks
  else
    -- Quarto combines every book input into one Pandoc document. Replace the
    -- landing-page blocks before the first chapter H1 with complete-edition
    -- print front matter.
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
  end
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
