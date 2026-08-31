-- Render explicit technical-block semantics consistently across formats.

local input_name = (PANDOC_STATE and PANDOC_STATE.input_files and PANDOC_STATE.input_files[1]) or ""
local fixed_chapter = tonumber(input_name:match("chapters[/\\](%d+)[-_]"))
local chapter = fixed_chapter or 0
local algorithm = 0

local semantic_classes = {
  ["program-code"] = true, algorithm = true, terminal = true,
  ["program-output"] = true, configuration = true, ["data-example"] = true,
  ["text-diagram"] = true, ["inline-example"] = true, ["technical-other"] = true,
}

local function has_class(el, wanted)
  for _, class in ipairs(el.classes) do
    if class == wanted then return true end
  end
  return false
end

local function semantic_class(el)
  for _, class in ipairs(el.classes) do
    if semantic_classes[class] then return class end
  end
  return "technical-other"
end

local function language_name(el)
  local names = {python="Python", java="Java", javascript="JavaScript", js="JavaScript",
    c="C", cpp="C++", ["c++"]="C++", bash="Bash", sh="Shell", yaml="YAML",
    yml="YAML", json="JSON", markdown="Markdown", bibtex="BibTeX"}
  for _, class in ipairs(el.classes) do
    if names[class] then return names[class] end
  end
  return nil
end

function Header(el)
  if el.level == 1 and not has_class(el, "unnumbered") then
    local title = pandoc.utils.stringify(el)
    if not title:match("^Part [IVX]+:") then
      chapter = fixed_chapter or (chapter + 1)
      algorithm = 0
    end
  end
end

function CodeBlock(el)
  -- Quarto's self-contained book PDF merge can serialize the landing page's
  -- HTML-only front matter as a literal code block.  It is valid page metadata,
  -- not publication content, so discard only this exact generated signature.
  -- The HTML render consumes the metadata normally and never enters this path.
  if FORMAT:match("latex") and el.text:match("^%-%-%-") and
     el.text:match("number%-sections:%s*false") and
     el.text:match("page%-layout:%s*full") and
     el.text:match("body%-classes:%s*aca%-home") and
     el.text:match("format:%s*\n%s*html:%s*\n%s*toc:%s*false") then
    return {}
  end

  local kind = semantic_class(el)
  local attributed_chapter = tonumber(el.attributes["chapter"] or "")
  if attributed_chapter and attributed_chapter ~= chapter then
    chapter = attributed_chapter
    algorithm = 0
  end
  local lines = 1
  for _ in el.text:gmatch("\n") do lines = lines + 1 end
  local label = nil
  local numbered = false
  if kind == "algorithm" and lines >= 8 then
    algorithm = algorithm + 1
    label = string.format("Algorithm %d.%d", chapter, algorithm)
    numbered = true
    if el.identifier == "" then el.identifier = string.format("alg-%02d-%02d", chapter, algorithm) end
  elseif kind == "terminal" then
    label = "Terminal"
  elseif kind == "program-output" then
    label = "Output"
  elseif kind == "configuration" then
    label = language_name(el) or "Configuration"
  end

  local classes = {"technical-block", kind}
  if lines <= 4 then table.insert(classes, "technical-compact") end
  if numbered then table.insert(classes, "numbered-algorithm") end
  local attrs = {}
  if label then attrs["data-technical-label"] = label end
  local attr = pandoc.Attr("", classes, attrs)

  if FORMAT:match("latex") then
    local env = ({["program-code"]="ACAProgram", algorithm="ACAAlgorithm",
      terminal="ACATerminal", ["program-output"]="ACAOutput",
      configuration="ACAConfig", ["data-example"]="ACAConfig",
      ["text-diagram"]="ACANeutral", ["inline-example"]="ACANeutral",
      ["technical-other"]="ACANeutral"})[kind]
    local safe = (label or ""):gsub("([%%#&{}_])", "\\%1")
    if kind ~= "terminal" and lines > 24 then
      local heading = label or language_name(el) or (kind == "algorithm" and "Algorithm" or "Program Code")
      heading = heading:gsub("([%%#&{}_])", "\\%1")
      return {pandoc.RawBlock("latex", "\\ACALongCodeHeader{" .. heading .. "}"), el}
    end
    if kind == "terminal" then
      local raw = "\\begin{ACATerminal}{" .. safe .. "}\n" ..
        "\\begin{Verbatim}[fontsize=\\fontsize{8.5}{10}\\selectfont,formatcom=\\color{white}]\n" ..
        el.text .. "\n\\end{Verbatim}\n\\end{ACATerminal}\\color{black}"
      return pandoc.RawBlock("latex", raw)
    end
    -- Narrative examples, diagrams, and captured output are not source code.
    -- Render them without syntax-token macros so a token style can never create
    -- the dark-on-dark bars that previously hid mathematical expressions.
    if kind == "program-output" or kind == "data-example" or kind == "text-diagram" or
       kind == "inline-example" or kind == "technical-other" then
      local raw = "\\begin{" .. env .. "}{" .. safe .. "}\n" ..
        "\\begin{Verbatim}[fontsize=\\fontsize{8.5}{10}\\selectfont,formatcom=\\color{ACAPrintInk}]\n" ..
        el.text .. "\n\\end{Verbatim}\n\\end{" .. env .. "}"
      return pandoc.RawBlock("latex", raw)
    end
    return {pandoc.RawBlock("latex", "\\begin{" .. env .. "}{" .. safe .. "}"),
      el, pandoc.RawBlock("latex", "\\end{" .. env .. "}")}
  end

  local content = {}
  if label then
    table.insert(content, pandoc.Plain({pandoc.Span({pandoc.Str(label)},
      pandoc.Attr("", {"technical-label", "visually-distinct-label"}, { ["aria-hidden"]="true" }))}))
  end
  table.insert(content, el)
  return pandoc.Div(content, attr)
end
