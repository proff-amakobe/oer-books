-- Print-only structural transformations. Source Markdown remains portable.

local stringify = pandoc.utils.stringify

local function latex_inlines(inlines)
  local text = pandoc.write(pandoc.Pandoc({pandoc.Plain(inlines)}), "latex")
  return text:gsub("%s+$", "")
end

local function latex_blocks(blocks)
  local text = pandoc.write(pandoc.Pandoc(blocks), "latex")
  return text:gsub("%s+$", "")
end

local categories = {
  ["Introduction to Artificial Intelligence"] = "Foundations",
  ["Navigating the Space of Possibilities"] = "Search and Problem Solving",
  ["The Language of Thought"] = "Knowledge and Reasoning",
  ["Planning Intelligently"] = "Planning and Decisions",
  ["Reasoning Under Uncertainty"] = "Probabilistic Reasoning",
  ["Learning From Examples"] = "Machine Learning",
  ["Finding Structure in the Dark"] = "Machine Learning",
  ["The Architecture of Learning"] = "Neural Systems",
  ["Machines That Read"] = "Language Intelligence",
  ["Machines That See"] = "Visual Intelligence",
  ["Learning by Doing"] = "Adaptive Systems",
  ["The Return of Rules"] = "Knowledge Systems",
  ["Machines That Create"] = "Generative Systems",
  ["Building AI We Can Live With"] = "Responsible AI",
  ["From Lab to Life"] = "Systems Engineering",
  ["What Comes Next"] = "Future Directions"
}

local function terminal_code(block)
  for _, class in ipairs(block.classes) do
    if class == "bash" or class == "shell" or class == "console" or class == "terminal" then
      return pandoc.RawBlock("latex", "\\begin{EISTerminal}\n" .. block.text .. "\n\\end{EISTerminal}")
    end
  end
  return nil
end

function CodeBlock(block)
  if FORMAT:match("latex") then
    return terminal_code(block)
  end
end

function Pandoc(doc)
  if not FORMAT:match("latex") then
    return doc
  end

  local starts = {}
  for i, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.level == 1 then starts[#starts + 1] = i end
  end

  local rebuilt = pandoc.Blocks{}

  for section_number, first in ipairs(starts) do
    local last = (starts[section_number + 1] or (#doc.blocks + 1)) - 1
    local heading = doc.blocks[first]
    local title = stringify(heading.content)

    if title:match("^Part%s+[IVX]+") then
      local part_title = title:gsub("^Part%s+[IVX]+%s*[—%-]%s*", "")
      local overview = pandoc.Blocks{}
      for i = first + 1, last do overview:insert(doc.blocks[i]) end
      rebuilt:insert(pandoc.RawBlock("latex", "\\EISPartOpener{" .. part_title .. "}{" .. latex_blocks(overview) .. "}"))
    elseif title == "Engineering Intelligent Systems" or title == "Copyright and License" or title == "Copyright and Publication Information" then
      -- The designed copyright page is emitted with the title matter.
    else
      local category = categories[title]
      if category then
        local subtitle, subtitle_index = "", nil
        for i = first + 1, math.min(last, first + 5) do
          if doc.blocks[i].t == "Para" then
            subtitle, subtitle_index = stringify(doc.blocks[i]), i
            break
          end
        end

        local objectives_header, objectives_end, objectives_list
        for i = first + 1, last do
          local block = doc.blocks[i]
          if block.t == "Header" and block.level == 2 then
            if stringify(block.content):lower() == "learning objectives" then
              objectives_header = i
            elseif objectives_header then
              objectives_end = i - 1
              break
            end
          elseif objectives_header and (block.t == "BulletList" or block.t == "OrderedList") then
            objectives_list = block
          end
        end
        objectives_end = objectives_end or last

        local items = {}
        if objectives_list then
          for _, item in ipairs(objectives_list.content) do items[#items + 1] = "\\item " .. latex_blocks(item) end
        else
          items[1] = "\\item " .. subtitle
        end
        rebuilt:insert(pandoc.RawBlock("latex", "\\EISChapterOpener{" .. category .. "}{" .. latex_inlines(heading.content) .. "}{" .. subtitle .. "}{" .. table.concat(items, "\n") .. "}"))

        local premium_quote_done = false
        for i = first + 1, last do
          local block = doc.blocks[i]
          local remove = i == subtitle_index or
            (subtitle_index and i < first + 6 and block.t == "HorizontalRule") or
            (objectives_header and i >= objectives_header and i <= objectives_end)
          if not remove then
            if not premium_quote_done and block.t == "BlockQuote" then
              rebuilt:insert(pandoc.RawBlock("latex", "\\begin{EISPullQuote}\n" .. latex_blocks(block.content) .. "\n\\end{EISPullQuote}"))
              premium_quote_done = true
            else
              rebuilt:insert(block)
            end
          end
        end
      else
        for i = first, last do rebuilt:insert(doc.blocks[i]) end
      end
    end
  end

  doc.blocks = rebuilt
  return doc
end
