-- Print-only structural transformations. Markdown remains portable for HTML/EPUB.
local stringify = pandoc.utils.stringify

local function latex_inlines(inlines)
  return pandoc.write(pandoc.Pandoc({pandoc.Plain(inlines)}), "latex"):gsub("%s+$", "")
end

local function latex_blocks(blocks)
  return pandoc.write(pandoc.Pandoc(blocks), "latex"):gsub("%s+$", "")
end

local chapters = {
  ["Introduction to Deep Learning"] = {"Foundations of Deep Learning", "Foundations, History, and the Ideas That Changed Everything"},
  ["Neural Network Architectures"] = {"Foundations of Deep Learning", "The Blueprint of Artificial Minds"},
  ["Deep Learning Training Fundamentals"] = {"Foundations of Deep Learning", "How Neural Networks Learn"},
  ["The Architecture of Sight"] = {"Vision Systems", "How Researchers Redesigned the CNN—and What They Built Instead"},
  ["Visual Tasks Beyond Classification"] = {"Vision Systems", "Detection, Segmentation, and the Backbone as Universal Tool"},
  ["Midpoint Integration"] = {"Vision Systems", "Evaluating, Auditing, and Completing the Vision Pipeline"},
  ["The Memory Problem"] = {"Sequence, Language, and Multimodal Learning", "Sequence Models and the Road to Attention"},
  ["The Architecture That Changed Everything"] = {"Sequence, Language, and Multimodal Learning", "Understanding the Transformer"},
  ["Learning from Everything"] = {"Sequence, Language, and Multimodal Learning", "Pre-trained Language Models and the Power of Transfer"},
  ["When Vision Meets Language"] = {"Sequence, Language, and Multimodal Learning", "Multimodal Systems and the Architecture of Shared Understanding"},
  ["Learning to Create"] = {"Generative and Adaptive Systems", "Variational Autoencoders and Generative Adversarial Networks"},
  ["The Art of Controlled Noise"] = {"Generative and Adaptive Systems", "Diffusion Models and Advanced Generative Systems"},
  ["Learning to Act"] = {"Generative and Adaptive Systems", "Reinforcement Learning and the Alignment Problem"},
  ["From Prototype to System"] = {"Engineering and Responsible Deep Learning", "Integration, Deployment, and the Production Gap"},
  ["AI at Scale"] = {"Engineering and Responsible Deep Learning", "Infrastructure, Governance, and Societal Consequence"},
  ["What We Owe Each Other"] = {"Engineering and Responsible Deep Learning", "Ethics, Futures, and the Practitioner’s Responsibility"}
}

local parts = {
  ["Foundations of Deep Learning"] = "Deep learning begins with a deceptively simple idea: layered computation can learn useful representations directly from data.",
  ["Vision Systems"] = "Vision systems transform spatial structure into increasingly abstract feature maps, predictions, and decisions.",
  ["Sequence, Language, and Multimodal Learning"] = "Sequences introduce memory; attention makes relationships directly accessible; multimodal learning asks different representations to share meaning.",
  ["Generative and Adaptive Systems"] = "Deep models can learn distributions, construct new samples, and adapt behavior through feedback.",
  ["Engineering and Responsible Deep Learning"] = "A model becomes consequential when it enters a system: deployment introduces drift, dependencies, scale, governance, and human impact."
}

local skip_sections = {
  ["Deep Learning: A Comprehensive Guide"] = true,
  ["Open Educational Resource"] = true,
  ["Copyright and Publication Information"] = true,
  ["Who This Book Is For"] = true
}

local panel_titles = {
  ["chapter summary"] = "CHAPTER SUMMARY",
  ["review questions"] = "REVIEW QUESTIONS",
  ["discussion questions"] = "DISCUSSION QUESTIONS",
  ["further reading"] = "FURTHER READING"
}

local function section_panel(blocks)
  local out, i = pandoc.Blocks{}, 1
  while i <= #blocks do
    local block = blocks[i]
    if block.t == "Header" and block.level == 2 then
      local plain = stringify(block.content)
      local low = plain:lower()
      local panel = panel_titles[low]
      if low:match("^hands%-on exploration") then panel = "HANDS-ON EXPLORATION" end
      if panel then
        out:insert(pandoc.RawBlock("latex", "\\addcontentsline{toc}{section}{" .. latex_inlines(block.content) .. "}\n\\begin{DLSectionPanel}{" .. panel .. "}"))
        i = i + 1
        while i <= #blocks and not (blocks[i].t == "Header" and blocks[i].level == 2) do
          out:insert(blocks[i]); i = i + 1
        end
        out:insert(pandoc.RawBlock("latex", "\\end{DLSectionPanel}"))
      else
        out:insert(block); i = i + 1
      end
    else
      out:insert(block); i = i + 1
    end
  end
  return out
end

local function technical_figure(div)
  if not FORMAT:match("latex") or not div.classes:includes("technical-figure") then return nil end
  local image, caption
  for _, block in ipairs(div.content) do
    if block.t == "Para" and #block.content == 1 and block.content[1].t == "Image" then image = block.content[1]
    elseif block.t == "Para" then caption = block end
  end
  if not image then return nil end
  local number = div.identifier:match("figure%-(%d+%-%d+)") or ""
  number = number:gsub("%-", ".")
  local target = image.src:gsub("%.svg$", ".pdf")
  local caplatex = caption and latex_blocks({caption}) or ""
  local capplain = caption and stringify(caption) or ""
  capplain = capplain:gsub("^Figure%s+[%d%.]+:%s*", ""):gsub("^Figure%s+[%d%.]+%s*[—%-]%s*", "")
  local tex = table.concat({
    "\\begin{figure}[htbp]",
    "\\centering",
    "\\includegraphics[width=\\linewidth,height=.72\\textheight,keepaspectratio]{" .. target .. "}",
    "\\caption*{" .. caplatex .. "}",
    "\\addcontentsline{lof}{figure}{\\protect\\numberline{" .. number .. "}" .. capplain .. "}",
    "\\label{" .. div.identifier .. "}",
    "\\end{figure}",
    "\\FloatBarrier"
  }, "\n")
  return pandoc.RawBlock("latex", tex)
end

function Div(div)
  return technical_figure(div)
end

function Pandoc(doc)
  if not FORMAT:match("latex") then return doc end

  -- Quarto serializes book parts as raw \part commands rather than level-one
  -- headers. Normalize those commands and their overview prose into designed
  -- part openers before segmenting the remaining chapters.
  local normalized = pandoc.Blocks{}
  local i = 1
  local contents_emitted = false
  while i <= #doc.blocks do
    local block = doc.blocks[i]
    local is_tex = block.t == "RawBlock" and (block.format == "latex" or block.format == "tex")
    local part_title = is_tex and block.text:match("^%s*\\part%{(.-)%}") or nil
    if part_title and parts[part_title] then
      if not contents_emitted then
        normalized:insert(pandoc.RawBlock("latex", "\\DLPrintContents\n\\DLStartMainMatter"))
        contents_emitted = true
      end
      local overview = pandoc.Blocks{}
      i = i + 1
      while i <= #doc.blocks and not (doc.blocks[i].t == "Header" and doc.blocks[i].level == 1) do
        local candidate = doc.blocks[i]
        local plain = stringify(candidate)
        if plain ~= "" and not plain:match("^Part [IVX]+$") and plain ~= parts[part_title] then
          overview:insert(candidate)
        end
        i = i + 1
      end
      normalized:insert(pandoc.RawBlock("latex", "\\DLPartOpener{" .. part_title .. "}{" .. parts[part_title] .. "}{" .. latex_blocks(overview) .. "}"))
    else
      normalized:insert(block)
      i = i + 1
    end
  end
  doc.blocks = normalized

  local starts = {}
  for i, block in ipairs(doc.blocks) do
    if block.t == "Header" and block.level == 1 then starts[#starts + 1] = i end
  end
  local rebuilt = pandoc.Blocks{}
  if starts[1] then
    for prefix = 1, starts[1] - 1 do rebuilt:insert(doc.blocks[prefix]) end
  end

  for section_number, first in ipairs(starts) do
    local last = (starts[section_number + 1] or (#doc.blocks + 1)) - 1
    local heading = doc.blocks[first]
    local title = stringify(heading.content)

    if skip_sections[title] then
      -- Designed title/copyright matter is emitted before the body.
    elseif chapters[title] then
      local objectives_header, objectives_end, objectives_list
      for i = first + 1, last do
        local block = doc.blocks[i]
        if block.t == "Header" and block.level == 2 then
          if stringify(block.content):lower() == "learning objectives" then objectives_header = i
          elseif objectives_header then objectives_end = i - 1; break end
        elseif objectives_header and (block.t == "BulletList" or block.t == "OrderedList") then objectives_list = block end
      end
      objectives_end = objectives_end or last
      local items = {}
      if objectives_list then
        for _, item in ipairs(objectives_list.content) do items[#items + 1] = "\\item " .. latex_blocks(item) end
      else
        items[1] = "\\item Review the chapter learning goals in the digital edition."
      end
      local info = chapters[title]
      rebuilt:insert(pandoc.RawBlock("latex", "\\DLChapterOpener{" .. info[1] .. "}{" .. latex_inlines(heading.content) .. "}{" .. info[2] .. "}{" .. table.concat(items, "\n") .. "}"))
      local body = pandoc.Blocks{}
      for i = first + 1, last do
        local block = doc.blocks[i]
        local remove = (objectives_header and i >= objectives_header and i <= objectives_end) or
          (block.t == "RawBlock" and block.format == "html" and block.text:match("chapter%-part")) or
          ((block.t == "Para" or block.t == "Plain") and stringify(block):match("^Part [IVX]+ ·"))
        if not remove then body:insert(block) end
      end
      body = section_panel(body)
      for _, block in ipairs(body) do rebuilt:insert(block) end
    else
      for i = first, last do rebuilt:insert(doc.blocks[i]) end
    end
  end

  doc.blocks = rebuilt
  return doc
end
