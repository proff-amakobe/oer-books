-- Preserve literal content and attach semantic labels without executing examples.
local labels = {
  ['program-code']='Program code (fragment)', ['terminal']='Terminal',
  ['program-output']='Program output', ['configuration']='Configuration',
  ['data']='Data example', ['pseudocode']='Pseudocode',
  ['markdown-example']='Markdown example', ['prompt']='Prompt example',
  ['model-response']='Model response example', ['structured-output']='Structured output',
  ['plain-verbatim']='Verbatim example'
}
function CodeBlock(el)
  if quarto.doc.is_format("pdf") then
    local symbols = {["👍"]="[thumbs up]", ["👎"]="[thumbs down]", ["→"]="->", ["↑"]="[up]", ["↓"]="[down]", ["├"]="|", ["└"]="+", ["─"]="-"}
    for symbol, replacement in pairs(symbols) do el.text = el.text:gsub(symbol, replacement) end
  end
  for _, class in ipairs(el.classes) do
    if labels[class] then
      local items = {pandoc.Para({pandoc.Strong({pandoc.Str(labels[class])})}), el}
      if quarto.doc.is_format('pdf') then
        items = {pandoc.RawBlock('latex', '\\Needspace{4\\baselineskip}'), items[1], pandoc.RawBlock('latex', '\\nopagebreak'), el}
      end
      return pandoc.Div(items,
        pandoc.Attr('', {'technical-example', class}))
    end
  end
end
function Header(el)
  if quarto.doc.is_format('pdf') and pandoc.utils.stringify(el.content):match('^Project Milestone:') then
    return {pandoc.RawBlock('latex', '\\Needspace{20\\baselineskip}'), el}
  end
  if (quarto.doc.is_format('pdf') or quarto.doc.is_format('epub')) and el.classes:includes('unlisted') and pandoc.utils.stringify(el.content) == 'Generative AI' then
    return {}
  end
end

function Div(el)
  if el.classes:includes('web-only') and (quarto.doc.is_format('pdf') or quarto.doc.is_format('epub')) then
    return {}
  end
end
