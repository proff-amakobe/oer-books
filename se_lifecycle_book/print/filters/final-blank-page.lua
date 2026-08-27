function Pandoc(doc)
  if not FORMAT:match("latex") then return doc end
  doc.blocks:insert(pandoc.RawBlock("latex", "\\clearpage\\thispagestyle{empty}\\null"))
  return doc
end
