function Pandoc(doc)
  if not FORMAT:match("latex") then return doc end
  -- After \clearpage, the page counter points at the next physical page. If
  -- that number is even, the final content page was odd and perfect binding
  -- needs one blank verso. If it is odd, the document is already even.
  doc.blocks:insert(pandoc.RawBlock("latex", "\\clearpage\\ifodd\\value{page}\\relax\\else\\thispagestyle{empty}\\null\\clearpage\\fi"))
  return doc
end
