-- Publication-only bridge for one authored raw-LaTeX image in non-LaTeX editions.
-- No CodeBlock or Header handler: all manuscript content/classes remain untouched.
function RawBlock(el)
  if el.format == 'tex' or el.format == 'latex' then
    if quarto.doc.is_format('pdf') then
      el.text = el.text:gsub('assets/images/chapters/marcus%.png', 'assets/images/chapters/marcus.jpg')
      return el
    else
      local source = el.text:match('\\includegraphics%b[]{([^}]+)}')
      if source then
        return pandoc.Para({pandoc.Image({}, '/' .. source:gsub('marcus%.png', 'marcus.jpg'))})
      end
    end
  end
end
