local volume = nil

function Meta(meta)
  if meta["volume-number"] then
    volume = tonumber(pandoc.utils.stringify(meta["volume-number"]))
    meta["cover-image"] = nil
    meta.isbn = nil
  end
  return meta
end

function Pandoc(doc)
  if not volume then return doc end
  local input = PANDOC_STATE.input_files[1] or ""
  if input:match("index%.qmd$") then
    doc.blocks = pandoc.Blocks({})
  end
  return doc
end
