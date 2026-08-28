local volume = nil

local function chapter_target(master)
  if volume == 1 then
    if master <= 8 then return nil, master end
    return "Volume II", master - 8
  end
  if volume == 2 then
    if master >= 9 then return nil, master - 8 end
    return "Volume I", master
  end
  return nil, master
end

local function rewrite_chapter_inlines(inlines)
  local out = pandoc.List()
  local i = 1
  while i <= #inlines do
    local current = inlines[i]
    local space = inlines[i + 1]
    local number = inlines[i + 2]
    if current and current.t == "Str" and
      (current.text == "this" or current.text == "This") and
      space and space.t == "Space" and number and number.t == "Str" and
      number.text:match("^[Cc]ourse") then
      out:insert(current)
      out:insert(space)
      out:insert(pandoc.Str(number.text:gsub("^[Cc]ourse", current.text == "This" and "Volume" or "volume")))
      i = i + 3
    elseif current and current.t == "Str" and current.text:lower() == "semester" and
      space and space.t == "Space" and number and number.t == "Str" and
      number.text:lower():match("^project") then
      out:insert(number)
      i = i + 3
    elseif current and current.t == "Str" and current.text:lower():match("^semester%-long") then
      out:insert(pandoc.Str(current.text:gsub("[Ss]emester%-long", "extended")))
      i = i + 1
    elseif current and current.t == "Str" and current.text == "Chapter"
      and space and space.t == "Space" and number and number.t == "Str" then
      local digits, suffix = number.text:match("^(%d+)(.*)$")
      if digits then
        local label, local_number = chapter_target(tonumber(digits))
        if label then
          out:insert(pandoc.Str(label .. ","))
          out:insert(pandoc.Space())
        end
        out:insert(pandoc.Str("Chapter"))
        out:insert(pandoc.Space())
        out:insert(pandoc.Str(tostring(local_number) .. suffix))
        i = i + 3
      else
        out:insert(current)
        i = i + 1
      end
    else
      out:insert(current)
      i = i + 1
    end
  end
  return out
end

function Pandoc(doc)
  if doc.meta["volume-number"] then
    volume = tonumber(pandoc.utils.stringify(doc.meta["volume-number"]))
  end
  if not volume then return doc end
  doc = doc:walk({ Inlines = rewrite_chapter_inlines })
  if volume == 2 then
    doc = doc:walk({
      Para = function(para)
        local text = pandoc.utils.stringify(para)
        if text:match("^Throughout this volume, you.ve learned individual skills and concepts") then
          return pandoc.Para(pandoc.read(
            "Across the software lifecycle, engineering decisions made in requirements, design, version control, and testing shape every production system. This volume concentrates on the next connected responsibilities: CI/CD, data management, cloud deployment, security, maintenance, professional practice, and lifecycle integration.",
            "markdown").blocks[1].content)
        end
        if text:match("^This volume has covered the breadth of software engineering") then
          return pandoc.Para(pandoc.read(
            "This volume has followed software from continuous integration through deployment, operation, security, maintenance, and professional practice. It is now time to synthesize those responsibilities into a coherent lifecycle perspective.",
            "markdown").blocks[1].content)
        end
        return nil
      end
    })
  end
  return doc
end
