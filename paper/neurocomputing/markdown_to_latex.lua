-- Conversion helpers for paper/TimeRole_中文主稿.md.
-- The filter preserves the source wording while adapting Markdown conventions
-- to the Elsevier LaTeX manuscript.

function Header(el)
  local first = el.content[1]
  if first and first.t == "Str" and first.text:match("^%d+[%.%d]*$") then
    table.remove(el.content, 1)
    if el.content[1] and el.content[1].t == "Space" then
      table.remove(el.content, 1)
    end
  end
  el.identifier = ""
  el.attributes = {}
  return el
end

function RawInline(el)
  if el.format == "html" and el.text:match("^<u[%s>]") then
    return pandoc.RawInline("latex", "\\underline{")
  end
  if el.format == "html" and el.text == "</u>" then
    return pandoc.RawInline("latex", "}")
  end
  return el
end

function Image(el)
  if el.src == "picture/Fig1_TimeRole_Architecture.png" then
    el.src = "figures/Fig1_TimeRole_Architecture.pdf"
  end
  return el
end
