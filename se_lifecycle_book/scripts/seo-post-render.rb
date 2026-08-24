#!/usr/bin/env ruby

require "pathname"

output = Pathname.new(__dir__).join("..", "_book").expand_path
base = "https://proff-amakobe.github.io/oer-books/se_lifecycle_book/"

output.glob("**/*.html").each do |file|
  next if file.basename.to_s == "404.html"

  html = file.read
  next if html.include?('rel="canonical"')

  relative = file.relative_path_from(output).to_s
  canonical = relative == "index.html" ? base : "#{base}#{relative}"
  html.sub!("</head>", %(<link rel="canonical" href="#{canonical}">\n</head>))
  file.write(html)
end
