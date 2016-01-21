progress_bar = {}

function progress_bar.print(i, m, extra) 
  i = i / m * 100
  m = 100
  for _=1,i do
    io.write("=")
  end
  io.write(">")
  for _=i,m do 
    io.write(".")
  end
  if extra then
    io.write(extra)
  end
  io.write("\27[K\r")
  io.flush()
  
end
