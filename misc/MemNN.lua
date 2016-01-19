----NOTE: FIND WHY WITH MEMSIZE = 1 the jointable crashes

require 'nn'
require 'nngraph'
require 'misc.Peek'

local MemNN = {}

function MemNN.build_memory(input_size, output_size, mem_size)
    
  local inputs = {}

  -- each query is of size batch_size x rnn_size
  -- last layer of previous unit
  local query = nn.Identity()()
  table.insert(inputs, query) --insert mem entry one by on 

  --for memories
  local mem_entries = {}
  for i=1,mem_size do
    --each entry is of size batch_size x image_size
    local entry = nn.Identity()()
    table.insert(inputs, entry) --insert mem entry one by one
    -- mem entry is of size batch_size x rnn_size
    local mem_entry = nn.ReLU()(nn.Linear(input_size, output_size)(entry)) --embedding entry in memory

    if i>1 then -- share parameters of each mem_entry with first one
        mem_entry.data.module:share(mem_entries[1].data.module,'weight','bias','gradWeight','gradBias')
    end
    table.insert(mem_entries, mem_entry)
  end


  --create a tensor out of the table of size (mem_size x x batch_size x rnn_size)
  local all_mem_entries = nn.JoinTable(2)(mem_entries)
  --convert this into a 3D of size batch x mem_size x rnn_size
  local mem_entries_3D = nn.View(#mem_entries,-1):setNumInputDims(1)(all_mem_entries)
   --query has batch_size x rnn_size ->   (batch_size x rnn_size x 1)
  local query_3D= nn.View(output_size, -1):setNumInputDims(1)(query)
  --dot product for similarity between query and memories (1 x mem_size)
  local dot_product = nn.MM(false, false)
  -- dot product result (batch_size x mem_size x 1)
  local sims = dot_product({mem_entries_3D, query_3D})
  -- throw the dummy dimension 
  local sims_2D = nn.Select(3,1)(sims)
  -- similarities to attention probabilities (batch_size x mem_size)
  local probs = nn.SoftMax()(sims_2D)
  -- add dummy dimension to convert probs 2D tensor to 3D for MM 
  local probs_3D = nn.View(mem_size,-1):setNumInputDims(1)(nn.Peek()(probs))
  --weighted average
  local weighted_average = nn.MM(true, false)
  -- (batch_size x rnn_size x 1)
  local I = weighted_average({mem_entries_3D, probs_3D})
  --get rid of dummy dimension
  local Im = nn.Select(3,1)(I)

  return nn.gModule(inputs, {Im})
   
end
return MemNN
